"""
MCP Server RAG Auto-Ameliorant

Pipeline complet:
- Auto-decouverte des fichiers .md, .pdf, .txt, .json
- File watcher temps reel + rebuild BM25 automatique
- Chunking adaptatif par type de contenu
- Contextual retrieval (methode Anthropic)
- Query expansion (synonymes, acronymes, corrections)
- Hybrid Search (Vector + BM25) -> RRF Fusion -> Cross-Encoder Rerank
- Time-weighted scoring (boost contenu recent)
- Semantic caching (TTL 24h, seuil 0.88)
- Enrichissement automatique (entites, tags, liens)
- Feedback loop (logging + analytics)
- Maintenance intelligente (archivage, contradictions, qualite)
- Highlighting des termes matches
- Tokenizer BM25 ameliore avec stopwords FR/EN
- Hash tracking par chemin relatif (fix sous-dossiers)

Modules v3.1:
- config.py : Configuration centralisee
- utils.py : Helpers partages
- indexing.py : Logique indexation
- search.py : Logique recherche hybride
- tools.py : Definitions outils MCP
"""

# ============================================================
# CONFIGURATION ENVIRONNEMENT
# ============================================================

import os
import sys
import importlib

# Detecte si on est dans Docker ou en local
_in_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", False)

if _in_docker:
    _base_dir = "/app"
else:
    _base_dir = os.path.dirname(__file__)

_models_dir = os.path.join(_base_dir, "models")
os.makedirs(_models_dir, exist_ok=True)

# Configure tous les caches pour le stockage local
os.environ.setdefault("HF_HOME", _models_dir)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _models_dir)
os.environ.setdefault("TORCH_HOME", _models_dir)
os.environ.setdefault("XDG_CACHE_HOME", _models_dir)

# ============================================================
# IMPORTS
# ============================================================

import asyncio
import re
from datetime import datetime
from pathlib import Path

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server.session import ServerSession
from mcp.shared.context import RequestContext
from sentence_transformers import SentenceTransformer

# Modules internes
import config
from chunker import AdaptiveChunker, TokenChunker
from hybrid_search import HybridSearcher
from reranker import Reranker
from auto_indexer import AutoIndexer
from feedback_logger import FeedbackLogger
from enricher import Enricher
from maintenance import MaintenanceManager
from query_expander import QueryExpander, SpellChecker
from contextual_retrieval import ContextualRetrieval, ParentChildChunker
from semantic_cache import SemanticCache, TimeWeightedScorer
from pdf_processor import PDFProcessor, MultiFileProcessor

# ============================================================
# CONFIGURATION
# ============================================================

if _in_docker:
    KNOWLEDGE_PATH = Path("/app/knowledge")
    DB_PATH = Path("/app/db")
    STATE_PATH = Path("/app/db/state")
else:
    KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge"
    DB_PATH = Path(__file__).parent / "db"
    STATE_PATH = Path(__file__).parent / "db" / "state"

# Modele multilingue optimise pour le francais
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Fichiers avec templates (seront crees si absents) - Structure avec dossiers
TEMPLATE_FILES = {
    "projet/project.md": """# Projet

## Vision

[A definir - Decris ton idee en une phrase]

## Contexte

[A definir - Pourquoi ce projet ? Quel probleme resout-il ?]

## Objectifs

[A definir - Qu'est-ce que le projet doit accomplir ?]

## Utilisateurs cibles

[A definir - Qui va utiliser le produit ?]

## Contraintes

[A definir - Budget, delais, technologies imposees, etc.]
""",
    "projet/specs.md": """# Specifications

## User Stories

[A definir apres phase ideation]

## Fonctionnalites MVP

| Feature | Priorite | Status |
|---------|----------|--------|
| | | |

## Fonctionnalites V2+

[A definir plus tard]

## Hors scope

[Ce qu'on ne fait PAS dans ce projet]
""",
    "projet/architecture.md": """# Architecture

## Stack technique

[A definir apres specs]

## Structure du projet

[A definir]

## Decisions techniques

[Historique des choix avec justifications]
""",
    "journal/progress.md": """# Progression

[Les entrees s'ajoutent automatiquement ci-dessous]
""",
    "journal/changelog.md": """# Changelog

[Les versions s'ajoutent automatiquement ci-dessous]
""",
    "journal/bugs.md": """# Bugs resolus

[Les bugs s'ajoutent automatiquement ci-dessous]
""",
    "journal/learnings.md": """# Apprentissages

[Les apprentissages s'ajoutent automatiquement ci-dessous]
"""
}

# Fichiers append-only (utilisent append_doc) - detecte par nom ou dossier
APPEND_ONLY_PATTERNS = ["progress", "changelog", "bugs", "learnings", "journal/"]

# Fichiers editables (utilisent update_doc) - detecte par nom ou dossier
EDITABLE_PATTERNS = ["project", "specs", "architecture", "projet/"]


def log(msg: str):
    """Log vers stderr pour ne pas interferer avec MCP."""
    print(msg, file=sys.stderr, flush=True)


# ============================================================
# INITIALISATION DES COMPOSANTS
# ============================================================

log(f"=== RAG Auto-Ameliorant v{config.VERSION} ===")

log("Chargement du modele d'embedding multilingue...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

log("Chargement du cross-encoder pour reranking...")
reranker = Reranker()

log("Initialisation du chunker adaptatif...")
chunker = AdaptiveChunker(adaptive=True)

log("Initialisation de l'enrichisseur...")
enricher = Enricher(embedder=embedder)

log("Connexion a ChromaDB...")
db_client = chromadb.PersistentClient(path=str(DB_PATH))
collection = db_client.get_or_create_collection(
    name="knowledge",
    metadata={"hnsw:space": "cosine"}
)

log("Initialisation du feedback logger...")
STATE_PATH.mkdir(parents=True, exist_ok=True)
feedback_logger = FeedbackLogger(STATE_PATH / "feedback_logs.json")

log("Initialisation du maintenance manager...")
maintenance_manager = MaintenanceManager(
    state_path=STATE_PATH / "maintenance_state.json",
    reranker=reranker
)

log("Initialisation du query expander...")
query_expander = QueryExpander()
spell_checker = SpellChecker()

log("Initialisation du contextual retrieval...")
contextual_retrieval = ContextualRetrieval()

log("Initialisation du semantic cache...")
semantic_cache = SemanticCache(
    cache_path=STATE_PATH / "semantic_cache.json",
    embedder=embedder,
    ttl_seconds=86400,  # 24 heures (meilleur hit rate entre sessions)
    similarity_threshold=0.88  # Plus permissif pour queries similaires
)

log("Initialisation du time-weighted scorer...")
time_scorer = TimeWeightedScorer(half_life_days=30)

log("Initialisation du multi-file processor...")
multi_processor = MultiFileProcessor()

# Variables globales pour hybrid search
all_documents: list[str] = []
all_metadatas: list[dict] = []
all_ids: list[str] = []
all_embeddings: list[list[float]] = []
hybrid_searcher: HybridSearcher | None = None

# Auto-indexer (initialise plus tard)
auto_indexer: AutoIndexer | None = None

server = Server("project-rag")


# ============================================================
# HELPERS
# ============================================================

def ensure_files_exist():
    """Cree les fichiers templates s'ils n'existent pas."""
    KNOWLEDGE_PATH.mkdir(parents=True, exist_ok=True)

    for filename, template in TEMPLATE_FILES.items():
        filepath = KNOWLEDGE_PATH / filename
        if not filepath.exists():
            # Cree les dossiers parents si necessaire
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(template, encoding="utf-8")
            log(f"  Cree {filename}")


def is_append_only(filename: str) -> bool:
    """Verifie si un fichier est append-only."""
    name_lower = filename.lower()
    return any(pattern in name_lower for pattern in APPEND_ONLY_PATTERNS)


def is_editable(filename: str) -> bool:
    """Verifie si un fichier est editable."""
    name_lower = filename.lower()
    return any(pattern in name_lower for pattern in EDITABLE_PATTERNS)


def get_all_knowledge_files() -> list[str]:
    """Retourne la liste de tous les fichiers supportes dans /knowledge (recursif)."""
    if not KNOWLEDGE_PATH.exists():
        return []

    # Extensions supportees
    supported_extensions = ['.md', '.pdf', '.txt', '.json']
    files = []

    for ext in supported_extensions:
        for f in KNOWLEDGE_PATH.glob(f"**/*{ext}"):
            # Skip les fichiers caches/temporaires
            if not f.name.startswith('.') and not f.name.startswith('_'):
                files.append(str(f.relative_to(KNOWLEDGE_PATH)))

    return sorted(files)


def index_file(filepath: Path) -> int:
    """
    Indexe un fichier dans ChromaDB avec enrichissement.
    Supporte: .md, .pdf, .txt, .json

    Returns:
        Nombre de chunks indexes
    """
    global all_documents, all_metadatas, all_ids, all_embeddings

    # Utilise le chemin relatif comme identifiant (supporte sous-dossiers)
    try:
        filename = str(filepath.relative_to(KNOWLEDGE_PATH))
    except ValueError:
        filename = filepath.name

    # Traite selon le type de fichier
    ext = filepath.suffix.lower()
    if ext == '.pdf':
        # Utilise le multi-processor pour PDF
        text = multi_processor.process(filepath)
        if not text:
            log(f"  Impossible de traiter le PDF: {filename}")
            return 0
    elif ext == '.json':
        text = multi_processor.process(filepath)
        if not text:
            return 0
    else:
        # .md et .txt
        text = filepath.read_text(encoding="utf-8")

    # Chunking adaptatif
    chunks = chunker.chunk_document(text, filename)

    # Contextual retrieval - ajoute le contexte document aux chunks
    doc_context = contextual_retrieval.extract_document_context(text, filename)
    for i, chunk in enumerate(chunks):
        ctx_prefix = contextual_retrieval.generate_contextual_prefix(
            chunk["content"], doc_context, i, len(chunks)
        )
        chunk["contextual_content"] = f"{ctx_prefix}\n\n{chunk['content']}"
        chunk["doc_type"] = doc_context.document_type
        chunk["doc_title"] = doc_context.title

    # Supprime les anciens chunks de ce fichier
    try:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    # Indexe les nouveaux chunks avec enrichissement
    indexed = 0
    for chunk in chunks:
        if not chunk["content"].strip():
            continue

        try:
            # Embedding - utilise le contenu contextuel pour meilleur retrieval
            content_for_embedding = chunk.get("contextual_content", chunk["content"])
            embedding = embedder.encode(content_for_embedding).tolist()

            # Enrichissement
            enrichment = enricher.enrich_chunk(
                text=chunk["content"],
                chunk_id=chunk["chunk_id"],
                metadata=chunk
            )

            # Metadata enrichie
            metadata = {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "token_count": chunk["token_count"],
                "content_type": chunk.get("content_type", "mixed"),
                "context": chunk.get("context", ""),
                "indexed_at": datetime.now().isoformat(),
                # Contextual retrieval
                "doc_type": chunk.get("doc_type", "general"),
                "doc_title": chunk.get("doc_title", ""),
                # Enrichissement
                "tags": ",".join(enrichment.tags[:10]),
                "quality_score": enrichment.quality_score,
                "entities_count": sum(len(v) for v in enrichment.entities.values()),
                # Headers hierarchy (v3.2)
                "h1": chunk.get("h1", ""),
                "h2": chunk.get("h2", ""),
                "h3": chunk.get("h3", ""),
                "header_path": chunk.get("header_path", ""),
            }

            collection.add(
                ids=[chunk["chunk_id"]],
                embeddings=[embedding],
                documents=[chunk["content"]],
                metadatas=[metadata]
            )
            indexed += 1

        except Exception as e:
            log(f"  Erreur indexation chunk: {e}")

    return indexed


def delete_file_from_index(filename: str):
    """Supprime un fichier de l'index."""
    try:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            log(f"  Supprime de l'index: {filename}")
    except Exception as e:
        log(f"  Erreur suppression {filename}: {e}")


def rebuild_hybrid_index():
    """Reconstruit l'index BM25 et les caches pour la recherche hybride."""
    global all_documents, all_metadatas, all_ids, all_embeddings, hybrid_searcher

    try:
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        all_ids = result["ids"]
        all_documents = result["documents"]
        all_metadatas = result["metadatas"]
        all_embeddings = result["embeddings"] if result["embeddings"] is not None else []

        if all_documents:
            hybrid_searcher = HybridSearcher(all_documents)
            log(f"  Index hybride reconstruit: {len(all_documents)} chunks")
        else:
            hybrid_searcher = None

    except Exception as e:
        log(f"  Erreur reconstruction index: {e}")
        hybrid_searcher = None


def index_all_files() -> dict[str, int]:
    """Indexe tous les fichiers et reconstruit l'index hybride."""
    ensure_files_exist()
    stats = {}

    files = get_all_knowledge_files()
    for filename in files:
        filepath = KNOWLEDGE_PATH / filename
        if filepath.exists():
            chunks = index_file(filepath)
            stats[filename] = chunks
            log(f"  {filename}: {chunks} chunks")

    rebuild_hybrid_index()
    return stats


def highlight_matches(text: str, query_terms: list[str], marker: str = "**") -> str:
    """
    Surligne les termes de la query dans le texte.

    Args:
        text: Texte a surligner
        query_terms: Termes a chercher
        marker: Marqueur de surlignage (defaut: **bold**)

    Returns:
        Texte avec termes surlignees
    """
    highlighted = text
    for term in query_terms:
        if len(term) < 3:
            continue
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f"{marker}{term}{marker}", highlighted)
    return highlighted


def hybrid_search(
    query: str,
    n_results: int = 5,
    use_cache: bool = True,
    use_expansion: bool = True,
    use_time_weight: bool = True,
    use_mmr: bool = None,
    mmr_lambda: float = None,
    filters: dict = None
) -> list[dict]:
    """
    Recherche hybride avec toutes les ameliorations v3.2.

    Pipeline:
    1. Check semantic cache
    2. Query expansion (synonymes, acronymes)
    3. Vector search (top 15)
    4. BM25 search (top 15)
    5. RRF Fusion avec poids d'archivage
    6. Cross-encoder rerank + MMR diversite
    7. Time-weighted scoring
    8. Highlighting
    9. Cache results
    10. Return top n_results

    Args:
        query: Requete de recherche
        n_results: Nombre de resultats
        use_cache: Utiliser le cache semantique
        use_expansion: Utiliser l'expansion de query
        use_time_weight: Appliquer le scoring temporel
        use_mmr: Activer MMR pour diversite (defaut: config.MMR_ENABLED_DEFAULT)
        mmr_lambda: Balance pertinence/diversite 0-1 (defaut: config.MMR_LAMBDA_DEFAULT)
        filters: Filtres optionnels {source, date_from, date_to, doc_type, tags}

    Returns:
        Liste de resultats enrichis
    """
    global all_documents, all_metadatas, all_ids, all_embeddings, hybrid_searcher

    # Valeurs par defaut MMR depuis config
    if use_mmr is None:
        use_mmr = config.MMR_ENABLED_DEFAULT
    if mmr_lambda is None:
        mmr_lambda = config.MMR_LAMBDA_DEFAULT

    if collection.count() == 0:
        return []

    # 1. Query expansion (AVANT embedding pour beneficier des termes ajoutes)
    expanded_query = query
    expansion_info = None

    if use_expansion:
        expansion_info = query_expander.expand_query(query)
        expanded_query = expansion_info["expanded"]
        if expansion_info["terms_added"]:
            log(f"  [QUERY] Expansion: +{len(expansion_info['terms_added'])} termes")
        if expansion_info["corrections"]:
            log(f"  [QUERY] Corrections: {expansion_info['corrections']}")

    # 2. Calcul de l'embedding avec la query etendue
    query_embedding = embedder.encode(expanded_query).tolist()

    # Termes pour le highlighting
    query_terms = expanded_query.lower().split()

    # 3. Check semantic cache (avec query originale pour coherence)
    if use_cache:
        cached = semantic_cache.get(query, query_embedding)
        if cached:
            return cached

    n_candidates = 15
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_candidates, collection.count())
    )

    # Convertir en format (index, score) avec poids d'archivage
    vector_ranking = []
    for i, doc_id in enumerate(vector_results["ids"][0]):
        if doc_id in all_ids:
            idx = all_ids.index(doc_id)
            distance = vector_results["distances"][0][i] if vector_results["distances"] else 0
            score = 1 - distance

            # Applique le poids d'archivage
            archive_weight = maintenance_manager.get_archive_weight(doc_id)
            score *= archive_weight

            vector_ranking.append((idx, score))

    # 2. BM25 search (utilise expanded_query)
    bm25_ranking = []
    if hybrid_searcher:
        bm25_results = hybrid_searcher.search_bm25(expanded_query, top_k=n_candidates)
        # Applique le poids d'archivage
        for idx, score in bm25_results:
            if idx < len(all_ids):
                archive_weight = maintenance_manager.get_archive_weight(all_ids[idx])
                bm25_ranking.append((idx, score * archive_weight))

    # 3. RRF Fusion
    rankings = [vector_ranking]
    weights = [0.7]
    if bm25_ranking:
        rankings.append(bm25_ranking)
        weights.append(0.3)

    fused = HybridSearcher.reciprocal_rank_fusion(rankings, weights)

    # Recuperer les documents candidats et leurs embeddings
    candidate_indices = [idx for idx, _ in fused[:n_candidates]]
    candidate_docs = [all_documents[idx] for idx in candidate_indices]
    candidate_metas = [all_metadatas[idx] for idx in candidate_indices]
    candidate_embeddings = [
        all_embeddings[idx] for idx in candidate_indices
        if idx < len(all_embeddings)
    ] if len(all_embeddings) > 0 else []

    if not candidate_docs:
        return []

    # 4. Cross-encoder rerank + MMR diversite
    # NOTE: Utilise query ORIGINALE - l'expansion nuit au cross-encoder (benchmark prouve)
    reranked = reranker.rerank(
        query,
        candidate_docs,
        top_k=n_results * 2,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        embeddings=candidate_embeddings if len(candidate_embeddings) == len(candidate_docs) else None
    )

    # 5. Construire les resultats
    results = []
    for doc, score in reranked:
        idx = candidate_docs.index(doc)
        meta = candidate_metas[idx]
        results.append({
            "content": doc,
            "metadata": meta,
            "score": float(score)
        })

    # 6. Appliquer les filtres si presents
    if filters:
        results = apply_filters(results, filters)

    # 7. Time-weighted scoring
    if use_time_weight:
        results = time_scorer.apply_time_weight(results)

    # 8. Highlighting
    for result in results:
        result["highlighted_content"] = highlight_matches(
            result["content"], query_terms
        )

    # Limite aux n_results demandes
    results = results[:n_results]

    # 9. Cache les resultats
    if use_cache and results:
        semantic_cache.set(query, query_embedding, results)

    return results


def apply_filters(results: list[dict], filters: dict) -> list[dict]:
    """
    Applique des filtres aux resultats.

    Args:
        results: Resultats a filtrer
        filters: {source, date_from, date_to, doc_type, tags}

    Returns:
        Resultats filtres
    """
    filtered = results

    # Filtre par source
    if filters.get("source"):
        source_filter = filters["source"].lower()
        filtered = [
            r for r in filtered
            if source_filter in r["metadata"].get("source", "").lower()
        ]

    # Filtre par type de document
    if filters.get("doc_type"):
        doc_type = filters["doc_type"].lower()
        filtered = [
            r for r in filtered
            if doc_type in r["metadata"].get("doc_type", "").lower()
        ]

    # Filtre par tags
    if filters.get("tags"):
        tag_filter = filters["tags"].lower()
        filtered = [
            r for r in filtered
            if tag_filter in r["metadata"].get("tags", "").lower()
        ]

    # Filtre par date
    if filters.get("date_from") or filters.get("date_to"):
        from datetime import datetime as dt

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        def in_date_range(result):
            indexed_at = result["metadata"].get("indexed_at", "")
            if not indexed_at:
                return True
            try:
                # Parse et normalise sans timezone pour comparaison coherente
                result_date = dt.fromisoformat(indexed_at.replace("Z", "+00:00"))
                result_date = result_date.replace(tzinfo=None)

                if date_from:
                    from_dt = dt.strptime(date_from, "%Y-%m-%d")
                    if result_date < from_dt:
                        return False
                if date_to:
                    # Inclut toute la journee de fin
                    to_dt = dt.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
                    if result_date > to_dt:
                        return False
                return True
            except Exception:
                return True

        filtered = [r for r in filtered if in_date_range(r)]

    return filtered


# ============================================================
# OUTILS MCP
# ============================================================

@server.list_tools()
async def list_tools():
    # Liste dynamique des fichiers pour les enums
    all_files = get_all_knowledge_files()
    append_files = [f for f in all_files if is_append_only(f)]
    edit_files = [f for f in all_files if is_editable(f)]

    return [
        Tool(
            name="search",
            description="Cherche dans la documentation du projet avec recherche hybride (Vector + BM25) et reranking. UTILISE CET OUTIL AVANT de repondre sur les specs, l'architecture, l'avancement, les bugs, ou toute info projet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Mots-cles a chercher (court et precis)"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Nombre de resultats (defaut: 5)",
                        "default": 5
                    },
                    "source": {
                        "type": "string",
                        "description": "Filtrer par fichier source (ex: specs, progress)"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Filtrer par type de document (project, specs, progress, changelog, bugs, learnings)"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filtrer depuis cette date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Filtrer jusqu'a cette date (YYYY-MM-DD)"
                    },
                    "tags": {
                        "type": "string",
                        "description": "Filtrer par tags (ex: api, frontend)"
                    },
                    "boost_recent": {
                        "type": "boolean",
                        "description": "Booster le contenu recent (defaut: true)",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="update_doc",
            description="Met a jour une section d'un fichier de documentation et reindexe. Pour les fichiers editables (projet/project, projet/specs, projet/architecture).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Nom du fichier (ex: projet/project.md)",
                        "enum": edit_files if edit_files else ["projet/project.md", "projet/specs.md", "projet/architecture.md"]
                    },
                    "section": {
                        "type": "string",
                        "description": "Nom de la section a mettre a jour (ex: Vision, Stack technique)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Nouveau contenu de la section"
                    }
                },
                "required": ["file", "section", "content"]
            }
        ),
        Tool(
            name="append_doc",
            description="Ajoute une entree horodatee a un fichier append-only et reindexe. Pour progress, changelog, bugs, learnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Nom du fichier",
                        "enum": append_files if append_files else ["journal/progress.md", "journal/changelog.md", "journal/bugs.md", "journal/learnings.md"]
                    },
                    "title": {
                        "type": "string",
                        "description": "Titre court de l'entree"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu detaille"
                    }
                },
                "required": ["file", "title", "content"]
            }
        ),
        Tool(
            name="reindex",
            description="Force la reindexation de tous les fichiers /knowledge. Utilise apres modification manuelle des fichiers.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_stats",
            description="Affiche les statistiques du RAG (fichiers indexes, nombre de chunks, modeles utilises)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_feedback",
            description="Affiche les analytics du feedback loop (recherches echouees, patterns, suggestions)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="run_maintenance",
            description="Execute la maintenance du RAG (archivage vieux contenus, detection contradictions, analyse qualite)",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_archive": {
                        "type": "boolean",
                        "description": "Archive automatiquement les vieux chunks (defaut: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="log_session",
            description="UTILISE CET OUTIL A LA FIN DE CHAQUE SESSION pour memoriser ce qui a ete fait. Cree un fichier dans sessions/",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Resume court de ce qui a ete fait dans cette session (2-3 phrases)"
                    },
                    "details": {
                        "type": "string",
                        "description": "Details: fichiers modifies, decisions prises, problemes resolus"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "Prochaines etapes a faire (optionnel)",
                        "default": ""
                    }
                },
                "required": ["summary", "details"]
            }
        ),
        # ============================================================
        # OUTILS DE LECTURE / NAVIGATION
        # ============================================================
        Tool(
            name="read_doc",
            description="Lit le contenu complet d'un fichier du RAG. Utile pour voir un fichier entier sans chercher.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier (ex: projet/specs.md, sessions/2025-01-10_03-45.md)"
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="list_files",
            description="Liste les fichiers dans le RAG. Sans argument: tous les fichiers. Avec folder: fichiers du dossier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Dossier a lister (ex: projet, journal, sessions). Vide = tout.",
                        "default": ""
                    }
                }
            }
        ),
        Tool(
            name="get_summary",
            description="Retourne un resume du projet: vision, specs principales, et derniere session. Utile en debut de conversation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        # ============================================================
        # OUTILS DE GESTION DE FICHIERS
        # ============================================================
        Tool(
            name="create_doc",
            description="Cree un nouveau fichier dans /knowledge et l'indexe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin du fichier a creer (ex: references/api.md)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu initial du fichier"
                    }
                },
                "required": ["path", "content"]
            }
        ),
        Tool(
            name="delete_doc",
            description="Supprime un fichier du RAG et de l'index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier a supprimer"
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="move_doc",
            description="Deplace ou renomme un fichier dans le RAG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Chemin actuel du fichier"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Nouveau chemin du fichier"
                    }
                },
                "required": ["source", "destination"]
            }
        ),
        # ============================================================
        # OUTILS DE RECHERCHE AVANCEE
        # ============================================================
        Tool(
            name="search_by_date",
            description="Cherche les documents crees/modifies dans une periode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_date": {
                        "type": "string",
                        "description": "Date debut (YYYY-MM-DD)"
                    },
                    "to_date": {
                        "type": "string",
                        "description": "Date fin (YYYY-MM-DD). Defaut: aujourd'hui.",
                        "default": ""
                    }
                },
                "required": ["from_date"]
            }
        ),
        Tool(
            name="get_related",
            description="Trouve les fichiers/chunks similaires a un fichier donne.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier de reference"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Nombre de resultats (defaut: 5)",
                        "default": 5
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="get_timeline",
            description="Retourne une timeline chronologique des evenements du projet (sessions, progress, bugs, changelog).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre max d'evenements (defaut: 20)",
                        "default": 20
                    }
                }
            }
        ),
        # ============================================================
        # OUTILS EXPORT / IMPORT
        # ============================================================
        Tool(
            name="export_all",
            description="Exporte toute la knowledge base dans un seul fichier markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Nom du fichier de sortie (defaut: knowledge_export.md)",
                        "default": "knowledge_export.md"
                    }
                }
            }
        ),
        Tool(
            name="import_file",
            description="Importe un fichier externe dans /knowledge et l'indexe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Chemin du fichier a importer (absolu ou relatif au projet)"
                    },
                    "dest_folder": {
                        "type": "string",
                        "description": "Dossier destination dans /knowledge (defaut: references)",
                        "default": "references"
                    }
                },
                "required": ["source_path"]
            }
        ),
        Tool(
            name="generate_map",
            description="Genere une carte du projet (_map.md) avec la structure des fichiers et leurs descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Dossier a scanner (defaut: racine du projet)",
                        "default": ""
                    },
                    "extensions": {
                        "type": "string",
                        "description": "Extensions a inclure separees par virgule (defaut: py,js,ts,tsx,jsx,md)",
                        "default": "py,js,ts,tsx,jsx,md"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict, ctx: RequestContext[ServerSession, None] = None):
    global auto_indexer
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")

    # ----------------------------------------------------------
    # SEARCH
    # ----------------------------------------------------------
    if name == "search":
        query = arguments["query"]
        n_results = arguments.get("n_results", 5)
        boost_recent = arguments.get("boost_recent", True)
        use_mmr = arguments.get("use_mmr", None)  # None = use default from config
        mmr_lambda = arguments.get("mmr_lambda", None)  # None = use default from config

        # Construit les filtres
        filters = {}
        if arguments.get("source"):
            filters["source"] = arguments["source"]
        if arguments.get("doc_type"):
            filters["doc_type"] = arguments["doc_type"]
        if arguments.get("date_from"):
            filters["date_from"] = arguments["date_from"]
        if arguments.get("date_to"):
            filters["date_to"] = arguments["date_to"]
        if arguments.get("tags"):
            filters["tags"] = arguments["tags"]

        if collection.count() == 0:
            return [TextContent(
                type="text",
                text="RAG vide. Utilise `reindex` ou ajoute du contenu."
            )]

        try:
            results = hybrid_search(
                query,
                n_results,
                use_cache=True,
                use_expansion=True,
                use_time_weight=boost_recent,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                filters=filters if filters else None
            )

            # Log pour feedback loop
            feedback_logger.log_search(query, results)

            if not results:
                return [TextContent(
                    type="text",
                    text=f"Recherche : \"{query}\"\nAucun resultat trouve."
                )]

            # Formate les resultats
            output = f"Recherche : \"{query}\" (Hybrid + Rerank + v{config.VERSION})\n"

            # Affiche les filtres appliques
            if filters:
                filter_parts = []
                for k, v in filters.items():
                    filter_parts.append(f"{k}={v}")
                output += f"Filtres: {', '.join(filter_parts)}\n"

            output += "\n"

            for i, result in enumerate(results):
                source = result["metadata"].get("source", "?")
                score = result["score"]
                content = result.get("highlighted_content", result["content"])
                tags = result["metadata"].get("tags", "")
                quality = result["metadata"].get("quality_score", 0)
                doc_type = result["metadata"].get("doc_type", "")
                time_weight = result.get("time_weight", 1.0)

                # Indicateur d'archivage
                chunk_id = f"{source}_chunk_{result['metadata'].get('chunk_index', 0)}"
                archived = " [ARCHIVE]" if maintenance_manager.is_archived(chunk_id) else ""

                # Indicateur de boost recent
                boosted = " [RECENT]" if result.get("boosted") else ""

                output += f"### [{i+1}] {source}{archived}{boosted} (score: {score:.3f})\n"
                if doc_type:
                    output += f"Type: {doc_type} | "
                if tags:
                    output += f"Tags: {tags}\n"
                elif doc_type:
                    output += "\n"
                output += f"{content}\n\n---\n\n"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            log(f"Erreur search: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # UPDATE_DOC
    # ----------------------------------------------------------
    elif name == "update_doc":
        filename = arguments["file"]
        section_name = arguments["section"]
        new_content = arguments["content"]

        if not is_editable(filename):
            return [TextContent(
                type="text",
                text=f"{filename} n'est pas editable. Utilise append_doc pour les fichiers append-only."
            )]

        filepath = KNOWLEDGE_PATH / filename

        if not filepath.exists():
            ensure_files_exist()

        try:
            text = filepath.read_text(encoding="utf-8")

            # Trouve et remplace la section
            section_pattern = rf"(## {re.escape(section_name)}\n)(.*?)(?=\n## |\Z)"

            def replacer(match):
                return f"## {section_name}\n{new_content}\n\n"

            new_text, count = re.subn(section_pattern, replacer, text, flags=re.DOTALL)

            if count == 0:
                new_text = text.rstrip() + f"\n\n## {section_name}\n{new_content}\n"

            filepath.write_text(new_text, encoding="utf-8")
            chunks = index_file(filepath)
            rebuild_hybrid_index()

            # Marque comme indexe pour l'auto-indexer
            if auto_indexer:
                auto_indexer.mark_indexed(filepath)

            # Invalide le cache pour ce fichier (queries liees)
            semantic_cache.invalidate(pattern=filename)

            return [TextContent(
                type="text",
                text=f"Mis a jour : {filename} > {section_name}\nReindexe : {chunks} chunks"
            )]

        except Exception as e:
            log(f"Erreur update_doc: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # APPEND_DOC
    # ----------------------------------------------------------
    elif name == "append_doc":
        filename = arguments["file"]
        title = arguments["title"]
        content = arguments["content"]

        if not is_append_only(filename):
            return [TextContent(
                type="text",
                text=f"{filename} n'est pas append-only. Utilise update_doc."
            )]

        filepath = KNOWLEDGE_PATH / filename

        if not filepath.exists():
            ensure_files_exist()

        try:
            entry = f"\n---\n\n## {timestamp} - {title}\n\n{content}\n"

            # Changelog : inserer en haut (apres header) pour ordre chronologique inverse
            if "changelog" in filename.lower():
                existing = filepath.read_text(encoding="utf-8")
                # Trouver le premier --- (fin du header)
                first_sep = existing.find("\n---\n")
                if first_sep != -1:
                    # Inserer apres le premier separateur
                    insert_pos = first_sep + 5  # len("\n---\n")
                    new_content = existing[:insert_pos] + entry + existing[insert_pos:]
                    filepath.write_text(new_content, encoding="utf-8")
                else:
                    # Pas de header, append normal
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write(entry)
            else:
                # Autres fichiers : append normal
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(entry)

            chunks = index_file(filepath)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_indexed(filepath)

            # Invalide le cache pour ce fichier (queries liees)
            semantic_cache.invalidate(pattern=filename)

            return [TextContent(
                type="text",
                text=f"Ajoute a {filename} : {title}\nReindexe : {chunks} chunks"
            )]

        except Exception as e:
            log(f"Erreur append_doc: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # REINDEX
    # ----------------------------------------------------------
    elif name == "reindex":
        try:
            if auto_indexer:
                stats = auto_indexer.force_reindex_all()
            else:
                stats = index_all_files()

            rebuild_hybrid_index()

            # Invalide le cache semantique car l'index a change
            semantic_cache.invalidate()

            # Notifie le client que la liste des ressources a change (MCP 2.1+)
            if ctx and hasattr(ctx, 'session') and ctx.session:
                try:
                    await ctx.session.send_resource_list_changed()
                    log("Notification list_changed envoyee")
                except Exception as notif_err:
                    log(f"Notification list_changed non supportee: {notif_err}")

            output = "Reindexation complete\n\n"
            total = 0
            for filename, count in stats.items():
                if count >= 0:
                    output += f"  {filename}: {count} chunks\n"
                    total += count
                else:
                    output += f"  {filename}: (inchange)\n"

            output += f"\nTotal : {total} chunks indexes"
            output += f"\nModele embedding : {EMBEDDING_MODEL}"
            output += f"\nModele reranking : cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            output += f"\nChunking : Adaptatif + Contextual Retrieval"
            output += f"\nCache semantique : Invalide (reset)"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            log(f"Erreur reindex: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GET_STATS
    # ----------------------------------------------------------
    elif name == "get_stats":
        try:
            # Lire version directement depuis fichier (session MCP persistante)
            version = config.VERSION
            try:
                config_path = Path(__file__).parent / "config.py"
                config_content = config_path.read_text(encoding="utf-8")
                match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', config_content)
                if match:
                    version = match.group(1)
            except Exception:
                pass  # Fallback sur config.VERSION

            total_chunks = collection.count()
            files = get_all_knowledge_files()

            # Compte par fichier
            stats_by_file = {}
            for filename in files:
                try:
                    result = collection.get(where={"source": filename})
                    stats_by_file[filename] = len(result["ids"])
                except Exception:
                    stats_by_file[filename] = 0

            output = f"Statistiques RAG v{version}\n\n"
            output += f"**Version:** {version}\n"
            output += f"**Total chunks indexes:** {total_chunks}\n\n"

            output += "**Modeles:**\n"
            output += f"  Embedding: {EMBEDDING_MODEL}\n"
            output += f"  Reranking: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1\n"
            output += f"  Chunking: Adaptatif (auto-detection)\n\n"

            output += "**Fichiers indexes:**\n"
            for filename, count in stats_by_file.items():
                status = "+" if count > 0 else "-"
                file_type = "append" if is_append_only(filename) else "edit" if is_editable(filename) else "auto"
                output += f"  [{status}] {filename}: {count} ({file_type})\n"

            output += f"\n**Index BM25:** {'Actif' if hybrid_searcher else 'Inactif'}"
            output += f"\n**File watcher:** {'Actif' if auto_indexer else 'Inactif'}"
            output += f"\n**Archives:** {len(maintenance_manager._archived_chunks)} chunks"

            # Stats v3.1.0
            cache_stats = semantic_cache.get_stats()
            output += f"\n\n**=== Features v{version} ===**"
            output += f"\n**Semantic Cache:** {cache_stats.entries_count} entrees, {cache_stats.hit_rate():.1f}% hit rate (TTL: 24h)"
            output += f"\n**Query Expansion:** Actif ({len(query_expander.SYNONYMS)} synonymes)"
            output += f"\n**Contextual Retrieval:** Actif"
            output += f"\n**Time-weighted Scoring:** Actif (demi-vie: {time_scorer.half_life_days}j)"
            output += f"\n**BM25 Tokenizer:** Ameliore (stopwords FR/EN)"
            output += f"\n**PDF Support:** {'Actif' if multi_processor.pdf_processor.is_available() else 'Inactif'}"

            return [TextContent(type="text", text=output)]

        except Exception as e:
            log(f"Erreur get_stats: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GET_FEEDBACK
    # ----------------------------------------------------------
    elif name == "get_feedback":
        try:
            report = feedback_logger.generate_report()
            return [TextContent(type="text", text=report)]

        except Exception as e:
            log(f"Erreur get_feedback: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # READ_DOC
    # ----------------------------------------------------------
    elif name == "read_doc":
        file_path = arguments["file"]
        filepath = KNOWLEDGE_PATH / file_path

        if not filepath.exists():
            return [TextContent(type="text", text=f"Fichier non trouve: {file_path}")]

        try:
            content = filepath.read_text(encoding="utf-8")
            output = f"# {file_path}\n\n{content}"
            return [TextContent(type="text", text=output)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erreur lecture: {str(e)}")]

    # ----------------------------------------------------------
    # LIST_FILES
    # ----------------------------------------------------------
    elif name == "list_files":
        folder = arguments.get("folder", "")

        try:
            if folder:
                search_path = KNOWLEDGE_PATH / folder
                if not search_path.exists():
                    return [TextContent(type="text", text=f"Dossier non trouve: {folder}")]
                files = list(search_path.glob("**/*.md"))
                base = search_path
            else:
                files = list(KNOWLEDGE_PATH.glob("**/*.md"))
                base = KNOWLEDGE_PATH

            # Organise par dossier
            by_folder = {}
            for f in sorted(files):
                rel_path = str(f.relative_to(KNOWLEDGE_PATH))
                parts = rel_path.split("/")
                if len(parts) > 1:
                    folder_name = parts[0]
                else:
                    folder_name = "(racine)"
                if folder_name not in by_folder:
                    by_folder[folder_name] = []
                by_folder[folder_name].append(rel_path)

            output = f"Fichiers dans /knowledge{'/' + folder if folder else ''}\n\n"
            for folder_name, file_list in sorted(by_folder.items()):
                output += f"**{folder_name}/**\n"
                for f in file_list:
                    output += f"  - {f}\n"
                output += "\n"

            output += f"Total: {len(files)} fichiers"
            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GET_SUMMARY
    # ----------------------------------------------------------
    elif name == "get_summary":
        try:
            output = "# Resume du projet\n\n"

            # Vision (projet/project.md)
            filepath = KNOWLEDGE_PATH / "projet/project.md"
            if filepath.exists():
                content = filepath.read_text(encoding="utf-8")
                # Extrait la section Vision
                match = re.search(r'## Vision\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
                if match:
                    output += f"## Vision\n{match.group(1).strip()}\n\n"

            # Specs principales (projet/specs.md)
            for pattern in ["projet/specs.md"]:
                filepath = KNOWLEDGE_PATH / pattern
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    # Extrait MVP
                    match = re.search(r'## Fonctionnalites MVP\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
                    if match:
                        output += f"## Specs MVP\n{match.group(1).strip()}\n\n"
                    break

            # Derniere session
            sessions_dir = KNOWLEDGE_PATH / "sessions"
            if sessions_dir.exists():
                sessions = sorted(sessions_dir.glob("*.md"), reverse=True)
                if sessions:
                    last_session = sessions[0]
                    content = last_session.read_text(encoding="utf-8")
                    output += f"## Derniere session ({last_session.stem})\n{content[:500]}...\n"

            if output == "# Resume du projet\n\n":
                output += "Aucune donnee trouvee. Utilisez update_doc pour ajouter du contenu."

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # CREATE_DOC
    # ----------------------------------------------------------
    elif name == "create_doc":
        file_path = arguments["path"]
        content = arguments["content"]
        filepath = KNOWLEDGE_PATH / file_path

        if filepath.exists():
            return [TextContent(type="text", text=f"Fichier existe deja: {file_path}. Utilisez update_doc.")]

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")

            chunks = index_file(filepath)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_indexed(filepath)

            return [TextContent(type="text", text=f"Cree: {file_path}\nIndexe: {chunks} chunks")]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # DELETE_DOC
    # ----------------------------------------------------------
    elif name == "delete_doc":
        file_path = arguments["file"]
        filepath = KNOWLEDGE_PATH / file_path

        if not filepath.exists():
            return [TextContent(type="text", text=f"Fichier non trouve: {file_path}")]

        try:
            # Supprime de l'index
            delete_file_from_index(file_path)

            # Supprime le fichier
            filepath.unlink()
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_deleted(file_path)

            return [TextContent(type="text", text=f"Supprime: {file_path}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # MOVE_DOC
    # ----------------------------------------------------------
    elif name == "move_doc":
        source = arguments["source"]
        destination = arguments["destination"]
        src_path = KNOWLEDGE_PATH / source
        dest_path = KNOWLEDGE_PATH / destination

        if not src_path.exists():
            return [TextContent(type="text", text=f"Source non trouvee: {source}")]

        if dest_path.exists():
            return [TextContent(type="text", text=f"Destination existe deja: {destination}")]

        try:
            # Supprime l'ancien de l'index
            delete_file_from_index(source)

            # Deplace
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            src_path.rename(dest_path)

            # Indexe le nouveau
            chunks = index_file(dest_path)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_deleted(source)
                auto_indexer.mark_indexed(dest_path)

            return [TextContent(type="text", text=f"Deplace: {source} -> {destination}\nReindexe: {chunks} chunks")]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # SEARCH_BY_DATE
    # ----------------------------------------------------------
    elif name == "search_by_date":
        from_date = arguments["from_date"]
        to_date = arguments.get("to_date", "") or now.strftime("%Y-%m-%d")

        try:
            from datetime import datetime as dt

            from_dt = dt.strptime(from_date, "%Y-%m-%d")
            to_dt = dt.strptime(to_date, "%Y-%m-%d")

            results = []
            for filepath in KNOWLEDGE_PATH.glob("**/*.md"):
                mtime = dt.fromtimestamp(filepath.stat().st_mtime)
                if from_dt <= mtime <= to_dt:
                    rel_path = str(filepath.relative_to(KNOWLEDGE_PATH))
                    results.append((rel_path, mtime.strftime("%Y-%m-%d %H:%M")))

            results.sort(key=lambda x: x[1], reverse=True)

            output = f"Fichiers modifies entre {from_date} et {to_date}\n\n"
            for path, date in results:
                output += f"- {date} : {path}\n"

            if not results:
                output += "Aucun fichier trouve."

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GET_RELATED
    # ----------------------------------------------------------
    elif name == "get_related":
        file_path = arguments["file"]
        n_results = arguments.get("n_results", 5)
        filepath = KNOWLEDGE_PATH / file_path

        if not filepath.exists():
            return [TextContent(type="text", text=f"Fichier non trouve: {file_path}")]

        try:
            content = filepath.read_text(encoding="utf-8")
            # Utilise le contenu comme query
            results = hybrid_search(content[:1000], n_results * 3)

            # Filtre le fichier source et deduplique par fichier (garde meilleur score)
            seen_sources = set()
            filtered = []
            for r in results:
                source = r["metadata"].get("source")
                if source and source != file_path and source not in seen_sources:
                    seen_sources.add(source)
                    filtered.append(r)
                    if len(filtered) >= n_results:
                        break

            output = f"Fichiers similaires a {file_path}\n\n"
            for i, result in enumerate(filtered):
                source = result["metadata"].get("source", "?")
                score = result["score"]
                output += f"{i+1}. {source} (score: {score:.3f})\n"

            if not filtered:
                output += "Aucun fichier similaire trouve."

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GET_TIMELINE
    # ----------------------------------------------------------
    elif name == "get_timeline":
        limit = arguments.get("limit", 20)

        try:
            events = []

            # Sessions
            sessions_dir = KNOWLEDGE_PATH / "sessions"
            if sessions_dir.exists():
                for f in sessions_dir.glob("*.md"):
                    # Parse date from filename: 2025-01-10_03-45.md
                    try:
                        date_str = f.stem.replace("_", " ").replace("-", ":", 2).replace(":", "-", 2)
                        events.append((f.stem, "session", f"sessions/{f.name}"))
                    except:
                        pass

            # Journal (progress, bugs, changelog)
            for journal_file in ["journal/progress.md", "journal/bugs.md", "journal/changelog.md"]:
                filepath = KNOWLEDGE_PATH / journal_file
                if filepath.exists():
                    content = filepath.read_text(encoding="utf-8")
                    # Trouve les entrees ## YYYY-MM-DD
                    for match in re.finditer(r'## (\d{4}-\d{2}-\d{2}[^#\n]*)', content):
                        title = match.group(1).strip()
                        date_part = title[:10]
                        events.append((date_part, journal_file.split("/")[1], title))

            # Trie par date decroissante
            events.sort(key=lambda x: x[0], reverse=True)
            events = events[:limit]

            output = "# Timeline du projet\n\n"
            current_date = ""
            for date, type_, title in events:
                if date[:10] != current_date:
                    current_date = date[:10]
                    output += f"\n## {current_date}\n"
                output += f"- [{type_}] {title}\n"

            if not events:
                output += "Aucun evenement trouve."

            return [TextContent(type="text", text=output)]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # EXPORT_ALL
    # ----------------------------------------------------------
    elif name == "export_all":
        output_file = arguments.get("output_file", "knowledge_export.md")

        try:
            export_content = f"# Export Knowledge Base\n\nExporte le {timestamp}\n\n"
            export_content += "=" * 60 + "\n\n"

            files = sorted(KNOWLEDGE_PATH.glob("**/*.md"))
            for filepath in files:
                rel_path = str(filepath.relative_to(KNOWLEDGE_PATH))
                content = filepath.read_text(encoding="utf-8")
                export_content += f"# FILE: {rel_path}\n\n"
                export_content += content
                export_content += "\n\n" + "-" * 60 + "\n\n"

            # Ecrit dans le dossier parent (pas dans knowledge)
            export_path = KNOWLEDGE_PATH.parent / output_file
            export_path.write_text(export_content, encoding="utf-8")

            return [TextContent(type="text", text=f"Exporte: {output_file}\n{len(files)} fichiers")]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # IMPORT_FILE
    # ----------------------------------------------------------
    elif name == "import_file":
        source_path = arguments["source_path"]
        dest_folder = arguments.get("dest_folder", "references")

        # Resout le chemin source
        src = Path(source_path)
        if not src.is_absolute():
            src = KNOWLEDGE_PATH.parent / source_path

        if not src.exists():
            return [TextContent(type="text", text=f"Fichier source non trouve: {source_path}")]

        try:
            dest_dir = KNOWLEDGE_PATH / dest_folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src.name

            # Copie le contenu
            content = src.read_text(encoding="utf-8")
            dest_path.write_text(content, encoding="utf-8")

            # Indexe
            chunks = index_file(dest_path)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_indexed(dest_path)

            rel_dest = str(dest_path.relative_to(KNOWLEDGE_PATH))
            return [TextContent(type="text", text=f"Importe: {source_path} -> {rel_dest}\nIndexe: {chunks} chunks")]

        except Exception as e:
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # RUN_MAINTENANCE
    # ----------------------------------------------------------
    elif name == "run_maintenance":
        auto_archive = arguments.get("auto_archive", True)

        try:
            # Prepare les donnees pour la maintenance
            chunks_data = []
            for i, (doc, meta, chunk_id) in enumerate(zip(all_documents, all_metadatas, all_ids)):
                chunks_data.append({
                    "content": doc,
                    "metadata": meta,
                    "chunk_id": chunk_id
                })

            # Execute la maintenance
            report = maintenance_manager.run_maintenance(
                chunks=chunks_data,
                embeddings=all_embeddings,
                knowledge_files=get_all_knowledge_files(),
                feedback_gaps=feedback_logger.get_search_gaps(),
                auto_archive=auto_archive
            )

            # Genere le rapport
            summary = maintenance_manager.get_maintenance_summary(report)
            return [TextContent(type="text", text=summary)]

        except Exception as e:
            log(f"Erreur run_maintenance: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # GENERATE_MAP
    # ----------------------------------------------------------
    elif name == "generate_map":
        folder = arguments.get("folder", "")
        extensions = arguments.get("extensions", "py,js,ts,tsx,jsx,md").split(",")
        extensions = [e.strip() for e in extensions]

        try:
            # Dossier a scanner (racine du projet = parent de knowledge)
            project_root = KNOWLEDGE_PATH.parent
            if folder:
                scan_path = project_root / folder
            else:
                scan_path = project_root

            if not scan_path.exists():
                return [TextContent(type="text", text=f"Dossier non trouve: {folder}")]

            # Scan les fichiers
            files_by_folder = {}
            for ext in extensions:
                for filepath in scan_path.glob(f"**/*.{ext}"):
                    # Ignore certains dossiers
                    rel_path = filepath.relative_to(project_root)
                    path_str = str(rel_path)

                    # Skip node_modules, venv, __pycache__, .git, db, models
                    skip_patterns = ['node_modules', 'venv', '__pycache__', '.git', '/db/', '/models/', '\\db\\', '\\models\\']
                    if any(p in path_str for p in skip_patterns):
                        continue

                    # Organise par dossier
                    parts = path_str.replace("\\", "/").split("/")
                    if len(parts) > 1:
                        folder_name = parts[0]
                    else:
                        folder_name = "(racine)"

                    if folder_name not in files_by_folder:
                        files_by_folder[folder_name] = []

                    # Extrait description (premiere ligne de commentaire ou docstring)
                    description = ""
                    try:
                        content = filepath.read_text(encoding="utf-8", errors="ignore")
                        lines = content.split("\n")

                        # Python: docstring ou # comment
                        if ext == "py":
                            for line in lines[:10]:
                                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                                    desc_line = line.strip().strip('"\'')
                                    if desc_line:
                                        description = desc_line[:80]
                                        break
                                elif line.strip().startswith("#") and not line.strip().startswith("#!"):
                                    description = line.strip("#").strip()[:80]
                                    break

                        # JS/TS: // comment ou /* */
                        elif ext in ["js", "ts", "tsx", "jsx"]:
                            for line in lines[:10]:
                                if line.strip().startswith("//"):
                                    description = line.strip("/").strip()[:80]
                                    break
                                elif line.strip().startswith("/*"):
                                    description = line.strip("/*").strip()[:80]
                                    break

                        # Markdown: premier titre ou premiere ligne
                        elif ext == "md":
                            for line in lines[:5]:
                                if line.strip().startswith("#"):
                                    description = line.strip("#").strip()[:80]
                                    break
                                elif line.strip():
                                    description = line.strip()[:80]
                                    break

                    except Exception:
                        pass

                    files_by_folder[folder_name].append({
                        "path": path_str.replace("\\", "/"),
                        "name": filepath.name,
                        "description": description or "(pas de description)"
                    })

            # Genere le contenu markdown
            content = f"# Carte du Projet\n\n"
            content += f"> Generee le {timestamp}\n\n"
            content += f"## Structure\n\n"

            total_files = 0
            for folder_name in sorted(files_by_folder.keys()):
                files = files_by_folder[folder_name]
                total_files += len(files)

                content += f"### {folder_name}/\n\n"
                content += "| Fichier | Description |\n"
                content += "|---------|-------------|\n"

                for f in sorted(files, key=lambda x: x["name"]):
                    content += f"| `{f['name']}` | {f['description']} |\n"

                content += "\n"

            content += f"---\n\n**Total: {total_files} fichiers**\n"

            # Ecrit dans knowledge/codebase/_map.md
            codebase_dir = KNOWLEDGE_PATH / "codebase"
            codebase_dir.mkdir(parents=True, exist_ok=True)
            map_file = codebase_dir / "_map.md"
            map_file.write_text(content, encoding="utf-8")

            # Indexe
            chunks = index_file(map_file)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_indexed(map_file)

            return [TextContent(
                type="text",
                text=f"Carte generee: codebase/_map.md\n{total_files} fichiers scannes\nIndexe: {chunks} chunks"
            )]

        except Exception as e:
            log(f"Erreur generate_map: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # LOG_SESSION
    # ----------------------------------------------------------
    elif name == "log_session":
        summary = arguments["summary"]
        details = arguments["details"]
        next_steps = arguments.get("next_steps", "")

        # Dossier sessions/
        sessions_dir = KNOWLEDGE_PATH / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Nom de fichier unique : YYYY-MM-DD_HH-MM.md
        file_timestamp = now.strftime("%Y-%m-%d_%H-%M")
        session_file = sessions_dir / f"{file_timestamp}.md"

        # Contenu du fichier
        content = f"# Session du {now.strftime('%Y-%m-%d %H:%M')}\n\n"
        content += f"## Resume\n\n{summary}\n\n"
        content += f"## Details\n\n{details}\n"
        if next_steps:
            content += f"\n## Prochaines etapes\n\n{next_steps}\n"

        try:
            # Ecrit le fichier
            session_file.write_text(content, encoding="utf-8")

            # Indexe
            chunks = index_file(session_file)
            rebuild_hybrid_index()

            if auto_indexer:
                auto_indexer.mark_indexed(session_file)

            return [TextContent(
                type="text",
                text=f"Session documentee : sessions/{file_timestamp}.md\nIndexe : {chunks} chunks"
            )]

        except Exception as e:
            log(f"Erreur log_session: {e}")
            return [TextContent(type="text", text=f"Erreur: {str(e)}")]

    # ----------------------------------------------------------
    # UNKNOWN
    # ----------------------------------------------------------
    return [TextContent(type="text", text=f"Outil inconnu : {name}")]


# ============================================================
# MAIN
# ============================================================

async def main():
    global auto_indexer

    log("Demarrage du serveur RAG auto-ameliorant...")
    ensure_files_exist()

    # Initialisation de l'auto-indexer
    log("Initialisation de l'auto-indexer...")
    auto_indexer = AutoIndexer(
        knowledge_path=KNOWLEDGE_PATH,
        state_path=STATE_PATH / "indexer_state.json",
        index_callback=lambda fp: index_file(fp),
        delete_callback=delete_file_from_index,
        rebuild_callback=rebuild_hybrid_index
    )

    # Indexation initiale (intelligente - skip les fichiers inchanges)
    log("Indexation initiale...")
    stats = auto_indexer.initial_index()

    # Compte les fichiers reellement indexes
    indexed_count = sum(1 for v in stats.values() if v > 0)
    skipped_count = sum(1 for v in stats.values() if v == -1)
    total_chunks = sum(v for v in stats.values() if v > 0)

    log(f"  {indexed_count} fichiers indexes, {skipped_count} inchanges")
    log(f"  {total_chunks} chunks total")

    # Reconstruit l'index hybride
    rebuild_hybrid_index()

    # Demarre le file watcher
    log("Demarrage du file watcher...")
    auto_indexer.start_watching()

    log("Serveur pret!")

    # Lance le serveur MCP
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        # Cleanup
        if auto_indexer:
            auto_indexer.stop_watching()
            log("File watcher arrete")


if __name__ == "__main__":
    asyncio.run(main())
