"""
Logique d'indexation des fichiers pour le RAG server.

Contient:
- ensure_files_exist() : Creation templates
- get_all_knowledge_files() : Decouverte fichiers
- index_file() : Indexation d'un fichier
- delete_file_from_index() : Suppression de l'index
- rebuild_hybrid_index() : Reconstruction index BM25
- index_all_files() : Indexation complete
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from config import (
    KNOWLEDGE_PATH,
    TEMPLATE_FILES,
    SUPPORTED_EXTENSIONS,
)
from utils import log

if TYPE_CHECKING:
    from hybrid_search import HybridSearcher

# ============================================================
# VARIABLES GLOBALES (initialisees par init_indexing)
# ============================================================

# Ces variables sont initialisees par server.py via init_indexing()
_embedder = None
_collection = None
_chunker = None
_enricher = None
_contextual_retrieval = None
_multi_processor = None

# Variables pour hybrid search
all_documents: list[str] = []
all_metadatas: list[dict] = []
all_ids: list[str] = []
all_embeddings: list[list[float]] = []
hybrid_searcher: "HybridSearcher | None" = None


def init_indexing(
    embedder,
    collection,
    chunker,
    enricher,
    contextual_retrieval,
    multi_processor
):
    """
    Initialise le module d'indexation avec les composants necessaires.

    Args:
        embedder: SentenceTransformer pour embeddings
        collection: Collection ChromaDB
        chunker: AdaptiveChunker pour chunking
        enricher: Enricher pour enrichissement
        contextual_retrieval: ContextualRetrieval pour contexte
        multi_processor: MultiFileProcessor pour PDF/JSON
    """
    global _embedder, _collection, _chunker, _enricher
    global _contextual_retrieval, _multi_processor

    _embedder = embedder
    _collection = collection
    _chunker = chunker
    _enricher = enricher
    _contextual_retrieval = contextual_retrieval
    _multi_processor = multi_processor


def ensure_files_exist():
    """Cree les fichiers templates s'ils n'existent pas."""
    KNOWLEDGE_PATH.mkdir(parents=True, exist_ok=True)

    for filename, template in TEMPLATE_FILES.items():
        filepath = KNOWLEDGE_PATH / filename
        # Cree le dossier parent si necessaire
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            filepath.write_text(template, encoding="utf-8")
            log(f"  Cree {filename}")


def get_all_knowledge_files() -> list[str]:
    """
    Retourne la liste de tous les fichiers supportes dans /knowledge (recursif).

    Returns:
        Liste des chemins relatifs des fichiers
    """
    if not KNOWLEDGE_PATH.exists():
        return []

    files = []

    for ext in SUPPORTED_EXTENSIONS:
        for f in KNOWLEDGE_PATH.glob(f"**/*{ext}"):
            # Skip les fichiers caches/temporaires
            if not f.name.startswith('.') and not f.name.startswith('_'):
                files.append(str(f.relative_to(KNOWLEDGE_PATH)))

    return sorted(files)


def index_file(filepath: Path) -> int:
    """
    Indexe un fichier dans ChromaDB avec enrichissement.
    Supporte: .md, .pdf, .txt, .json

    Args:
        filepath: Chemin absolu du fichier

    Returns:
        Nombre de chunks indexes
    """
    global all_documents, all_metadatas, all_ids, all_embeddings

    if _collection is None:
        raise RuntimeError("Indexation non initialisee. Appelez init_indexing() d'abord.")

    # Utilise le chemin relatif comme identifiant (supporte sous-dossiers)
    try:
        filename = str(filepath.relative_to(KNOWLEDGE_PATH))
    except ValueError:
        filename = filepath.name

    # Traite selon le type de fichier
    ext = filepath.suffix.lower()
    if ext == '.pdf':
        text = _multi_processor.process(filepath)
        if not text:
            log(f"  Impossible de traiter le PDF: {filename}")
            return 0
    elif ext == '.json':
        text = _multi_processor.process(filepath)
        if not text:
            return 0
    else:
        # .md et .txt
        text = filepath.read_text(encoding="utf-8")

    # Chunking adaptatif
    chunks = _chunker.chunk_document(text, filename)

    # Contextual retrieval - ajoute le contexte document aux chunks
    doc_context = _contextual_retrieval.extract_document_context(text, filename)
    for i, chunk in enumerate(chunks):
        ctx_prefix = _contextual_retrieval.generate_contextual_prefix(
            chunk["content"], doc_context, i, len(chunks)
        )
        chunk["contextual_content"] = f"{ctx_prefix}\n\n{chunk['content']}"
        chunk["doc_type"] = doc_context.document_type
        chunk["doc_title"] = doc_context.title

    # Supprime les anciens chunks de ce fichier
    try:
        existing = _collection.get(where={"source": filename})
        if existing["ids"]:
            _collection.delete(ids=existing["ids"])
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
            embedding = _embedder.encode(content_for_embedding).tolist()

            # Enrichissement
            enrichment = _enricher.enrich_chunk(
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

            _collection.add(
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
    """
    Supprime un fichier de l'index.

    Args:
        filename: Chemin relatif du fichier
    """
    if _collection is None:
        return

    try:
        existing = _collection.get(where={"source": filename})
        if existing["ids"]:
            _collection.delete(ids=existing["ids"])
            log(f"  Supprime de l'index: {filename}")
    except Exception as e:
        log(f"  Erreur suppression {filename}: {e}")


def rebuild_hybrid_index():
    """Reconstruit l'index BM25 et les caches pour la recherche hybride."""
    global all_documents, all_metadatas, all_ids, all_embeddings, hybrid_searcher

    if _collection is None:
        return

    # Import local pour eviter import circulaire
    from hybrid_search import HybridSearcher

    try:
        result = _collection.get(include=["documents", "metadatas", "embeddings"])
        all_ids = result["ids"]
        all_documents = result["documents"]
        all_metadatas = result["metadatas"]
        all_embeddings = result["embeddings"] if result["embeddings"] else []

        if all_documents:
            hybrid_searcher = HybridSearcher(all_documents)
            log(f"  Index hybride reconstruit: {len(all_documents)} chunks")
        else:
            hybrid_searcher = None

    except Exception as e:
        log(f"  Erreur reconstruction index: {e}")
        hybrid_searcher = None


def index_all_files() -> dict[str, int]:
    """
    Indexe tous les fichiers et reconstruit l'index hybride.

    Returns:
        Dict {filename: nb_chunks}
    """
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


def get_collection_count() -> int:
    """Retourne le nombre de documents dans la collection."""
    if _collection is None:
        return 0
    return _collection.count()


def get_embedder():
    """Retourne l'embedder (pour usage externe)."""
    return _embedder
