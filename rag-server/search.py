"""
Logique de recherche hybride pour le RAG server.

Contient:
- hybrid_search() : Recherche complete avec tous les ameliorations v3.2
- apply_filters() : Application des filtres sur les resultats

v3.2: Ajout MMR (Maximal Marginal Relevance) pour diversite des resultats
"""

from datetime import datetime as dt

import config
from hybrid_search import HybridSearcher
from utils import log, highlight_matches

# ============================================================
# VARIABLES GLOBALES (initialisees par init_search)
# ============================================================

_embedder = None
_collection = None
_reranker = None
_query_expander = None
_semantic_cache = None
_time_scorer = None
_maintenance_manager = None

# Import des variables d'indexation
import indexing


def init_search(
    embedder,
    collection,
    reranker,
    query_expander,
    semantic_cache,
    time_scorer,
    maintenance_manager
):
    """
    Initialise le module de recherche avec les composants necessaires.

    Args:
        embedder: SentenceTransformer pour embeddings
        collection: Collection ChromaDB
        reranker: Reranker cross-encoder
        query_expander: QueryExpander pour expansion
        semantic_cache: SemanticCache pour caching
        time_scorer: TimeWeightedScorer pour scoring temporel
        maintenance_manager: MaintenanceManager pour archivage
    """
    global _embedder, _collection, _reranker, _query_expander
    global _semantic_cache, _time_scorer, _maintenance_manager

    _embedder = embedder
    _collection = collection
    _reranker = reranker
    _query_expander = query_expander
    _semantic_cache = semantic_cache
    _time_scorer = time_scorer
    _maintenance_manager = maintenance_manager


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
    if _collection is None or _collection.count() == 0:
        return []

    # Acces aux variables d'indexation
    all_documents = indexing.all_documents
    all_metadatas = indexing.all_metadatas
    all_ids = indexing.all_ids
    all_embeddings = indexing.all_embeddings
    hybrid_searcher = indexing.hybrid_searcher

    # Valeurs par defaut MMR depuis config
    if use_mmr is None:
        use_mmr = config.MMR_ENABLED_DEFAULT
    if mmr_lambda is None:
        mmr_lambda = config.MMR_LAMBDA_DEFAULT

    # 0. Calcul de l'embedding de la query (pour cache)
    query_embedding = _embedder.encode(query).tolist()

    # 1. Check semantic cache
    if use_cache and _semantic_cache:
        cached = _semantic_cache.get(query, query_embedding)
        if cached:
            return cached

    # 2. Query expansion
    expanded_query = query
    query_terms = query.lower().split()

    if use_expansion and _query_expander:
        expansion = _query_expander.expand_query(query)
        expanded_query = expansion["expanded"]
        query_terms.extend(expansion["terms_added"])
        if expansion["corrections"]:
            log(f"  [QUERY] Corrections: {expansion['corrections']}")

    n_candidates = 15
    vector_results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_candidates, _collection.count())
    )

    # Convertir en format (index, score) avec poids d'archivage
    vector_ranking = []
    for i, doc_id in enumerate(vector_results["ids"][0]):
        if doc_id in all_ids:
            idx = all_ids.index(doc_id)
            distance = vector_results["distances"][0][i] if vector_results["distances"] else 0
            score = 1 - distance

            # Applique le poids d'archivage
            if _maintenance_manager:
                archive_weight = _maintenance_manager.get_archive_weight(doc_id)
                score *= archive_weight

            vector_ranking.append((idx, score))

    # 3. BM25 search
    bm25_ranking = []
    if hybrid_searcher:
        bm25_results = hybrid_searcher.search_bm25(query, top_k=n_candidates)
        # Applique le poids d'archivage
        for idx, score in bm25_results:
            if idx < len(all_ids):
                if _maintenance_manager:
                    archive_weight = _maintenance_manager.get_archive_weight(all_ids[idx])
                    bm25_ranking.append((idx, score * archive_weight))
                else:
                    bm25_ranking.append((idx, score))

    # 4. RRF Fusion
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
    ] if all_embeddings else []

    if not candidate_docs:
        return []

    # 5. Cross-encoder rerank + MMR diversite
    reranked = _reranker.rerank(
        query,
        candidate_docs,
        top_k=n_results * 2,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        embeddings=candidate_embeddings if len(candidate_embeddings) == len(candidate_docs) else None
    )

    # 6. Construire les resultats
    results = []
    for doc, score in reranked:
        idx = candidate_docs.index(doc)
        meta = candidate_metas[idx]
        results.append({
            "content": doc,
            "metadata": meta,
            "score": float(score)
        })

    # 7. Appliquer les filtres si presents
    if filters:
        results = apply_filters(results, filters)

    # 8. Time-weighted scoring
    if use_time_weight and _time_scorer:
        results = _time_scorer.apply_time_weight(results)

    # 9. Highlighting
    for result in results:
        result["highlighted_content"] = highlight_matches(
            result["content"], query_terms
        )

    # Limite aux n_results demandes
    results = results[:n_results]

    # 10. Cache les resultats
    if use_cache and _semantic_cache and results:
        _semantic_cache.set(query, query_embedding, results)

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
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")

        def in_date_range(result):
            indexed_at = result["metadata"].get("indexed_at", "")
            if not indexed_at:
                return True
            try:
                # Parse et normalise sans timezone
                result_date = dt.fromisoformat(indexed_at.replace("Z", "+00:00"))
                result_date = result_date.replace(tzinfo=None)

                if date_from:
                    from_dt = dt.strptime(date_from, "%Y-%m-%d")
                    if result_date < from_dt:
                        return False
                if date_to:
                    to_dt = dt.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
                    if result_date > to_dt:
                        return False
                return True
            except Exception:
                return True

        filtered = [r for r in filtered if in_date_range(r)]

    return filtered


def get_semantic_cache():
    """Retourne le cache semantique (pour invalidation externe)."""
    return _semantic_cache


def get_maintenance_manager():
    """Retourne le maintenance manager."""
    return _maintenance_manager
