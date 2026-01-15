"""
Cross-encoder reranking avec MMR pour ameliorer precision ET diversite.
Modele multilingue optimise francais.

MMR (Maximal Marginal Relevance):
- Evite les resultats redondants
- Balance pertinence vs diversite via lambda
- lambda=1.0 -> 100% pertinence (pas de MMR)
- lambda=0.7 -> 70% pertinence, 30% diversite (defaut)
"""
import numpy as np
from sentence_transformers import CrossEncoder

import config

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def mmr_select(
    relevance_scores: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int,
    lambda_param: float = 0.7
) -> list[int]:
    """
    Selection MMR (Maximal Marginal Relevance).

    Selectionne iterativement les documents en maximisant:
    score_mmr = lambda * relevance - (1-lambda) * max_similarity(selected)

    Args:
        relevance_scores: Scores cross-encoder [N]
        doc_embeddings: Embeddings documents [N, D]
        top_k: Nombre de documents a selectionner
        lambda_param: Balance pertinence/diversite (0.0-1.0)

    Returns:
        Liste des indices selectionnes dans l'ordre MMR
    """
    if len(relevance_scores) == 0:
        return []

    if len(relevance_scores) <= top_k:
        # Pas assez de documents pour MMR, retourne tous tries par score
        return list(np.argsort(relevance_scores)[::-1])

    selected = []
    remaining = list(range(len(relevance_scores)))

    # Normalise les scores de pertinence a [0, 1]
    score_min = relevance_scores.min()
    score_max = relevance_scores.max()
    if score_max - score_min > 1e-8:
        rel_norm = (relevance_scores - score_min) / (score_max - score_min)
    else:
        rel_norm = np.ones_like(relevance_scores)

    for _ in range(min(top_k, len(remaining))):
        mmr_scores = []

        for idx in remaining:
            # Terme de pertinence
            relevance = rel_norm[idx]

            # Terme de diversite (max similarite avec documents deja selectionnes)
            if selected:
                selected_embs = doc_embeddings[selected]
                doc_emb = doc_embeddings[idx]

                # Similarite cosinus
                norms_selected = np.linalg.norm(selected_embs, axis=1)
                norm_doc = np.linalg.norm(doc_emb)

                if norm_doc > 1e-8:
                    similarities = np.dot(selected_embs, doc_emb) / (
                        norms_selected * norm_doc + 1e-8
                    )
                    max_sim = similarities.max()
                else:
                    max_sim = 0
            else:
                max_sim = 0

            # Score MMR
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append((idx, mmr))

        # Selectionne le meilleur score MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        use_mmr: bool = None,
        mmr_lambda: float = None,
        embeddings: list[list[float]] = None
    ) -> list[tuple[str, float]]:
        """
        Rerank documents par pertinence avec MMR optionnel.

        Args:
            query: Requete utilisateur
            documents: Liste de documents candidats
            top_k: Nombre de resultats a retourner
            use_mmr: Active MMR (defaut: config.MMR_ENABLED_DEFAULT)
            mmr_lambda: Balance pertinence/diversite (defaut: config.MMR_LAMBDA_DEFAULT)
            embeddings: Embeddings des documents pour MMR (requis si use_mmr=True)

        Returns:
            Liste de (document, score) tries par pertinence/diversite
        """
        if not documents:
            return []

        # Valeurs par defaut depuis config
        if use_mmr is None:
            use_mmr = config.MMR_ENABLED_DEFAULT
        if mmr_lambda is None:
            mmr_lambda = config.MMR_LAMBDA_DEFAULT

        # Cross-encoder scoring
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        scores = np.array(scores)

        # Applique MMR si active et embeddings fournis
        if use_mmr and embeddings is not None and len(embeddings) > 0:
            doc_embeddings = np.array(embeddings)
            selected_indices = mmr_select(
                scores,
                doc_embeddings,
                top_k,
                mmr_lambda
            )
            results = [(documents[i], float(scores[i])) for i in selected_indices]
        else:
            # Sans MMR: tri simple par score
            results = list(zip(documents, scores))
            results = sorted(results, key=lambda x: x[1], reverse=True)
            results = results[:top_k]

        return results
