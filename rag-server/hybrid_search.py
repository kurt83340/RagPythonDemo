"""
Hybrid search : Vector + BM25 avec fusion RRF.
Meilleur recall pour acronymes et termes specifiques.

v3.1.0: Tokenizer ameliore avec stopwords FR/EN
"""
import re
from rank_bm25 import BM25Okapi


# Stopwords francais/anglais pour meilleur recall BM25
STOPWORDS = {
    # Francais
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
    "ce", "cette", "ces", "mon", "ton", "son", "notre", "votre", "leur",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "est", "sont", "etre", "avoir", "fait", "faire", "peut", "doit",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "en",
    "plus", "moins", "tres", "bien", "aussi", "comme", "tout", "tous",
    # Anglais
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "must",
    "and", "or", "but", "if", "then", "else", "when", "where",
    "this", "that", "these", "those", "it", "its",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "not", "no", "yes", "all", "any", "some", "each", "every",
}


class HybridSearcher:
    def __init__(self, documents: list[str]):
        """
        Args:
            documents: Liste des documents indexes
        """
        self.documents = documents
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenization amelioree avec normalisation et filtrage stopwords.

        Args:
            text: Texte a tokenizer

        Returns:
            Liste de tokens filtres
        """
        # Lowercase
        text = text.lower()
        # Garde lettres, chiffres et accents francais
        text = re.sub(r'[^\w\sàâäéèêëïîôùûüçœæ]', ' ', text)
        # Split
        words = text.split()
        # Filtre stopwords et mots trop courts (< 3 chars)
        return [w for w in words if w not in STOPWORDS and len(w) > 2]

    def search_bm25(self, query: str, top_k: int = 15) -> list[tuple[int, float]]:
        """Recherche BM25 (keyword matching) avec tokenization amelioree."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Retourne (index, score) tries
        results = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return results

    @staticmethod
    def reciprocal_rank_fusion(
        rankings: list[list[tuple[int, float]]],
        weights: list[float] = None,
        k: int = 60
    ) -> list[tuple[int, float]]:
        """
        Fusionne plusieurs rankings avec RRF.

        Args:
            rankings: Liste de rankings [(index, score), ...]
            weights: Poids pour chaque ranking (defaut: egaux)
            k: Parametre de lissage RRF (defaut: 60)

        Returns:
            Ranking fusionne [(index, rrf_score), ...]
        """
        if weights is None:
            weights = [1.0] * len(rankings)

        rrf_scores = {}

        for ranking, weight in zip(rankings, weights):
            for rank, (doc_idx, _) in enumerate(ranking):
                if doc_idx not in rrf_scores:
                    rrf_scores[doc_idx] = 0
                rrf_scores[doc_idx] += weight / (k + rank + 1)

        # Trier par score RRF decroissant
        results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return results
