"""
Semantic Cache: Cache intelligent pour les requetes similaires.

Fonctionnalites:
- Cache les resultats de recherche
- Detecte les requetes semantiquement similaires
- TTL configurable
- Statistiques d'utilisation
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class CacheEntry:
    """Entree de cache."""
    query: str
    query_embedding: list[float]
    results: list[dict]
    created_at: float
    hits: int = 0
    last_hit: float = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_embedding": self.query_embedding,
            "results": self.results,
            "created_at": self.created_at,
            "hits": self.hits,
            "last_hit": self.last_hit
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(
            query=data["query"],
            query_embedding=data["query_embedding"],
            results=data["results"],
            created_at=data["created_at"],
            hits=data.get("hits", 0),
            last_hit=data.get("last_hit", 0)
        )


@dataclass
class CacheStats:
    """Statistiques du cache."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    entries_count: int = 0
    avg_hit_rate: float = 0.0

    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries * 100


class SemanticCache:
    """
    Cache semantique pour les recherches RAG.

    Fonctionnement:
    1. Hash la query et son embedding
    2. Cherche des queries similaires dans le cache
    3. Si similarite > seuil, retourne le cache
    4. Sinon, execute la recherche et cache le resultat
    """

    # Seuil de similarite pour considerer une query comme "identique"
    SIMILARITY_THRESHOLD = 0.92

    # TTL par defaut (1 heure)
    DEFAULT_TTL_SECONDS = 3600

    # Taille max du cache
    MAX_ENTRIES = 500

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        embedder=None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):
        """
        Args:
            cache_path: Chemin pour persister le cache (optionnel)
            embedder: Modele d'embedding (SentenceTransformer)
            ttl_seconds: Duree de vie des entrees
            similarity_threshold: Seuil de similarite
        """
        self.cache_path = cache_path
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold

        self._cache: dict[str, CacheEntry] = {}
        self._stats = CacheStats()

        if cache_path:
            self._load_cache()

    def _load_cache(self):
        """Charge le cache depuis le disque."""
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.get("entries", {}).items():
                        self._cache[key] = CacheEntry.from_dict(entry_data)

                    stats_data = data.get("stats", {})
                    self._stats = CacheStats(
                        total_queries=stats_data.get("total_queries", 0),
                        cache_hits=stats_data.get("cache_hits", 0),
                        cache_misses=stats_data.get("cache_misses", 0)
                    )
                    self._stats.entries_count = len(self._cache)

                    log(f"  Cache charge: {len(self._cache)} entrees")
            except Exception as e:
                log(f"  Erreur chargement cache: {e}")

    def _save_cache(self):
        """Sauvegarde le cache sur le disque."""
        if not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "entries": {k: v.to_dict() for k, v in self._cache.items()},
                "stats": {
                    "total_queries": self._stats.total_queries,
                    "cache_hits": self._stats.cache_hits,
                    "cache_misses": self._stats.cache_misses
                },
                "last_saved": datetime.now().isoformat()
            }

            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"  Erreur sauvegarde cache: {e}")

    def _generate_key(self, query: str) -> str:
        """Genere une cle de cache pour une query."""
        # Normalise la query
        normalized = query.lower().strip()
        return normalized

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Verifie si une entree est expiree."""
        return time.time() - entry.created_at > self.ttl_seconds

    def _find_similar_entry(self, query_embedding: list[float]) -> Optional[CacheEntry]:
        """
        Trouve une entree similaire dans le cache.

        Args:
            query_embedding: Embedding de la query

        Returns:
            CacheEntry si trouvee, None sinon
        """
        if not self._cache:
            return None

        query_vec = np.array(query_embedding)
        best_match = None
        best_similarity = 0

        for entry in self._cache.values():
            if self._is_expired(entry):
                continue

            entry_vec = np.array(entry.query_embedding)

            # Similarite cosine
            similarity = np.dot(query_vec, entry_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-8
            )

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry

        return best_match

    def get(self, query: str, query_embedding: Optional[list[float]] = None) -> Optional[list[dict]]:
        """
        Recupere les resultats du cache si disponibles.

        Args:
            query: Requete de recherche
            query_embedding: Embedding de la query (optionnel)

        Returns:
            Resultats caches ou None
        """
        self._stats.total_queries += 1

        # 1. Cherche par cle exacte
        key = self._generate_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                entry.hits += 1
                entry.last_hit = time.time()
                self._stats.cache_hits += 1
                log(f"  [CACHE HIT] Query exacte: '{query}'")
                return entry.results

        # 2. Cherche par similarite semantique
        if query_embedding:
            similar = self._find_similar_entry(query_embedding)
            if similar:
                similar.hits += 1
                similar.last_hit = time.time()
                self._stats.cache_hits += 1
                log(f"  [CACHE HIT] Query similaire: '{query}' ~ '{similar.query}'")
                return similar.results

        self._stats.cache_misses += 1
        return None

    def set(
        self,
        query: str,
        query_embedding: list[float],
        results: list[dict]
    ):
        """
        Ajoute des resultats au cache.

        Args:
            query: Requete de recherche
            query_embedding: Embedding de la query
            results: Resultats a cacher
        """
        # Nettoie si necessaire
        if len(self._cache) >= self.MAX_ENTRIES:
            self._evict_old_entries()

        key = self._generate_key(query)

        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding,
            results=results,
            created_at=time.time(),
            hits=0,
            last_hit=time.time()
        )

        self._cache[key] = entry
        self._stats.entries_count = len(self._cache)

        # Sauvegarde periodiquement
        if len(self._cache) % 10 == 0:
            self._save_cache()

    def _evict_old_entries(self):
        """Supprime les entrees les plus anciennes."""
        # Trie par dernier acces
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_hit or x[1].created_at
        )

        # Supprime le quart le plus ancien
        to_remove = len(sorted_entries) // 4
        for key, _ in sorted_entries[:to_remove]:
            del self._cache[key]

        log(f"  [CACHE] Eviction: {to_remove} entrees supprimees")

    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalide des entrees du cache.

        Args:
            pattern: Pattern de query a invalider (None = tout)
        """
        if pattern is None:
            self._cache.clear()
            log("  [CACHE] Cache invalide completement")
        else:
            to_remove = [
                key for key, entry in self._cache.items()
                if pattern.lower() in entry.query.lower()
            ]
            for key in to_remove:
                del self._cache[key]
            log(f"  [CACHE] {len(to_remove)} entrees invalidees (pattern: {pattern})")

        self._stats.entries_count = len(self._cache)
        self._save_cache()

    def get_stats(self) -> CacheStats:
        """Retourne les statistiques du cache."""
        self._stats.entries_count = len(self._cache)
        self._stats.avg_hit_rate = self._stats.hit_rate()
        return self._stats

    def cleanup_expired(self):
        """Nettoie les entrees expirees."""
        expired = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]

        for key in expired:
            del self._cache[key]

        if expired:
            log(f"  [CACHE] {len(expired)} entrees expirees supprimees")
            self._save_cache()

    def get_report(self) -> str:
        """Genere un rapport sur le cache."""
        stats = self.get_stats()

        report = "# Rapport Cache Semantique\n\n"
        report += f"**Entrees:** {stats.entries_count}\n"
        report += f"**Requetes totales:** {stats.total_queries}\n"
        report += f"**Cache hits:** {stats.cache_hits}\n"
        report += f"**Cache misses:** {stats.cache_misses}\n"
        report += f"**Taux de hit:** {stats.hit_rate():.1f}%\n\n"

        report += "## Top queries cachees\n"
        top_queries = sorted(
            self._cache.values(),
            key=lambda x: x.hits,
            reverse=True
        )[:10]

        for entry in top_queries:
            report += f"- `{entry.query}`: {entry.hits} hits\n"

        return report


class TimeWeightedScorer:
    """
    Systeme de scoring pondere par le temps.

    Booste le contenu recent, reduit le poids du vieux contenu.
    """

    # Demi-vie en jours (apres ce temps, le score est reduit de 50%)
    HALF_LIFE_DAYS = 30

    def __init__(self, half_life_days: int = HALF_LIFE_DAYS):
        """
        Args:
            half_life_days: Demi-vie pour le decay temporel
        """
        self.half_life_days = half_life_days
        self.decay_rate = np.log(2) / half_life_days

    def compute_time_weight(self, timestamp: datetime) -> float:
        """
        Calcule le poids temporel d'un document.

        Args:
            timestamp: Date du document

        Returns:
            Poids entre 0 et 1
        """
        now = datetime.now()
        age_days = (now - timestamp).days

        # Decay exponentiel
        weight = np.exp(-self.decay_rate * age_days)

        # Minimum 0.1 pour ne pas ignorer completement le vieux contenu
        return max(0.1, weight)

    def apply_time_weight(
        self,
        results: list[dict],
        date_field: str = "indexed_at"
    ) -> list[dict]:
        """
        Applique le poids temporel aux resultats.

        Args:
            results: Resultats de recherche
            date_field: Champ contenant la date

        Returns:
            Resultats avec scores ajustes
        """
        weighted_results = []

        for result in results:
            # Copie le resultat
            weighted = result.copy()

            # Extrait la date
            date_str = result.get("metadata", {}).get(date_field, "")
            if date_str:
                try:
                    timestamp = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    time_weight = self.compute_time_weight(timestamp)

                    # Ajuste le score
                    original_score = result.get("score", 1.0)
                    weighted["score"] = original_score * time_weight
                    weighted["time_weight"] = time_weight
                    weighted["original_score"] = original_score
                except Exception:
                    pass

            weighted_results.append(weighted)

        # Re-trie par score ajuste
        weighted_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return weighted_results

    def boost_recent(
        self,
        results: list[dict],
        boost_days: int = 7,
        boost_factor: float = 1.5
    ) -> list[dict]:
        """
        Booste les resultats recents.

        Args:
            results: Resultats de recherche
            boost_days: Nombre de jours pour le boost
            boost_factor: Facteur de boost

        Returns:
            Resultats boostes
        """
        now = datetime.now()
        cutoff = now - timedelta(days=boost_days)

        boosted_results = []

        for result in results:
            boosted = result.copy()

            date_str = result.get("metadata", {}).get("indexed_at", "")
            if date_str:
                try:
                    timestamp = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if timestamp > cutoff:
                        original_score = result.get("score", 1.0)
                        boosted["score"] = original_score * boost_factor
                        boosted["boosted"] = True
                except Exception:
                    pass

            boosted_results.append(boosted)

        # Re-trie
        boosted_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return boosted_results
