"""
Feedback Logger: Systeme de feedback loop pour ameliorer les recherches.

Fonctionnalites:
- Log toutes les recherches avec resultats
- Detecte les recherches sans resultat (gaps)
- Analyse les patterns de recherche
- Suggere des ameliorations
"""

import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class SearchLog:
    """Represente un log de recherche."""

    def __init__(
        self,
        query: str,
        results_count: int,
        top_sources: list[str],
        top_score: float,
        timestamp: Optional[datetime] = None
    ):
        self.query = query
        self.results_count = results_count
        self.top_sources = top_sources
        self.top_score = top_score
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results_count": self.results_count,
            "top_sources": self.top_sources,
            "top_score": self.top_score,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchLog":
        return cls(
            query=data["query"],
            results_count=data["results_count"],
            top_sources=data.get("top_sources", []),
            top_score=data.get("top_score", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


class FeedbackLogger:
    """
    Systeme de feedback loop pour les recherches.

    Analyse:
    - Recherches echouees (0 resultats)
    - Recherches a faible score (resultats peu pertinents)
    - Patterns de recherche frequents
    - Termes populaires non documentes
    """

    # Seuil de score en dessous duquel on considere les resultats comme "faibles"
    LOW_SCORE_THRESHOLD = 0.3

    # Nombre max de logs a conserver
    MAX_LOGS = 1000

    def __init__(self, log_path: Path):
        """
        Args:
            log_path: Chemin vers le fichier de logs JSON
        """
        self.log_path = log_path
        self._logs: list[SearchLog] = []
        self._load_logs()

    def _load_logs(self):
        """Charge les logs existants."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._logs = [SearchLog.from_dict(d) for d in data.get("logs", [])]
                    log(f"  Feedback logs charges: {len(self._logs)} entrees")
            except Exception as e:
                log(f"  Erreur chargement logs: {e}")
                self._logs = []

    def _save_logs(self):
        """Sauvegarde les logs."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            # Garde seulement les MAX_LOGS derniers
            if len(self._logs) > self.MAX_LOGS:
                self._logs = self._logs[-self.MAX_LOGS:]

            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "logs": [l.to_dict() for l in self._logs],
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log(f"  Erreur sauvegarde logs: {e}")

    def log_search(
        self,
        query: str,
        results: list[dict],
    ):
        """
        Log une recherche.

        Args:
            query: Requete de recherche
            results: Liste des resultats [{content, metadata, score}, ...]
        """
        results_count = len(results)
        top_sources = [r["metadata"].get("source", "?") for r in results[:3]]
        top_score = results[0]["score"] if results else 0.0

        search_log = SearchLog(
            query=query,
            results_count=results_count,
            top_sources=top_sources,
            top_score=top_score
        )

        self._logs.append(search_log)
        self._save_logs()

        # Log pour debug
        if results_count == 0:
            log(f"  [FEEDBACK] Recherche echouee: '{query}'")
        elif top_score < self.LOW_SCORE_THRESHOLD:
            log(f"  [FEEDBACK] Recherche faible ({top_score:.2f}): '{query}'")

    def get_failed_searches(self, days: int = 30) -> list[str]:
        """
        Retourne les requetes qui n'ont donne aucun resultat.

        Args:
            days: Nombre de jours a analyser

        Returns:
            Liste des requetes echouees (uniques)
        """
        cutoff = datetime.now() - timedelta(days=days)
        failed = [
            l.query for l in self._logs
            if l.timestamp > cutoff and l.results_count == 0
        ]
        return list(set(failed))

    def get_low_score_searches(self, days: int = 30) -> list[tuple[str, float]]:
        """
        Retourne les requetes avec des scores faibles.

        Args:
            days: Nombre de jours a analyser

        Returns:
            Liste de (query, score) pour les recherches faibles
        """
        cutoff = datetime.now() - timedelta(days=days)
        low_score = [
            (l.query, l.top_score) for l in self._logs
            if l.timestamp > cutoff
            and l.results_count > 0
            and l.top_score < self.LOW_SCORE_THRESHOLD
        ]
        # Deduplique en gardant le score le plus recent
        seen = {}
        for query, score in low_score:
            seen[query] = score
        return list(seen.items())

    def get_popular_queries(self, days: int = 30, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Retourne les requetes les plus frequentes.

        Args:
            days: Nombre de jours a analyser
            top_k: Nombre de resultats

        Returns:
            Liste de (query, count) tries par frequence
        """
        cutoff = datetime.now() - timedelta(days=days)
        queries = [
            l.query.lower() for l in self._logs
            if l.timestamp > cutoff
        ]
        counter = Counter(queries)
        return counter.most_common(top_k)

    def get_popular_sources(self, days: int = 30, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Retourne les sources les plus consultees.

        Args:
            days: Nombre de jours a analyser
            top_k: Nombre de resultats

        Returns:
            Liste de (source, count) tries par frequence
        """
        cutoff = datetime.now() - timedelta(days=days)
        sources = []
        for l in self._logs:
            if l.timestamp > cutoff:
                sources.extend(l.top_sources)

        counter = Counter(sources)
        return counter.most_common(top_k)

    def get_search_gaps(self) -> dict:
        """
        Analyse les "trous" dans la documentation.

        Returns:
            Dict avec:
            - failed_queries: Requetes sans resultat
            - low_quality_queries: Requetes avec mauvais scores
            - suggestions: Suggestions d'amelioration
        """
        failed = self.get_failed_searches()
        low_score = self.get_low_score_searches()

        suggestions = []

        if failed:
            suggestions.append(
                f"Documenter les sujets non couverts: {', '.join(failed[:5])}"
            )

        if low_score:
            weak_topics = [q for q, s in low_score[:5]]
            suggestions.append(
                f"Ameliorer la documentation sur: {', '.join(weak_topics)}"
            )

        return {
            "failed_queries": failed,
            "low_quality_queries": low_score,
            "suggestions": suggestions
        }

    def get_analytics(self) -> dict:
        """
        Retourne des analytics completes.

        Returns:
            Dict avec toutes les stats
        """
        total = len(self._logs)
        if total == 0:
            return {
                "total_searches": 0,
                "success_rate": 0,
                "avg_score": 0,
                "popular_queries": [],
                "popular_sources": [],
                "gaps": {}
            }

        successful = sum(1 for l in self._logs if l.results_count > 0)
        avg_score = sum(l.top_score for l in self._logs if l.results_count > 0) / max(successful, 1)

        return {
            "total_searches": total,
            "success_rate": successful / total * 100,
            "avg_score": avg_score,
            "popular_queries": self.get_popular_queries(),
            "popular_sources": self.get_popular_sources(),
            "gaps": self.get_search_gaps()
        }

    def generate_report(self) -> str:
        """
        Genere un rapport texte des analytics.

        Returns:
            Rapport formate en markdown
        """
        analytics = self.get_analytics()

        report = "# Rapport Feedback RAG\n\n"
        report += f"**Total recherches:** {analytics['total_searches']}\n"
        report += f"**Taux de succes:** {analytics['success_rate']:.1f}%\n"
        report += f"**Score moyen:** {analytics['avg_score']:.2f}\n\n"

        report += "## Requetes populaires\n"
        for query, count in analytics['popular_queries'][:5]:
            report += f"- `{query}`: {count} fois\n"

        report += "\n## Sources populaires\n"
        for source, count in analytics['popular_sources'][:5]:
            report += f"- {source}: {count} fois\n"

        gaps = analytics['gaps']
        if gaps['failed_queries']:
            report += "\n## Recherches echouees\n"
            for query in gaps['failed_queries'][:10]:
                report += f"- `{query}`\n"

        if gaps['suggestions']:
            report += "\n## Suggestions\n"
            for suggestion in gaps['suggestions']:
                report += f"- {suggestion}\n"

        return report
