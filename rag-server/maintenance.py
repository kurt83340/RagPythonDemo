"""
Maintenance: Systeme de maintenance intelligente du RAG.

Fonctionnalites:
- Archivage des vieux chunks (reduit leur poids)
- Detection de contradictions entre chunks
- Nettoyage des chunks orphelins/invalides
- Suggestions d'amelioration
"""

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Contradiction:
    """Represente une contradiction detectee."""
    chunk_id_1: str
    chunk_id_2: str
    source_1: str
    source_2: str
    snippet_1: str
    snippet_2: str
    similarity_score: float
    contradiction_type: str  # "version", "date", "fact", "status"
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "chunk_id_1": self.chunk_id_1,
            "chunk_id_2": self.chunk_id_2,
            "source_1": self.source_1,
            "source_2": self.source_2,
            "snippet_1": self.snippet_1[:200],
            "snippet_2": self.snippet_2[:200],
            "similarity_score": self.similarity_score,
            "contradiction_type": self.contradiction_type,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class MaintenanceReport:
    """Rapport de maintenance."""
    archived_chunks: list[str] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    orphan_chunks: list[str] = field(default_factory=list)
    quality_issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    executed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "archived_chunks": self.archived_chunks,
            "contradictions": [c.to_dict() for c in self.contradictions],
            "orphan_chunks": self.orphan_chunks,
            "quality_issues": self.quality_issues,
            "suggestions": self.suggestions,
            "executed_at": self.executed_at.isoformat()
        }


class MaintenanceManager:
    """
    Gestionnaire de maintenance du RAG.

    Taches:
    1. Archivage des vieux contenus (> N jours)
    2. Detection de contradictions (meme sujet, infos differentes)
    3. Nettoyage des chunks orphelins
    4. Analyse de qualite globale
    """

    # Age apres lequel un chunk est considere "vieux" (jours)
    ARCHIVE_AGE_DAYS = 90

    # Seuil de similarite pour detecter des contradictions potentielles
    CONTRADICTION_SIMILARITY_THRESHOLD = 0.75

    # Score de qualite minimum
    MIN_QUALITY_SCORE = 0.3

    def __init__(
        self,
        state_path: Path,
        reranker=None
    ):
        """
        Args:
            state_path: Chemin vers le fichier d'etat de maintenance
            reranker: Cross-encoder pour comparer les chunks
        """
        self.state_path = state_path
        self.reranker = reranker
        self._archived_chunks: set[str] = set()
        self._known_contradictions: list[Contradiction] = []
        self._load_state()

    def _load_state(self):
        """Charge l'etat de maintenance."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._archived_chunks = set(data.get("archived_chunks", []))
                    log(f"  Etat maintenance charge: {len(self._archived_chunks)} archives")
            except Exception as e:
                log(f"  Erreur chargement etat maintenance: {e}")

    def _save_state(self):
        """Sauvegarde l'etat de maintenance."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "archived_chunks": list(self._archived_chunks),
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            log(f"  Erreur sauvegarde etat maintenance: {e}")

    def is_archived(self, chunk_id: str) -> bool:
        """Verifie si un chunk est archive."""
        return chunk_id in self._archived_chunks

    def get_archive_weight(self, chunk_id: str) -> float:
        """
        Retourne le poids d'un chunk pour le scoring.
        Les chunks archives ont un poids reduit.
        """
        if chunk_id in self._archived_chunks:
            return 0.5  # Poids reduit de 50%
        return 1.0

    def identify_old_chunks(
        self,
        chunks: list[dict],
        age_days: Optional[int] = None
    ) -> list[str]:
        """
        Identifie les chunks vieux a archiver.

        Args:
            chunks: Liste de chunks avec metadonnees
            age_days: Age en jours (defaut: ARCHIVE_AGE_DAYS)

        Returns:
            Liste des chunk_ids a archiver
        """
        age_days = age_days or self.ARCHIVE_AGE_DAYS
        cutoff = datetime.now() - timedelta(days=age_days)
        old_chunks = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            indexed_at = chunk.get("metadata", {}).get("indexed_at", "")

            if indexed_at:
                try:
                    chunk_date = datetime.fromisoformat(indexed_at.replace("Z", "+00:00"))
                    if chunk_date < cutoff:
                        old_chunks.append(chunk_id)
                except Exception:
                    pass

            # Verifie aussi les dates dans le contenu (pour progress, bugs, etc.)
            content = chunk.get("content", "")
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
            if date_match:
                try:
                    content_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                    if content_date < cutoff:
                        if chunk_id not in old_chunks:
                            old_chunks.append(chunk_id)
                except Exception:
                    pass

        return old_chunks

    def archive_chunks(self, chunk_ids: list[str]):
        """Archive des chunks (reduit leur poids dans les recherches)."""
        for chunk_id in chunk_ids:
            self._archived_chunks.add(chunk_id)
        self._save_state()
        log(f"  {len(chunk_ids)} chunks archives")

    def unarchive_chunks(self, chunk_ids: list[str]):
        """Desarchive des chunks."""
        for chunk_id in chunk_ids:
            self._archived_chunks.discard(chunk_id)
        self._save_state()

    def detect_contradictions(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ) -> list[Contradiction]:
        """
        Detecte les contradictions potentielles entre chunks.

        Approche:
        1. Trouve les chunks similaires (meme sujet)
        2. Analyse les differences potentielles
        3. Detecte les patterns de contradiction

        Args:
            chunks: Liste de chunks avec contenu et metadonnees
            embeddings: Embeddings correspondants

        Returns:
            Liste de contradictions detectees
        """
        contradictions = []

        if len(chunks) < 2:
            return contradictions

        # Patterns de contradiction
        version_pattern = re.compile(r'v?\d+\.\d+(?:\.\d+)?')
        status_pattern = re.compile(r'\b(termine|fini|done|complete|en cours|todo|pending|bloque|blocked)\b', re.IGNORECASE)
        date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

        # Compare les chunks similaires
        embeddings_array = np.array(embeddings)

        for i, (chunk_i, emb_i) in enumerate(zip(chunks, embeddings)):
            for j, (chunk_j, emb_j) in enumerate(zip(chunks[i+1:], embeddings[i+1:]), start=i+1):
                # Similarite cosine
                similarity = np.dot(emb_i, emb_j) / (
                    np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
                )

                if similarity < self.CONTRADICTION_SIMILARITY_THRESHOLD:
                    continue

                # Chunks similaires trouves, cherche les contradictions
                content_i = chunk_i.get("content", "")
                content_j = chunk_j.get("content", "")

                # 1. Contradiction de version
                versions_i = set(version_pattern.findall(content_i))
                versions_j = set(version_pattern.findall(content_j))
                if versions_i and versions_j and versions_i != versions_j:
                    contradictions.append(Contradiction(
                        chunk_id_1=chunk_i.get("chunk_id", str(i)),
                        chunk_id_2=chunk_j.get("chunk_id", str(j)),
                        source_1=chunk_i.get("metadata", {}).get("source", "?"),
                        source_2=chunk_j.get("metadata", {}).get("source", "?"),
                        snippet_1=content_i[:200],
                        snippet_2=content_j[:200],
                        similarity_score=float(similarity),
                        contradiction_type="version"
                    ))

                # 2. Contradiction de statut
                status_i = set(s.lower() for s in status_pattern.findall(content_i))
                status_j = set(s.lower() for s in status_pattern.findall(content_j))

                conflicting_status = {
                    ("termine", "en cours"), ("done", "pending"), ("complete", "todo"),
                    ("fini", "bloque"), ("done", "blocked")
                }

                for s1 in status_i:
                    for s2 in status_j:
                        if (s1, s2) in conflicting_status or (s2, s1) in conflicting_status:
                            contradictions.append(Contradiction(
                                chunk_id_1=chunk_i.get("chunk_id", str(i)),
                                chunk_id_2=chunk_j.get("chunk_id", str(j)),
                                source_1=chunk_i.get("metadata", {}).get("source", "?"),
                                source_2=chunk_j.get("metadata", {}).get("source", "?"),
                                snippet_1=content_i[:200],
                                snippet_2=content_j[:200],
                                similarity_score=float(similarity),
                                contradiction_type="status"
                            ))
                            break

        return contradictions

    def find_orphan_chunks(
        self,
        chunks: list[dict],
        knowledge_files: list[str]
    ) -> list[str]:
        """
        Trouve les chunks orphelins (fichier source supprime).

        Args:
            chunks: Liste de chunks
            knowledge_files: Liste des fichiers existants

        Returns:
            Liste des chunk_ids orphelins
        """
        orphans = []
        known_sources = set(knowledge_files)

        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "")
            if source and source not in known_sources:
                orphans.append(chunk.get("chunk_id", ""))

        return orphans

    def analyze_quality(
        self,
        chunks: list[dict],
        enrichments: list[dict] = None
    ) -> list[dict]:
        """
        Analyse la qualite globale des chunks.

        Args:
            chunks: Liste de chunks
            enrichments: Enrichissements (avec quality_score)

        Returns:
            Liste des problemes de qualite
        """
        issues = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", str(i))
            content = chunk.get("content", "")

            # Score de qualite si disponible
            if enrichments and i < len(enrichments):
                quality_score = enrichments[i].get("quality_score", 1.0)
                if quality_score < self.MIN_QUALITY_SCORE:
                    issues.append({
                        "chunk_id": chunk_id,
                        "issue": "low_quality",
                        "score": quality_score,
                        "suggestion": "Chunk de faible qualite, envisager une reecriture"
                    })

            # Chunk trop court
            if len(content.split()) < 20:
                issues.append({
                    "chunk_id": chunk_id,
                    "issue": "too_short",
                    "word_count": len(content.split()),
                    "suggestion": "Chunk trop court, fusionner avec un autre"
                })

            # Chunk sans structure
            if len(content) > 500 and not re.search(r'^[#\-\*]', content, re.MULTILINE):
                issues.append({
                    "chunk_id": chunk_id,
                    "issue": "no_structure",
                    "suggestion": "Chunk sans structure, ajouter des headers ou listes"
                })

        return issues

    def generate_suggestions(
        self,
        report: MaintenanceReport,
        feedback_gaps: dict = None
    ) -> list[str]:
        """
        Genere des suggestions d'amelioration.

        Args:
            report: Rapport de maintenance en cours
            feedback_gaps: Gaps du feedback logger

        Returns:
            Liste de suggestions
        """
        suggestions = []

        # Suggestions basees sur les archives
        if len(report.archived_chunks) > 10:
            suggestions.append(
                f"Beaucoup de contenu archive ({len(report.archived_chunks)} chunks). "
                "Envisager un nettoyage des vieux fichiers."
            )

        # Suggestions basees sur les contradictions
        if report.contradictions:
            by_type = defaultdict(list)
            for c in report.contradictions:
                by_type[c.contradiction_type].append(c)

            for ctype, contradictions in by_type.items():
                if len(contradictions) > 2:
                    suggestions.append(
                        f"Plusieurs contradictions de type '{ctype}' detectees. "
                        "Verifier la coherence de la documentation."
                    )

        # Suggestions basees sur les orphelins
        if report.orphan_chunks:
            suggestions.append(
                f"{len(report.orphan_chunks)} chunks orphelins detectes. "
                "Les fichiers sources ont ete supprimes."
            )

        # Suggestions basees sur les gaps du feedback
        if feedback_gaps:
            failed = feedback_gaps.get("failed_queries", [])
            if len(failed) > 5:
                suggestions.append(
                    f"{len(failed)} sujets recherches mais non documentes. "
                    f"Prioritaires: {', '.join(failed[:3])}"
                )

        # Suggestions basees sur la qualite
        low_quality = [i for i in report.quality_issues if i.get("issue") == "low_quality"]
        if len(low_quality) > 5:
            suggestions.append(
                f"{len(low_quality)} chunks de faible qualite. "
                "Envisager une amelioration de la documentation."
            )

        return suggestions

    def run_maintenance(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        knowledge_files: list[str],
        enrichments: list[dict] = None,
        feedback_gaps: dict = None,
        auto_archive: bool = True
    ) -> MaintenanceReport:
        """
        Execute une maintenance complete.

        Args:
            chunks: Liste de chunks
            embeddings: Embeddings
            knowledge_files: Fichiers existants
            enrichments: Enrichissements
            feedback_gaps: Gaps du feedback
            auto_archive: Archive automatiquement les vieux chunks

        Returns:
            Rapport de maintenance
        """
        report = MaintenanceReport()

        log("  [MAINTENANCE] Demarrage...")

        # 1. Identification des vieux chunks
        old_chunks = self.identify_old_chunks(chunks)
        if auto_archive and old_chunks:
            self.archive_chunks(old_chunks)
            report.archived_chunks = old_chunks
            log(f"  [MAINTENANCE] {len(old_chunks)} chunks archives")

        # 2. Detection des contradictions
        report.contradictions = self.detect_contradictions(chunks, embeddings)
        if report.contradictions:
            log(f"  [MAINTENANCE] {len(report.contradictions)} contradictions detectees")

        # 3. Recherche d'orphelins
        report.orphan_chunks = self.find_orphan_chunks(chunks, knowledge_files)
        if report.orphan_chunks:
            log(f"  [MAINTENANCE] {len(report.orphan_chunks)} chunks orphelins")

        # 4. Analyse de qualite
        report.quality_issues = self.analyze_quality(chunks, enrichments)
        if report.quality_issues:
            log(f"  [MAINTENANCE] {len(report.quality_issues)} problemes de qualite")

        # 5. Generation de suggestions
        report.suggestions = self.generate_suggestions(report, feedback_gaps)

        log("  [MAINTENANCE] Termine")
        return report

    def get_maintenance_summary(self, report: MaintenanceReport) -> str:
        """
        Genere un resume lisible du rapport.

        Returns:
            Resume en markdown
        """
        summary = "# Rapport de Maintenance RAG\n\n"
        summary += f"**Execute le:** {report.executed_at.strftime('%Y-%m-%d %H:%M')}\n\n"

        summary += "## Resume\n"
        summary += f"- Chunks archives: {len(report.archived_chunks)}\n"
        summary += f"- Contradictions: {len(report.contradictions)}\n"
        summary += f"- Orphelins: {len(report.orphan_chunks)}\n"
        summary += f"- Problemes qualite: {len(report.quality_issues)}\n\n"

        if report.contradictions:
            summary += "## Contradictions detectees\n"
            for c in report.contradictions[:5]:
                summary += f"- **{c.contradiction_type}**: {c.source_1} vs {c.source_2}\n"
                summary += f"  - `{c.snippet_1[:50]}...`\n"
                summary += f"  - `{c.snippet_2[:50]}...`\n"

        if report.suggestions:
            summary += "\n## Suggestions\n"
            for s in report.suggestions:
                summary += f"- {s}\n"

        return summary
