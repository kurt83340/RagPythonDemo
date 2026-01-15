"""
Enricher: Enrichissement automatique des chunks.

Fonctionnalites:
- Extraction d'entites (noms, dates, versions, acronymes)
- Generation de tags/keywords
- Detection de liens entre chunks similaires
- Calcul de metriques de qualite
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class ChunkEnrichment:
    """Enrichissement d'un chunk."""

    # Entites extraites
    entities: dict = field(default_factory=dict)
    # Tags generes
    tags: list[str] = field(default_factory=list)
    # IDs des chunks similaires
    related_chunks: list[str] = field(default_factory=list)
    # Score de qualite (0-1)
    quality_score: float = 0.0
    # Metadonnees additionnelles
    metadata: dict = field(default_factory=dict)


class Enricher:
    """
    Systeme d'enrichissement automatique des chunks.

    Enrichissements:
    1. Extraction d'entites (regex-based)
    2. Generation de tags (TF-IDF-like)
    3. Detection de liens (similarity threshold)
    4. Score de qualite
    """

    # Patterns pour extraction d'entites
    ENTITY_PATTERNS = {
        "version": r"v?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z]+(?:\.\d+)?)?",
        "date": r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?",
        "url": r"https?://[^\s<>\"']+",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "acronym": r"\b[A-Z]{2,6}\b",
        "tech_term": r"\b(?:API|SDK|CLI|UI|UX|DB|SQL|NoSQL|REST|GraphQL|JWT|OAuth|HTTP|HTTPS|JSON|XML|YAML|HTML|CSS|JS|TS|Python|JavaScript|TypeScript|React|Vue|Angular|Node|Docker|K8s|AWS|GCP|Azure)\b",
        "file_path": r"[/\\]?(?:[a-zA-Z0-9_-]+[/\\])+[a-zA-Z0-9_.-]+",
        "code_ref": r"`[^`]+`",
    }

    # Mots vides pour le francais et l'anglais
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

    # Seuil de similarite pour les liens
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, embedder=None):
        """
        Args:
            embedder: Modele d'embedding (SentenceTransformer) pour les liens
        """
        self.embedder = embedder
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE if name not in ["acronym", "tech_term"] else 0)
            for name, pattern in self.ENTITY_PATTERNS.items()
        }

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """
        Extrait les entites d'un texte.

        Args:
            text: Texte a analyser

        Returns:
            Dict {entity_type: [values]}
        """
        entities = {}

        for entity_type, pattern in self._compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Deduplique et trie
                unique_matches = sorted(set(matches))
                entities[entity_type] = unique_matches

        return entities

    def generate_tags(self, text: str, max_tags: int = 10) -> list[str]:
        """
        Genere des tags/keywords a partir du texte.

        Utilise une approche TF simple (frequence des mots).

        Args:
            text: Texte a analyser
            max_tags: Nombre max de tags

        Returns:
            Liste de tags
        """
        # Tokenize et nettoie
        words = re.findall(r'\b[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]{3,}\b', text.lower())

        # Filtre les stopwords
        words = [w for w in words if w not in self.STOPWORDS]

        # Compte les frequences
        counter = Counter(words)

        # Prend les plus frequents
        tags = [word for word, count in counter.most_common(max_tags) if count >= 2]

        # Ajoute les termes techniques detectes
        tech_terms = self.extract_entities(text).get("tech_term", [])
        for term in tech_terms:
            if term.lower() not in [t.lower() for t in tags]:
                tags.append(term)

        return tags[:max_tags]

    def compute_quality_score(self, text: str, metadata: dict = None) -> float:
        """
        Calcule un score de qualite pour un chunk.

        Criteres:
        - Longueur (ni trop court, ni trop long)
        - Presence de structure (headers, listes)
        - Richesse en entites
        - Pas trop de caracteres speciaux

        Args:
            text: Texte du chunk
            metadata: Metadonnees optionnelles

        Returns:
            Score entre 0 et 1
        """
        score = 0.0
        max_score = 5.0

        # 1. Longueur optimale (100-500 mots)
        word_count = len(text.split())
        if 100 <= word_count <= 500:
            score += 1.0
        elif 50 <= word_count < 100 or 500 < word_count <= 1000:
            score += 0.5
        elif word_count < 20:
            score += 0.0
        else:
            score += 0.3

        # 2. Structure (headers, listes)
        has_headers = bool(re.search(r'^#+\s', text, re.MULTILINE))
        has_lists = bool(re.search(r'^[\-\*]\s', text, re.MULTILINE))
        if has_headers:
            score += 0.5
        if has_lists:
            score += 0.5

        # 3. Richesse en entites
        entities = self.extract_entities(text)
        entity_count = sum(len(v) for v in entities.values())
        if entity_count >= 5:
            score += 1.0
        elif entity_count >= 2:
            score += 0.5

        # 4. Ratio texte/code equilibre
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        code_ratio = (code_blocks * 50 + inline_code * 5) / max(word_count, 1)
        if 0.1 <= code_ratio <= 0.5:
            score += 1.0
        elif code_ratio < 0.1 or 0.5 < code_ratio <= 0.8:
            score += 0.5

        # 5. Pas trop de caracteres speciaux/bruit
        special_ratio = len(re.findall(r'[^\w\s.,;:!?\'"-]', text)) / max(len(text), 1)
        if special_ratio < 0.05:
            score += 1.0
        elif special_ratio < 0.1:
            score += 0.5

        return score / max_score

    def find_related_chunks(
        self,
        chunk_embedding: list[float],
        all_embeddings: list[list[float]],
        all_ids: list[str],
        current_id: str,
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Trouve les chunks similaires.

        Args:
            chunk_embedding: Embedding du chunk courant
            all_embeddings: Tous les embeddings de la collection
            all_ids: IDs correspondants
            current_id: ID du chunk courant (a exclure)
            top_k: Nombre de resultats

        Returns:
            Liste de (chunk_id, similarity_score)
        """
        if not all_embeddings or len(all_embeddings) < 2:
            return []

        chunk_vec = np.array(chunk_embedding)
        similarities = []

        for i, (emb, chunk_id) in enumerate(zip(all_embeddings, all_ids)):
            if chunk_id == current_id:
                continue

            other_vec = np.array(emb)
            # Similarite cosine
            similarity = np.dot(chunk_vec, other_vec) / (
                np.linalg.norm(chunk_vec) * np.linalg.norm(other_vec) + 1e-8
            )

            if similarity >= self.SIMILARITY_THRESHOLD:
                similarities.append((chunk_id, float(similarity)))

        # Trie par similarite decroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def enrich_chunk(
        self,
        text: str,
        chunk_id: str,
        embedding: Optional[list[float]] = None,
        all_embeddings: Optional[list[list[float]]] = None,
        all_ids: Optional[list[str]] = None,
        metadata: Optional[dict] = None
    ) -> ChunkEnrichment:
        """
        Enrichit un chunk avec toutes les informations.

        Args:
            text: Contenu du chunk
            chunk_id: ID du chunk
            embedding: Embedding du chunk (pour liens)
            all_embeddings: Tous les embeddings (pour liens)
            all_ids: Tous les IDs (pour liens)
            metadata: Metadonnees existantes

        Returns:
            ChunkEnrichment avec toutes les infos
        """
        enrichment = ChunkEnrichment()

        # 1. Extraction d'entites
        enrichment.entities = self.extract_entities(text)

        # 2. Generation de tags
        enrichment.tags = self.generate_tags(text)

        # 3. Score de qualite
        enrichment.quality_score = self.compute_quality_score(text, metadata)

        # 4. Liens vers chunks similaires
        if embedding and all_embeddings and all_ids:
            related = self.find_related_chunks(
                embedding, all_embeddings, all_ids, chunk_id
            )
            enrichment.related_chunks = [chunk_id for chunk_id, _ in related]
            enrichment.metadata["similarity_scores"] = {
                chunk_id: score for chunk_id, score in related
            }

        # 5. Metadonnees additionnelles
        enrichment.metadata["word_count"] = len(text.split())
        enrichment.metadata["char_count"] = len(text)
        enrichment.metadata["enriched_at"] = datetime.now().isoformat()

        return enrichment

    def batch_enrich(
        self,
        chunks: list[dict],
        embeddings: Optional[list[list[float]]] = None
    ) -> list[ChunkEnrichment]:
        """
        Enrichit plusieurs chunks en batch.

        Args:
            chunks: Liste de {content, chunk_id, ...}
            embeddings: Embeddings correspondants (optionnel)

        Returns:
            Liste de ChunkEnrichment
        """
        enrichments = []
        all_ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

        for i, chunk in enumerate(chunks):
            embedding = embeddings[i] if embeddings else None

            enrichment = self.enrich_chunk(
                text=chunk.get("content", ""),
                chunk_id=chunk.get("chunk_id", str(i)),
                embedding=embedding,
                all_embeddings=embeddings,
                all_ids=all_ids,
                metadata=chunk
            )
            enrichments.append(enrichment)

        return enrichments


class ContentAnalyzer:
    """
    Analyse le contenu pour determiner le type et adapter le traitement.
    """

    @staticmethod
    def detect_content_type(text: str) -> str:
        """
        Detecte le type de contenu.

        Returns:
            Un de: "code", "documentation", "changelog", "config", "narrative"
        """
        # Ratio de code
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        lines = text.count('\n') + 1

        code_ratio = (code_blocks * 10 + inline_code) / max(lines, 1)

        if code_ratio > 2:
            return "code"

        # Patterns specifiques
        if re.search(r'^#+\s*v?\d+\.\d+', text, re.MULTILINE):
            return "changelog"

        if re.search(r'^(FROM|ENV|RUN|CMD|EXPOSE)\s', text, re.MULTILINE):
            return "config"

        if re.search(r'^[a-zA-Z_]+\s*[=:]\s*', text, re.MULTILINE):
            return "config"

        if re.search(r'##\s*(Vision|Objectif|Context|Architecture)', text, re.IGNORECASE):
            return "documentation"

        return "narrative"

    @staticmethod
    def get_optimal_chunk_size(content_type: str) -> tuple[int, int]:
        """
        Retourne la taille de chunk optimale pour un type de contenu.

        Returns:
            (chunk_size, overlap)
        """
        sizes = {
            "code": (256, 50),          # Petits chunks pour code
            "documentation": (512, 100), # Standard
            "changelog": (384, 75),      # Medium
            "config": (256, 25),         # Petits, peu d'overlap
            "narrative": (768, 150),     # Plus grands pour texte
        }
        return sizes.get(content_type, (512, 100))
