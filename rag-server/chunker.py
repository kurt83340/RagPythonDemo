"""
Chunker adaptatif: Chunking intelligent qui s'adapte au type de contenu.

Fonctionnalites:
- Detection automatique du type de contenu
- Taille de chunk adaptee (code vs documentation vs changelog)
- Preservation du contexte semantique
- Overlap intelligent
- Extraction hierarchie headers markdown (v3.2)
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# HEADER EXTRACTION (v3.2)
# ============================================================

class MarkdownHeaderExtractor:
    """
    Extrait la hierarchie des headers markdown.

    Permet de connaitre le contexte h1/h2/h3 pour chaque position
    dans un document, utile pour enrichir les metadata des chunks.
    """

    def __init__(self):
        # Pattern pour detecter les headers markdown (# to ######)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def extract_all_headers(self, text: str) -> list[dict]:
        """
        Extrait tous les headers avec leur position.

        Args:
            text: Document complet

        Returns:
            Liste de {level, text, start, end}
        """
        headers = []
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))  # Nombre de #
            headers.append({
                "level": level,
                "text": match.group(2).strip(),
                "start": match.start(),
                "end": match.end()
            })
        return headers

    def get_header_hierarchy(self, text: str, position: int) -> dict:
        """
        Retourne les headers parents pour une position donnee.

        Args:
            text: Document complet
            position: Position du chunk dans le texte

        Returns:
            {
                "h1": "Titre principal",
                "h2": "Section",
                "h3": "Sous-section",
                "header_path": "Titre > Section > Sous-section"
            }
        """
        headers = self.extract_all_headers(text)

        # Ne garde que les headers AVANT la position
        relevant = [h for h in headers if h["start"] < position]

        # Construit la hierarchie (garde le dernier de chaque niveau)
        hierarchy = {}
        for h in relevant:
            level = h["level"]
            hierarchy[level] = h["text"]
            # Efface les niveaux inferieurs quand un niveau superieur apparait
            for lower_level in range(level + 1, 7):
                hierarchy.pop(lower_level, None)

        result = {
            "h1": hierarchy.get(1, ""),
            "h2": hierarchy.get(2, ""),
            "h3": hierarchy.get(3, ""),
        }

        # Construit le breadcrumb path
        parts = [v for v in [result["h1"], result["h2"], result["h3"]] if v]
        result["header_path"] = " > ".join(parts) if parts else ""

        return result

    def get_chunk_headers(self, full_text: str, chunk_text: str, chunk_start: int = None) -> dict:
        """
        Extrait les headers pour un chunk donne.

        Args:
            full_text: Document complet
            chunk_text: Texte du chunk
            chunk_start: Position du chunk (si None, cherche dans le texte)

        Returns:
            Dict avec h1, h2, h3, header_path
        """
        if chunk_start is None:
            # Cherche la position du chunk dans le document
            chunk_start = full_text.find(chunk_text)
            if chunk_start == -1:
                chunk_start = 0

        return self.get_header_hierarchy(full_text, chunk_start)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class ContentType(Enum):
    """Types de contenu detectables."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CHANGELOG = "changelog"
    CONFIG = "config"
    NARRATIVE = "narrative"
    MIXED = "mixed"


@dataclass
class ChunkConfig:
    """Configuration de chunking par type de contenu."""
    chunk_size: int
    chunk_overlap: int
    separators: list[str]

    @classmethod
    def for_content_type(cls, content_type: ContentType) -> "ChunkConfig":
        """Retourne la config optimale pour un type de contenu."""
        configs = {
            ContentType.CODE: cls(
                chunk_size=256,
                chunk_overlap=50,
                separators=[
                    "\n\nclass ", "\n\ndef ", "\n\nasync def ",
                    "\nclass ", "\ndef ", "\nasync def ",
                    "\n\n", "\n", " ", ""
                ]
            ),
            ContentType.DOCUMENTATION: cls(
                chunk_size=512,
                chunk_overlap=100,
                separators=[
                    "\n## ", "\n### ", "\n#### ",
                    "\n\n", "\n", ". ", " ", ""
                ]
            ),
            ContentType.CHANGELOG: cls(
                chunk_size=384,
                chunk_overlap=75,
                separators=[
                    "\n## ", "\n### ", "\n---\n",
                    "\n\n", "\n- ", "\n", " ", ""
                ]
            ),
            ContentType.CONFIG: cls(
                chunk_size=256,
                chunk_overlap=25,
                separators=[
                    "\n\n", "\n", " ", ""
                ]
            ),
            ContentType.NARRATIVE: cls(
                chunk_size=768,
                chunk_overlap=150,
                separators=[
                    "\n\n\n", "\n\n", "\n", ". ", " ", ""
                ]
            ),
            ContentType.MIXED: cls(
                chunk_size=512,
                chunk_overlap=100,
                separators=[
                    "\n## ", "\n### ", "\n\n", "\n", " ", ""
                ]
            ),
        }
        return configs.get(content_type, configs[ContentType.MIXED])


class ContentTypeDetector:
    """Detecteur de type de contenu."""

    @staticmethod
    def detect(text: str) -> ContentType:
        """
        Detecte le type de contenu d'un texte.

        Heuristiques:
        - Ratio de code (code blocks, inline code)
        - Patterns specifiques (changelog, config)
        - Structure (headers markdown)
        """
        lines = text.split('\n')
        total_lines = len(lines)

        if total_lines == 0:
            return ContentType.NARRATIVE

        # Compte les elements
        code_blocks = len(re.findall(r'```[\s\S]*?```', text))
        inline_code = len(re.findall(r'`[^`]+`', text))
        headers = len(re.findall(r'^#+\s', text, re.MULTILINE))
        list_items = len(re.findall(r'^[\-\*]\s', text, re.MULTILINE))

        # Patterns specifiques
        has_version_headers = bool(re.search(r'^#+\s*v?\d+\.\d+', text, re.MULTILINE))
        has_date_headers = bool(re.search(r'^#+\s*\d{4}-\d{2}-\d{2}', text, re.MULTILINE))
        has_dockerfile = bool(re.search(r'^(FROM|ENV|RUN|CMD|EXPOSE|WORKDIR)\s', text, re.MULTILINE))
        has_yaml_like = bool(re.search(r'^[a-zA-Z_]+:\s*\S', text, re.MULTILINE))
        has_json_like = bool(re.search(r'^\s*["\{}\[\]]', text, re.MULTILINE))

        # Calcul des ratios
        code_ratio = (code_blocks * 20 + inline_code * 2) / max(total_lines, 1)
        structure_ratio = (headers + list_items) / max(total_lines, 1)

        # Decision
        if has_dockerfile or has_yaml_like or has_json_like:
            return ContentType.CONFIG

        if has_version_headers or has_date_headers:
            return ContentType.CHANGELOG

        if code_ratio > 1.5:
            return ContentType.CODE

        if structure_ratio > 0.15 and headers > 2:
            return ContentType.DOCUMENTATION

        if code_ratio > 0.3 and structure_ratio > 0.1:
            return ContentType.MIXED

        return ContentType.NARRATIVE


class AdaptiveChunker:
    """
    Chunker adaptatif qui ajuste ses parametres selon le contenu.

    Features:
    - Detection automatique du type de contenu
    - Chunking adaptatif par type
    - Metadata enrichie (type, contexte)
    - Support du chunking par fichier ou par section
    - Extraction hierarchie headers h1/h2/h3 (v3.2)
    """

    def __init__(
        self,
        default_chunk_size: int = 512,
        default_chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base",
        adaptive: bool = True
    ):
        """
        Args:
            default_chunk_size: Taille par defaut
            default_chunk_overlap: Overlap par defaut
            encoding_name: Encodage pour le comptage de tokens
            adaptive: Active l'adaptation automatique
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self.adaptive = adaptive
        self.detector = ContentTypeDetector()
        self.header_extractor = MarkdownHeaderExtractor()

        # Cache des splitters par config
        self._splitter_cache: dict[tuple, RecursiveCharacterTextSplitter] = {}

    def _get_splitter(self, config: ChunkConfig) -> RecursiveCharacterTextSplitter:
        """Retourne un splitter pour une config donnee (avec cache)."""
        cache_key = (config.chunk_size, config.chunk_overlap, tuple(config.separators))

        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=config.separators
            )

        return self._splitter_cache[cache_key]

    def detect_content_type(self, text: str) -> ContentType:
        """Detecte le type de contenu."""
        return self.detector.detect(text)

    def chunk_document(
        self,
        text: str,
        source: str,
        force_type: Optional[ContentType] = None
    ) -> list[dict]:
        """
        Decoupe un document en chunks avec metadata enrichie.

        Args:
            text: Texte a decouper
            source: Nom du fichier source
            force_type: Force un type de contenu (optionnel)

        Returns:
            Liste de chunks avec metadata (incluant h1, h2, h3, header_path)
        """
        # Detection du type
        if force_type:
            content_type = force_type
        elif self.adaptive:
            content_type = self.detect_content_type(text)
        else:
            content_type = ContentType.MIXED

        # Configuration adaptee
        config = ChunkConfig.for_content_type(content_type)
        splitter = self._get_splitter(config)

        # Chunking
        chunks = splitter.split_text(text)

        # Construction des resultats avec metadata enrichie
        result = []
        search_start = 0  # Pour trouver les positions de maniere efficace

        for i, chunk in enumerate(chunks):
            token_count = len(self.encoding.encode(chunk))

            # Detecte le contexte local (section header si present)
            context = self._extract_context(chunk)

            # Trouve la position du chunk dans le texte original
            chunk_pos = text.find(chunk, search_start)
            if chunk_pos == -1:
                chunk_pos = search_start  # Fallback
            else:
                search_start = chunk_pos + 1  # Pour le prochain chunk

            # Extrait la hierarchie des headers (v3.2)
            headers = self.header_extractor.get_header_hierarchy(text, chunk_pos)

            result.append({
                "content": chunk,
                "source": source,
                "chunk_id": f"{source}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(chunks),
                "token_count": token_count,
                "content_type": content_type.value,
                "chunk_config": {
                    "size": config.chunk_size,
                    "overlap": config.chunk_overlap
                },
                "context": context,
                # Headers hierarchy (v3.2)
                "h1": headers["h1"],
                "h2": headers["h2"],
                "h3": headers["h3"],
                "header_path": headers["header_path"]
            })

        return result

    def _extract_context(self, chunk: str) -> str:
        """Extrait le contexte (header) d'un chunk."""
        # Cherche le premier header markdown
        match = re.search(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
        if match:
            return match.group(2).strip()

        # Sinon premiers mots
        words = chunk.split()[:5]
        return " ".join(words) if words else ""

    def chunk_by_sections(
        self,
        text: str,
        source: str,
        section_pattern: str = r'^##\s+(.+)$'
    ) -> list[dict]:
        """
        Decoupe par sections markdown puis chunke chaque section.

        Utile pour les gros documents structures.

        Args:
            text: Texte a decouper
            source: Nom du fichier
            section_pattern: Pattern regex pour detecter les sections

        Returns:
            Liste de chunks avec metadata de section
        """
        # Split par sections
        sections = re.split(f'({section_pattern})', text, flags=re.MULTILINE)

        result = []
        current_section = "intro"
        chunk_index = 0

        i = 0
        while i < len(sections):
            section_text = sections[i]

            # Detecte si c'est un header de section
            if re.match(section_pattern, section_text, re.MULTILINE):
                current_section = section_text.strip('#').strip()
                i += 1
                if i < len(sections):
                    section_text = sections[i]
                else:
                    continue

            if not section_text.strip():
                i += 1
                continue

            # Chunke cette section
            section_chunks = self.chunk_document(
                section_text,
                source,
                force_type=None  # Auto-detect par section
            )

            # Enrichit avec l'info de section
            for chunk in section_chunks:
                chunk["section"] = current_section
                chunk["chunk_id"] = f"{source}_sec_{current_section[:20]}_{chunk_index}"
                chunk["chunk_index"] = chunk_index
                result.append(chunk)
                chunk_index += 1

            i += 1

        # Met a jour total_chunks
        for chunk in result:
            chunk["total_chunks"] = len(result)

        return result


# Alias pour compatibilite avec l'ancien code
class TokenChunker(AdaptiveChunker):
    """Alias de compatibilite pour AdaptiveChunker."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        super().__init__(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            adaptive=True  # Active l'adaptation par defaut
        )
