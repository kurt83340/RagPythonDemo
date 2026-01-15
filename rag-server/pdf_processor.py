"""
PDF Processor: Support pour l'indexation de fichiers PDF.

Fonctionnalites:
- Extraction de texte depuis PDF
- Preservation de la structure (pages, sections)
- Support des PDF avec images (OCR basique)
- Conversion en markdown pour indexation
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import conditionnel pour PDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class PDFPage:
    """Represente une page de PDF."""
    page_number: int
    text: str
    images_count: int = 0
    tables_count: int = 0


@dataclass
class PDFDocument:
    """Represente un document PDF extrait."""
    source: str
    title: str
    pages: list[PDFPage]
    total_pages: int
    metadata: dict


class PDFProcessor:
    """
    Processeur de fichiers PDF.

    Strategies d'extraction:
    1. PyMuPDF (fitz) - rapide, bonne qualite
    2. pypdf - fallback si PyMuPDF non disponible
    """

    def __init__(self):
        if PYMUPDF_AVAILABLE:
            self.backend = "pymupdf"
            log("  PDF processor: PyMuPDF")
        elif PYPDF_AVAILABLE:
            self.backend = "pypdf"
            log("  PDF processor: pypdf")
        else:
            self.backend = None
            log("  PDF processor: Non disponible (installer pymupdf ou pypdf)")

    def is_available(self) -> bool:
        """Verifie si le traitement PDF est disponible."""
        return self.backend is not None

    def extract(self, filepath: Path) -> Optional[PDFDocument]:
        """
        Extrait le contenu d'un fichier PDF.

        Args:
            filepath: Chemin vers le fichier PDF

        Returns:
            PDFDocument ou None si erreur
        """
        if not self.is_available():
            log(f"  PDF non disponible pour {filepath}")
            return None

        if not filepath.exists():
            log(f"  Fichier non trouve: {filepath}")
            return None

        try:
            if self.backend == "pymupdf":
                return self._extract_pymupdf(filepath)
            else:
                return self._extract_pypdf(filepath)
        except Exception as e:
            log(f"  Erreur extraction PDF {filepath}: {e}")
            return None

    def _extract_pymupdf(self, filepath: Path) -> PDFDocument:
        """Extraction avec PyMuPDF."""
        doc = fitz.open(str(filepath))

        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Compte les images
            images_count = len(page.get_images())

            # Detecte les tableaux (heuristique basique)
            tables_count = text.count("|") // 3  # Approximation

            pages.append(PDFPage(
                page_number=page_num + 1,
                text=text,
                images_count=images_count,
                tables_count=tables_count
            ))

        # Metadata
        metadata = doc.metadata or {}
        title = metadata.get("title", filepath.stem)

        doc.close()

        return PDFDocument(
            source=str(filepath.name),
            title=title,
            pages=pages,
            total_pages=len(pages),
            metadata=metadata
        )

    def _extract_pypdf(self, filepath: Path) -> PDFDocument:
        """Extraction avec pypdf."""
        reader = PdfReader(str(filepath))

        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            pages.append(PDFPage(
                page_number=i + 1,
                text=text,
                images_count=0,  # pypdf ne compte pas les images facilement
                tables_count=0
            ))

        # Metadata
        metadata = {}
        if reader.metadata:
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
            }

        title = metadata.get("title", "") or filepath.stem

        return PDFDocument(
            source=str(filepath.name),
            title=title,
            pages=pages,
            total_pages=len(pages),
            metadata=metadata
        )

    def to_markdown(self, pdf_doc: PDFDocument) -> str:
        """
        Convertit un PDF en markdown.

        Args:
            pdf_doc: Document PDF extrait

        Returns:
            Contenu en markdown
        """
        md_parts = []

        # Titre
        md_parts.append(f"# {pdf_doc.title}\n")
        md_parts.append(f"> Source: {pdf_doc.source} ({pdf_doc.total_pages} pages)\n")

        # Metadata si disponible
        if pdf_doc.metadata.get("author"):
            md_parts.append(f"> Auteur: {pdf_doc.metadata['author']}\n")

        md_parts.append("\n---\n")

        # Contenu par page
        for page in pdf_doc.pages:
            if pdf_doc.total_pages > 1:
                md_parts.append(f"\n## Page {page.page_number}\n")

            # Nettoie et structure le texte
            cleaned_text = self._clean_text(page.text)
            md_parts.append(cleaned_text)

            # Info sur les images/tableaux
            if page.images_count > 0:
                md_parts.append(f"\n*[{page.images_count} image(s) sur cette page]*\n")

        return "\n".join(md_parts)

    def _clean_text(self, text: str) -> str:
        """Nettoie et structure le texte extrait."""
        if not text:
            return ""

        # Normalise les espaces
        text = re.sub(r'\s+', ' ', text)

        # Detecte les titres potentiels (lignes courtes en majuscules)
        lines = text.split('. ')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Titre potentiel ?
            if len(line) < 50 and line.isupper():
                cleaned_lines.append(f"\n### {line.title()}\n")
            else:
                cleaned_lines.append(line + ".")

        return "\n".join(cleaned_lines)

    def chunk_pdf(
        self,
        pdf_doc: PDFDocument,
        chunk_by: str = "page"
    ) -> list[dict]:
        """
        Decoupe un PDF en chunks.

        Args:
            pdf_doc: Document PDF
            chunk_by: Strategie - "page" ou "section"

        Returns:
            Liste de chunks avec metadata
        """
        chunks = []

        if chunk_by == "page":
            for page in pdf_doc.pages:
                if page.text.strip():
                    chunks.append({
                        "content": self._clean_text(page.text),
                        "source": pdf_doc.source,
                        "chunk_id": f"{pdf_doc.source}_page_{page.page_number}",
                        "page_number": page.page_number,
                        "total_pages": pdf_doc.total_pages,
                        "content_type": "pdf",
                        "has_images": page.images_count > 0
                    })
        else:
            # Chunk par section (detecte les headers)
            full_text = "\n".join(page.text for page in pdf_doc.pages)
            sections = self._split_by_sections(full_text)

            for i, (title, content) in enumerate(sections):
                if content.strip():
                    chunks.append({
                        "content": self._clean_text(content),
                        "source": pdf_doc.source,
                        "chunk_id": f"{pdf_doc.source}_section_{i}",
                        "section_title": title,
                        "content_type": "pdf"
                    })

        return chunks

    def _split_by_sections(self, text: str) -> list[tuple[str, str]]:
        """Split le texte par sections detectees."""
        # Pattern pour detecter les titres (heuristique)
        section_pattern = r'\n([A-Z][A-Z\s]{3,50})\n'

        parts = re.split(section_pattern, text)

        sections = []
        current_title = "Introduction"
        current_content = []

        for part in parts:
            if part.isupper() and len(part.strip()) < 50:
                # C'est un titre
                if current_content:
                    sections.append((current_title, ' '.join(current_content)))
                current_title = part.strip().title()
                current_content = []
            else:
                current_content.append(part)

        # Derniere section
        if current_content:
            sections.append((current_title, ' '.join(current_content)))

        return sections


class MultiFileProcessor:
    """
    Processeur multi-format pour /knowledge.

    Supporte:
    - .md (Markdown)
    - .pdf (PDF)
    - .txt (Texte brut)
    - .json (Documentation JSON)
    """

    SUPPORTED_EXTENSIONS = {'.md', '.pdf', '.txt', '.json'}

    def __init__(self):
        self.pdf_processor = PDFProcessor()

    def can_process(self, filepath: Path) -> bool:
        """Verifie si un fichier peut etre traite."""
        ext = filepath.suffix.lower()

        if ext == '.pdf':
            return self.pdf_processor.is_available()

        return ext in self.SUPPORTED_EXTENSIONS

    def process(self, filepath: Path) -> Optional[str]:
        """
        Traite un fichier et retourne son contenu en markdown.

        Args:
            filepath: Chemin du fichier

        Returns:
            Contenu en markdown ou None
        """
        ext = filepath.suffix.lower()

        try:
            if ext == '.md':
                return filepath.read_text(encoding='utf-8')

            elif ext == '.txt':
                content = filepath.read_text(encoding='utf-8')
                return f"# {filepath.stem}\n\n{content}"

            elif ext == '.pdf':
                pdf_doc = self.pdf_processor.extract(filepath)
                if pdf_doc:
                    return self.pdf_processor.to_markdown(pdf_doc)
                return None

            elif ext == '.json':
                return self._process_json(filepath)

        except Exception as e:
            log(f"  Erreur traitement {filepath}: {e}")
            return None

        return None

    def _process_json(self, filepath: Path) -> str:
        """Traite un fichier JSON en markdown."""
        import json

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            md_parts = [f"# {filepath.stem}\n"]

            if isinstance(data, dict):
                for key, value in data.items():
                    md_parts.append(f"\n## {key}\n")
                    if isinstance(value, (list, dict)):
                        md_parts.append(f"```json\n{json.dumps(value, indent=2)}\n```\n")
                    else:
                        md_parts.append(f"{value}\n")

            elif isinstance(data, list):
                for i, item in enumerate(data):
                    md_parts.append(f"\n## Item {i+1}\n")
                    if isinstance(item, dict):
                        for k, v in item.items():
                            md_parts.append(f"- **{k}**: {v}\n")
                    else:
                        md_parts.append(f"{item}\n")

            return "\n".join(md_parts)

        except Exception as e:
            log(f"  Erreur JSON {filepath}: {e}")
            return None

    def get_chunks(self, filepath: Path) -> list[dict]:
        """
        Traite un fichier et retourne ses chunks.

        Args:
            filepath: Chemin du fichier

        Returns:
            Liste de chunks
        """
        ext = filepath.suffix.lower()

        if ext == '.pdf' and self.pdf_processor.is_available():
            pdf_doc = self.pdf_processor.extract(filepath)
            if pdf_doc:
                return self.pdf_processor.chunk_pdf(pdf_doc)
            return []

        # Pour les autres formats, retourne le contenu brut
        content = self.process(filepath)
        if content:
            return [{
                "content": content,
                "source": filepath.name,
                "chunk_id": f"{filepath.name}_full",
                "content_type": ext[1:]  # Sans le point
            }]

        return []
