"""
Contextual Retrieval: Methode Anthropic pour ameliorer le contexte des chunks.

Ajoute du contexte document-level a chaque chunk pour ameliorer le retrieval.
Ref: https://www.anthropic.com/news/contextual-retrieval

Fonctionnalites:
- Extraction du contexte global du document
- Prefixe contextuel pour chaque chunk
- Support parent-child chunking
"""

import re
import sys
from dataclasses import dataclass
from typing import Optional


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class DocumentContext:
    """Contexte global d'un document."""
    title: str
    summary: str
    main_topics: list[str]
    document_type: str  # project, specs, progress, changelog, etc.
    key_entities: list[str]


@dataclass
class ContextualChunk:
    """Chunk avec contexte enrichi."""
    content: str
    contextual_prefix: str
    full_content: str  # prefix + content
    parent_content: Optional[str]  # Chunk parent (plus large)
    metadata: dict


class ContextualRetrieval:
    """
    Systeme de contextual retrieval.

    Approche:
    1. Extrait le contexte global du document
    2. Genere un prefixe contextuel pour chaque chunk
    3. Stocke le chunk avec son contexte pour meilleur embedding
    4. Support parent-child pour retourner plus de contexte
    """

    # Patterns pour detecter le type de document
    DOC_TYPE_PATTERNS = {
        "project": r"(vision|contexte|objectif|projet)",
        "specs": r"(specification|user stor|fonctionnalite|mvp|feature)",
        "architecture": r"(architecture|stack|technique|decision)",
        "progress": r"(progression|avancement|termine|en cours)",
        "changelog": r"(changelog|version|release|v\d+\.\d+)",
        "bugs": r"(bug|erreur|fix|correction|issue)",
        "learnings": r"(apprentissage|learning|lesson|retour)",
        "session": r"(session|resume|details|prochaines)"
    }

    def __init__(self):
        pass

    def extract_document_context(self, document: str, source: str) -> DocumentContext:
        """
        Extrait le contexte global d'un document.

        Args:
            document: Contenu complet du document
            source: Nom du fichier source

        Returns:
            DocumentContext avec les infos extraites
        """
        # 1. Detecte le type de document
        doc_type = self._detect_document_type(document, source)

        # 2. Extrait le titre
        title = self._extract_title(document, source)

        # 3. Genere un resume
        summary = self._generate_summary(document)

        # 4. Extrait les topics principaux
        topics = self._extract_topics(document)

        # 5. Extrait les entites cles
        entities = self._extract_key_entities(document)

        return DocumentContext(
            title=title,
            summary=summary,
            main_topics=topics,
            document_type=doc_type,
            key_entities=entities
        )

    def _detect_document_type(self, document: str, source: str) -> str:
        """Detecte le type de document."""
        # D'abord par nom de fichier
        source_lower = source.lower()
        for doc_type in self.DOC_TYPE_PATTERNS:
            if doc_type in source_lower:
                return doc_type

        # Ensuite par contenu
        doc_lower = document.lower()
        scores = {}
        for doc_type, pattern in self.DOC_TYPE_PATTERNS.items():
            matches = len(re.findall(pattern, doc_lower, re.IGNORECASE))
            if matches > 0:
                scores[doc_type] = matches

        if scores:
            return max(scores, key=scores.get)

        return "general"

    def _extract_title(self, document: str, source: str) -> str:
        """Extrait le titre du document."""
        # Cherche le premier header #
        match = re.search(r'^#\s+(.+)$', document, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Sinon utilise le nom de fichier
        return source.replace('.md', '').replace('-', ' ').replace('_', ' ').title()

    def _generate_summary(self, document: str, max_length: int = 200) -> str:
        """Genere un resume du document."""
        # Cherche une section description/resume/overview
        for pattern in [r'##\s*(?:Description|Resume|Overview|Summary)\n(.*?)(?=\n##|\Z)',
                        r'##\s*Vision\n(.*?)(?=\n##|\Z)',
                        r'##\s*Contexte\n(.*?)(?=\n##|\Z)']:
            match = re.search(pattern, document, re.DOTALL | re.IGNORECASE)
            if match:
                summary = match.group(1).strip()
                # Nettoie et tronque
                summary = re.sub(r'\[.*?\]', '', summary)  # Retire les placeholders
                summary = ' '.join(summary.split())
                return summary[:max_length]

        # Sinon prend les premieres lignes non-header
        lines = []
        for line in document.split('\n'):
            if line.strip() and not line.startswith('#'):
                lines.append(line.strip())
                if len(' '.join(lines)) > max_length:
                    break

        return ' '.join(lines)[:max_length] if lines else ""

    def _extract_topics(self, document: str, max_topics: int = 5) -> list[str]:
        """Extrait les topics principaux (headers niveau 2)."""
        headers = re.findall(r'^##\s+(.+)$', document, re.MULTILINE)
        # Filtre les headers generiques
        generic = {'introduction', 'conclusion', 'resume', 'references', 'intro'}
        topics = [h.strip() for h in headers if h.lower().strip() not in generic]
        return topics[:max_topics]

    def _extract_key_entities(self, document: str, max_entities: int = 10) -> list[str]:
        """Extrait les entites cles du document."""
        entities = set()

        # Termes techniques (en backticks)
        code_refs = re.findall(r'`([^`]+)`', document)
        entities.update(code_refs[:max_entities])

        # Termes en gras
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', document)
        entities.update(bold_terms[:max_entities])

        # Acronymes
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', document)
        entities.update(acronyms[:max_entities])

        return list(entities)[:max_entities]

    def generate_contextual_prefix(
        self,
        chunk: str,
        context: DocumentContext,
        chunk_index: int = 0,
        total_chunks: int = 1
    ) -> str:
        """
        Genere un prefixe contextuel pour un chunk.

        Args:
            chunk: Contenu du chunk
            context: Contexte du document
            chunk_index: Index du chunk dans le document
            total_chunks: Nombre total de chunks

        Returns:
            Prefixe contextuel a prependre au chunk
        """
        parts = []

        # 1. Document source et type
        parts.append(f"Document: {context.title} ({context.document_type})")

        # 2. Resume si disponible
        if context.summary:
            parts.append(f"Contexte: {context.summary[:100]}")

        # 3. Position dans le document
        if total_chunks > 1:
            parts.append(f"Section {chunk_index + 1}/{total_chunks}")

        # 4. Topics si pertinents pour ce chunk
        relevant_topics = self._find_relevant_topics(chunk, context.main_topics)
        if relevant_topics:
            parts.append(f"Topics: {', '.join(relevant_topics[:3])}")

        return " | ".join(parts)

    def _find_relevant_topics(self, chunk: str, topics: list[str]) -> list[str]:
        """Trouve les topics pertinents pour un chunk."""
        chunk_lower = chunk.lower()
        relevant = []

        for topic in topics:
            # Verifie si le topic ou ses mots sont dans le chunk
            topic_words = topic.lower().split()
            if any(word in chunk_lower for word in topic_words if len(word) > 3):
                relevant.append(topic)

        return relevant

    def create_contextual_chunk(
        self,
        chunk: str,
        context: DocumentContext,
        chunk_index: int = 0,
        total_chunks: int = 1,
        parent_content: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> ContextualChunk:
        """
        Cree un chunk avec contexte complet.

        Args:
            chunk: Contenu brut du chunk
            context: Contexte du document
            chunk_index: Index du chunk
            total_chunks: Nombre total de chunks
            parent_content: Contenu du chunk parent (optionnel)
            metadata: Metadonnees additionnelles

        Returns:
            ContextualChunk avec tout le contexte
        """
        prefix = self.generate_contextual_prefix(
            chunk, context, chunk_index, total_chunks
        )

        full_content = f"{prefix}\n\n{chunk}"

        return ContextualChunk(
            content=chunk,
            contextual_prefix=prefix,
            full_content=full_content,
            parent_content=parent_content,
            metadata=metadata or {}
        )

    def process_document(
        self,
        document: str,
        source: str,
        chunks: list[dict],
        include_parent: bool = True
    ) -> list[ContextualChunk]:
        """
        Traite un document complet et enrichit ses chunks.

        Args:
            document: Contenu complet du document
            source: Nom du fichier
            chunks: Liste de chunks [{content, ...}, ...]
            include_parent: Inclure le contenu parent

        Returns:
            Liste de ContextualChunk
        """
        # Extrait le contexte global
        context = self.extract_document_context(document, source)

        contextual_chunks = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")

            # Trouve le chunk parent (section plus large)
            parent = None
            if include_parent:
                parent = self._find_parent_content(document, content)

            ctx_chunk = self.create_contextual_chunk(
                chunk=content,
                context=context,
                chunk_index=i,
                total_chunks=total,
                parent_content=parent,
                metadata=chunk
            )
            contextual_chunks.append(ctx_chunk)

        return contextual_chunks

    def _find_parent_content(
        self,
        document: str,
        chunk: str,
        expansion_lines: int = 5
    ) -> Optional[str]:
        """
        Trouve le contenu parent (contexte elargi) d'un chunk.

        Args:
            document: Document complet
            chunk: Chunk a trouver
            expansion_lines: Nombre de lignes a ajouter avant/apres

        Returns:
            Contenu parent ou None
        """
        # Trouve la position du chunk dans le document
        chunk_start = document.find(chunk[:100])  # Cherche les 100 premiers chars
        if chunk_start == -1:
            return None

        # Trouve les limites de section (headers ##)
        lines = document.split('\n')
        chunk_line = document[:chunk_start].count('\n')

        # Cherche le header precedent
        start_line = chunk_line
        for i in range(chunk_line, -1, -1):
            if lines[i].startswith('## '):
                start_line = i
                break
            if chunk_line - i > expansion_lines * 2:
                start_line = max(0, chunk_line - expansion_lines)
                break

        # Cherche le header suivant
        end_line = chunk_line + chunk.count('\n') + 1
        for i in range(end_line, len(lines)):
            if lines[i].startswith('## '):
                end_line = i
                break
            if i - chunk_line > expansion_lines * 2:
                end_line = min(len(lines), chunk_line + expansion_lines)
                break

        parent = '\n'.join(lines[start_line:end_line])

        # Ne retourne que si le parent est significativement plus grand
        if len(parent) > len(chunk) * 1.3:
            return parent

        return None


class ParentChildChunker:
    """
    Systeme de chunking parent-child.

    Strategie:
    - Petits chunks pour le retrieval (precision)
    - Retourne le chunk parent (contexte)
    """

    def __init__(
        self,
        child_size: int = 256,
        parent_size: int = 1024,
        child_overlap: int = 50,
        parent_overlap: int = 200
    ):
        """
        Args:
            child_size: Taille des chunks enfants (retrieval)
            parent_size: Taille des chunks parents (contexte)
            child_overlap: Overlap des chunks enfants
            parent_overlap: Overlap des chunks parents
        """
        self.child_size = child_size
        self.parent_size = parent_size
        self.child_overlap = child_overlap
        self.parent_overlap = parent_overlap

    def chunk_with_parents(
        self,
        document: str,
        source: str
    ) -> list[dict]:
        """
        Decoupe un document avec relation parent-child.

        Args:
            document: Contenu du document
            source: Nom du fichier

        Returns:
            Liste de chunks avec reference au parent
        """
        # 1. Cree les chunks parents (grandes sections)
        parent_chunks = self._create_chunks(
            document, self.parent_size, self.parent_overlap
        )

        # 2. Pour chaque parent, cree les chunks enfants
        all_chunks = []
        for parent_idx, parent in enumerate(parent_chunks):
            child_chunks = self._create_chunks(
                parent["content"],
                self.child_size,
                self.child_overlap
            )

            for child_idx, child in enumerate(child_chunks):
                all_chunks.append({
                    "content": child["content"],
                    "parent_content": parent["content"],
                    "parent_index": parent_idx,
                    "child_index": child_idx,
                    "source": source,
                    "chunk_id": f"{source}_p{parent_idx}_c{child_idx}",
                    "is_child": True
                })

        return all_chunks

    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> list[dict]:
        """Cree des chunks simples."""
        chunks = []
        words = text.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)

            chunks.append({
                "content": chunk_content,
                "start": start,
                "end": end
            })

            start = end - overlap
            if start >= len(words) - overlap:
                break

        return chunks

    def get_parent_for_retrieval(
        self,
        child_chunk: dict,
        return_siblings: bool = False
    ) -> dict:
        """
        Retourne le contexte parent pour un chunk retrieval.

        Args:
            child_chunk: Chunk enfant trouve par retrieval
            return_siblings: Inclure les chunks freres

        Returns:
            Dict avec parent et optionnellement siblings
        """
        result = {
            "child": child_chunk["content"],
            "parent": child_chunk.get("parent_content", child_chunk["content"])
        }

        if return_siblings:
            # A implementer si necessaire
            result["siblings"] = []

        return result
