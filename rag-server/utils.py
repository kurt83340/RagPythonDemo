"""
Utilitaires partages pour le RAG server.

Contient:
- Fonction de logging
- Helpers pour fichiers (is_append_only, is_editable)
- Highlighting des termes de recherche
"""

import re
import sys

from config import APPEND_ONLY_PATTERNS, EDITABLE_PATTERNS


def log(msg: str):
    """Log vers stderr pour ne pas interferer avec MCP."""
    print(msg, file=sys.stderr, flush=True)


def is_append_only(filename: str) -> bool:
    """
    Verifie si un fichier est append-only.

    Args:
        filename: Nom ou chemin du fichier

    Returns:
        True si le fichier utilise append_doc
    """
    name_lower = filename.lower()
    return any(pattern in name_lower for pattern in APPEND_ONLY_PATTERNS)


def is_editable(filename: str) -> bool:
    """
    Verifie si un fichier est editable.

    Args:
        filename: Nom ou chemin du fichier

    Returns:
        True si le fichier utilise update_doc
    """
    name_lower = filename.lower()
    return any(pattern in name_lower for pattern in EDITABLE_PATTERNS)


def highlight_matches(text: str, query_terms: list[str], marker: str = "**") -> str:
    """
    Surligne les termes de la query dans le texte.

    Args:
        text: Texte a surligner
        query_terms: Termes a chercher
        marker: Marqueur de surlignage (defaut: **bold**)

    Returns:
        Texte avec termes surlignees
    """
    highlighted = text
    for term in query_terms:
        if len(term) < 3:
            continue
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f"{marker}{term}{marker}", highlighted)
    return highlighted


def normalize_path(path: str) -> str:
    """
    Normalise un chemin de fichier (separateurs, casse).

    Args:
        path: Chemin a normaliser

    Returns:
        Chemin normalise avec forward slashes
    """
    return path.replace("\\", "/")


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Tronque un texte a une longueur maximale.

    Args:
        text: Texte a tronquer
        max_length: Longueur max
        suffix: Suffixe a ajouter si tronque

    Returns:
        Texte tronque
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
