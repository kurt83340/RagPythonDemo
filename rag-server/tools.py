"""
Definitions des outils MCP pour le RAG server.

Contient:
- get_tool_definitions() : Liste des outils disponibles
- Schemas JSON pour chaque outil
"""

from mcp.types import Tool

from config import KNOWLEDGE_PATH
from utils import is_append_only, is_editable
import indexing


def get_tool_definitions() -> list[Tool]:
    """
    Retourne la liste des outils MCP disponibles.

    Returns:
        Liste de Tool avec schemas JSON
    """
    # Liste dynamique des fichiers pour les enums
    all_files = indexing.get_all_knowledge_files()
    append_files = [f for f in all_files if is_append_only(f)]
    edit_files = [f for f in all_files if is_editable(f)]

    return [
        # ============================================================
        # RECHERCHE
        # ============================================================
        Tool(
            name="search",
            description="Cherche dans la documentation du projet avec recherche hybride (Vector + BM25) et reranking. UTILISE CET OUTIL AVANT de repondre sur les specs, l'architecture, l'avancement, les bugs, ou toute info projet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Mots-cles a chercher (court et precis)"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Nombre de resultats (defaut: 5)",
                        "default": 5
                    },
                    "source": {
                        "type": "string",
                        "description": "Filtrer par fichier source (ex: specs, progress)"
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Filtrer par type de document (project, specs, progress, changelog, bugs, learnings)"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filtrer depuis cette date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Filtrer jusqu'a cette date (YYYY-MM-DD)"
                    },
                    "tags": {
                        "type": "string",
                        "description": "Filtrer par tags (ex: api, frontend)"
                    },
                    "boost_recent": {
                        "type": "boolean",
                        "description": "Booster le contenu recent (defaut: true)",
                        "default": True
                    },
                    "use_mmr": {
                        "type": "boolean",
                        "description": "Activer MMR pour diversifier les resultats (defaut: true)",
                        "default": True
                    },
                    "mmr_lambda": {
                        "type": "number",
                        "description": "Balance pertinence/diversite: 1.0=pertinence pure, 0.7=defaut, 0.5=equilibre (defaut: 0.7)",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            }
        ),

        # ============================================================
        # EDITION
        # ============================================================
        Tool(
            name="update_doc",
            description="Met a jour une section d'un fichier de documentation et reindexe. Pour les fichiers editables (projet/project, projet/specs, projet/architecture).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Nom du fichier (ex: projet/project.md)",
                        "enum": edit_files if edit_files else ["projet/project.md", "projet/specs.md", "projet/architecture.md"]
                    },
                    "section": {
                        "type": "string",
                        "description": "Nom de la section a mettre a jour (ex: Vision, Stack technique)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Nouveau contenu de la section"
                    }
                },
                "required": ["file", "section", "content"]
            }
        ),
        Tool(
            name="append_doc",
            description="Ajoute une entree horodatee a un fichier append-only et reindexe. Pour progress, changelog, bugs, learnings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Nom du fichier",
                        "enum": append_files if append_files else ["journal/progress.md", "journal/changelog.md", "journal/bugs.md", "journal/learnings.md"]
                    },
                    "title": {
                        "type": "string",
                        "description": "Titre court de l'entree"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu detaille"
                    }
                },
                "required": ["file", "title", "content"]
            }
        ),

        # ============================================================
        # MAINTENANCE
        # ============================================================
        Tool(
            name="reindex",
            description="Force la reindexation de tous les fichiers /knowledge. Utilise apres modification manuelle des fichiers.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_stats",
            description="Affiche les statistiques du RAG (fichiers indexes, nombre de chunks, modeles utilises)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_feedback",
            description="Affiche les analytics du feedback loop (recherches echouees, patterns, suggestions)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="run_maintenance",
            description="Execute la maintenance du RAG (archivage vieux contenus, detection contradictions, analyse qualite)",
            inputSchema={
                "type": "object",
                "properties": {
                    "auto_archive": {
                        "type": "boolean",
                        "description": "Archive automatiquement les vieux chunks (defaut: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="log_session",
            description="UTILISE CET OUTIL A LA FIN DE CHAQUE SESSION pour memoriser ce qui a ete fait. Cree un fichier dans sessions/",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Resume court de ce qui a ete fait dans cette session (2-3 phrases)"
                    },
                    "details": {
                        "type": "string",
                        "description": "Details: fichiers modifies, decisions prises, problemes resolus"
                    },
                    "next_steps": {
                        "type": "string",
                        "description": "Prochaines etapes a faire (optionnel)",
                        "default": ""
                    }
                },
                "required": ["summary", "details"]
            }
        ),

        # ============================================================
        # LECTURE / NAVIGATION
        # ============================================================
        Tool(
            name="read_doc",
            description="Lit le contenu complet d'un fichier du RAG. Utile pour voir un fichier entier sans chercher.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier (ex: projet/specs.md, sessions/2025-01-10_03-45.md)"
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="list_files",
            description="Liste les fichiers dans le RAG. Sans argument: tous les fichiers. Avec folder: fichiers du dossier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Dossier a lister (ex: projet, journal, sessions). Vide = tout.",
                        "default": ""
                    }
                }
            }
        ),
        Tool(
            name="get_summary",
            description="Retourne un resume du projet: vision, specs principales, et derniere session. Utile en debut de conversation.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),

        # ============================================================
        # GESTION FICHIERS
        # ============================================================
        Tool(
            name="create_doc",
            description="Cree un nouveau fichier dans /knowledge et l'indexe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Chemin du fichier a creer (ex: references/api.md)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenu initial du fichier"
                    }
                },
                "required": ["path", "content"]
            }
        ),
        Tool(
            name="delete_doc",
            description="Supprime un fichier du RAG et de l'index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier a supprimer"
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="move_doc",
            description="Deplace ou renomme un fichier dans le RAG.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Chemin actuel du fichier"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Nouveau chemin du fichier"
                    }
                },
                "required": ["source", "destination"]
            }
        ),

        # ============================================================
        # RECHERCHE AVANCEE
        # ============================================================
        Tool(
            name="search_by_date",
            description="Cherche les documents crees/modifies dans une periode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_date": {
                        "type": "string",
                        "description": "Date debut (YYYY-MM-DD)"
                    },
                    "to_date": {
                        "type": "string",
                        "description": "Date fin (YYYY-MM-DD). Defaut: aujourd'hui.",
                        "default": ""
                    }
                },
                "required": ["from_date"]
            }
        ),
        Tool(
            name="get_related",
            description="Trouve les fichiers/chunks similaires a un fichier donne.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Chemin du fichier de reference"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Nombre de resultats (defaut: 5)",
                        "default": 5
                    }
                },
                "required": ["file"]
            }
        ),
        Tool(
            name="get_timeline",
            description="Retourne une timeline chronologique des evenements du projet (sessions, progress, bugs, changelog).",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Nombre max d'evenements (defaut: 20)",
                        "default": 20
                    }
                }
            }
        ),

        # ============================================================
        # EXPORT / IMPORT
        # ============================================================
        Tool(
            name="export_all",
            description="Exporte toute la knowledge base dans un seul fichier markdown.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Nom du fichier de sortie (defaut: knowledge_export.md)",
                        "default": "knowledge_export.md"
                    }
                }
            }
        ),
        Tool(
            name="import_file",
            description="Importe un fichier externe dans /knowledge et l'indexe.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Chemin du fichier a importer (absolu ou relatif au projet)"
                    },
                    "dest_folder": {
                        "type": "string",
                        "description": "Dossier destination dans /knowledge (defaut: references)",
                        "default": "references"
                    }
                },
                "required": ["source_path"]
            }
        ),
        Tool(
            name="generate_map",
            description="Genere une carte du projet (_map.md) avec la structure des fichiers et leurs descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Dossier a scanner (defaut: racine du projet)",
                        "default": ""
                    },
                    "extensions": {
                        "type": "string",
                        "description": "Extensions a inclure separees par virgule (defaut: py,js,ts,tsx,jsx,md)",
                        "default": "py,js,ts,tsx,jsx,md"
                    }
                }
            }
        )
    ]
