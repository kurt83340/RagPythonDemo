"""
Configuration centralisee du RAG server v3.2.12

Contient:
- Version du serveur
- Detection environnement (Docker/local)
- Chemins (knowledge, db, state, models)
- Configuration modeles (embedding, reranker)
- Templates fichiers
- Patterns fichiers (append-only, editable)
"""

import os
from pathlib import Path

# ============================================================
# VERSION
# ============================================================

VERSION = "3.2.15"

# ============================================================
# ENVIRONNEMENT
# ============================================================

# Detecte si on est dans Docker ou en local
IN_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER", False)

if IN_DOCKER:
    BASE_DIR = Path("/app")
else:
    BASE_DIR = Path(__file__).parent

# ============================================================
# CHEMINS
# ============================================================

if IN_DOCKER:
    KNOWLEDGE_PATH = Path("/app/knowledge")
    DB_PATH = Path("/app/db")
    STATE_PATH = Path("/app/db/state")
else:
    KNOWLEDGE_PATH = BASE_DIR.parent / "knowledge"
    DB_PATH = BASE_DIR / "db"
    STATE_PATH = BASE_DIR / "db" / "state"

MODELS_DIR = BASE_DIR / "models"

# Configure les caches pour stockage local
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODELS_DIR))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MODELS_DIR))

# ============================================================
# MODELES
# ============================================================

# Modele multilingue optimise pour le francais
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Cross-encoder multilingue pour reranking
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ============================================================
# CACHE SEMANTIQUE
# ============================================================

CACHE_TTL_SECONDS = 86400  # 24 heures
CACHE_SIMILARITY_THRESHOLD = 0.88

# ============================================================
# TIME-WEIGHTED SCORING
# ============================================================

TIME_WEIGHT_HALF_LIFE_DAYS = 30

# ============================================================
# MMR (Maximal Marginal Relevance)
# ============================================================

# Active MMR par defaut pour diversifier les resultats
MMR_ENABLED_DEFAULT = True

# Lambda: balance pertinence vs diversite
# 1.0 = 100% pertinence (pas de MMR)
# 0.7 = 70% pertinence, 30% diversite (defaut recommande)
# 0.5 = equilibre 50/50
MMR_LAMBDA_DEFAULT = 0.7

# Multiplicateur pour recuperer plus de candidats avant MMR
MMR_FETCH_K_MULTIPLIER = 2

# ============================================================
# TEMPLATES FICHIERS
# ============================================================

TEMPLATE_FILES = {
    "projet/project.md": """# Projet

## Vision

[A definir - Decris ton idee en une phrase]

## Contexte

[A definir - Pourquoi ce projet ? Quel probleme resout-il ?]

## Objectifs

[A definir - Qu'est-ce que le projet doit accomplir ?]

## Utilisateurs cibles

[A definir - Qui va utiliser le produit ?]

## Contraintes

[A definir - Budget, delais, technologies imposees, etc.]
""",
    "projet/specs.md": """# Specifications

## User Stories

[A definir apres phase ideation]

## Fonctionnalites MVP

| Feature | Priorite | Status |
|---------|----------|--------|
| | | |

## Fonctionnalites V2+

[A definir plus tard]

## Hors scope

[Ce qu'on ne fait PAS dans ce projet]
""",
    "projet/architecture.md": """# Architecture

## Stack technique

[A definir apres specs]

## Structure du projet

[A definir]

## Decisions techniques

[Historique des choix avec justifications]
""",
    "journal/progress.md": """# Progression

[Les entrees s'ajoutent automatiquement ci-dessous]
""",
    "journal/changelog.md": """# Changelog

[Les versions s'ajoutent automatiquement ci-dessous]
""",
    "journal/bugs.md": """# Bugs resolus

[Les bugs s'ajoutent automatiquement ci-dessous]
""",
    "journal/learnings.md": """# Apprentissages

[Les apprentissages s'ajoutent automatiquement ci-dessous]
"""
}

# ============================================================
# PATTERNS FICHIERS
# ============================================================

# Fichiers append-only (utilisent append_doc)
APPEND_ONLY_PATTERNS = ["progress", "changelog", "bugs", "learnings", "journal/"]

# Fichiers editables (utilisent update_doc)
EDITABLE_PATTERNS = ["project", "specs", "architecture", "projet/"]

# Extensions supportees pour l'indexation
SUPPORTED_EXTENSIONS = ['.md', '.pdf', '.txt', '.json']
