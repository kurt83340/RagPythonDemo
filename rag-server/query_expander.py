"""
Query Expander: Expansion automatique des requetes.

Fonctionnalites:
- Synonymes francais/anglais
- Expansion d'acronymes
- Termes techniques associes
- Correction orthographique basique
"""

import re
import sys
from difflib import get_close_matches
from typing import Optional


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class QueryExpander:
    """
    Systeme d'expansion de requetes pour ameliorer le recall.

    Strategies:
    1. Synonymes (FR/EN)
    2. Acronymes -> formes longues
    3. Termes techniques associes
    4. Correction de typos
    """

    # Synonymes francais/anglais (bidirectionnels)
    SYNONYMS = {
        # Bugs et erreurs
        "bug": ["erreur", "probleme", "issue", "defaut", "anomalie"],
        "erreur": ["bug", "error", "probleme", "issue", "faute"],
        "error": ["bug", "erreur", "probleme", "issue", "fault"],
        "probleme": ["bug", "erreur", "issue", "souci", "problem"],
        "issue": ["bug", "erreur", "probleme", "ticket"],

        # Fonctionnalites
        "feature": ["fonctionnalite", "fonction", "capacite"],
        "fonctionnalite": ["feature", "fonction", "capacite"],
        "fonction": ["feature", "fonctionnalite", "method", "methode"],

        # Actions
        "fix": ["corriger", "reparer", "correction", "patch"],
        "corriger": ["fix", "reparer", "patch", "resoudre"],
        "ajouter": ["add", "creer", "implementer", "create"],
        "add": ["ajouter", "creer", "implementer", "create"],
        "supprimer": ["delete", "remove", "enlever", "effacer"],
        "delete": ["supprimer", "remove", "enlever", "effacer"],
        "modifier": ["update", "changer", "edit", "change"],
        "update": ["modifier", "mettre a jour", "maj", "upgrade"],

        # Statuts
        "termine": ["done", "fini", "complete", "acheve"],
        "done": ["termine", "fini", "complete", "finished"],
        "en cours": ["in progress", "wip", "ongoing", "actif"],
        "cours": ["progress", "ongoing", "wip"],  # "en cours" split
        "progress": ["cours", "ongoing", "wip", "actif"],
        "todo": ["a faire", "pending", "planifie"],
        "wip": ["en cours", "in progress", "ongoing"],

        # Architecture
        "api": ["endpoint", "route", "service", "interface"],
        "base de donnees": ["database", "db", "bdd", "stockage"],
        "database": ["base de donnees", "db", "bdd", "storage"],
        "frontend": ["ui", "interface", "client", "vue"],
        "backend": ["api", "serveur", "server", "service"],
        "serveur": ["server", "backend", "service"],

        # Documentation
        "doc": ["documentation", "docs", "guide", "readme"],
        "documentation": ["doc", "docs", "guide", "manuel"],
        "spec": ["specification", "specs", "cahier des charges", "cdc"],
        "specification": ["spec", "specs", "requirements", "exigences"],

        # Concepts
        "utilisateur": ["user", "client", "usager"],
        "user": ["utilisateur", "client", "usager"],
        "authentification": ["auth", "login", "connexion", "authentication"],
        "auth": ["authentification", "login", "connexion", "authentication"],
        "config": ["configuration", "settings", "parametres", "reglages"],
        "configuration": ["config", "settings", "parametres", "setup"],

        # Temps
        "aujourd'hui": ["today", "maintenant", "actuel"],
        "hier": ["yesterday", "precedent"],
        "recent": ["dernier", "nouveau", "latest", "last"],

        # RAG / ML - CONSERVATEUR (pas d'expansion termes techniques)
        "multilingue": ["multilingual"],
        "multilingual": ["multilingue"],
        "optimiser": ["ameliorer"],
        "optimize": ["optimiser"],
    }

    # Acronymes et leurs expansions
    ACRONYMS = {
        "api": "Application Programming Interface",
        "ui": "User Interface",
        "ux": "User Experience",
        "db": "Database",
        "bdd": "Base de Donnees",
        "cdc": "Cahier des Charges",
        "mvp": "Minimum Viable Product",
        "poc": "Proof of Concept",
        "ci": "Continuous Integration",
        "cd": "Continuous Deployment",
        "pr": "Pull Request",
        "mr": "Merge Request",
        "wip": "Work In Progress",
        "tbd": "To Be Defined",
        "eta": "Estimated Time of Arrival",
        "asap": "As Soon As Possible",
        "crud": "Create Read Update Delete",
        "rest": "Representational State Transfer",
        "jwt": "JSON Web Token",
        "oauth": "Open Authorization",
        "ssl": "Secure Sockets Layer",
        "tls": "Transport Layer Security",
        "http": "HyperText Transfer Protocol",
        "https": "HTTP Secure",
        "json": "JavaScript Object Notation",
        "xml": "eXtensible Markup Language",
        "yaml": "YAML Ain't Markup Language",
        "sql": "Structured Query Language",
        "css": "Cascading Style Sheets",
        "html": "HyperText Markup Language",
        "dom": "Document Object Model",
        "sdk": "Software Development Kit",
        "cli": "Command Line Interface",
        "gui": "Graphical User Interface",
        "ide": "Integrated Development Environment",
        "rag": "Retrieval Augmented Generation",
        "llm": "Large Language Model",
        "nlp": "Natural Language Processing",
        "ml": "Machine Learning",
        "ai": "Artificial Intelligence",
    }

    # Termes techniques associes
    TECH_ASSOCIATIONS = {
        "react": ["component", "hook", "state", "props", "jsx", "tsx"],
        "vue": ["component", "composable", "ref", "reactive", "template"],
        "angular": ["component", "service", "module", "directive"],
        "python": ["def", "class", "import", "pip", "venv"],
        "javascript": ["function", "const", "let", "async", "await", "npm"],
        "typescript": ["interface", "type", "generic", "enum"],
        "docker": ["container", "image", "dockerfile", "compose", "volume"],
        "container": ["docker", "image", "pod", "kubernetes"],
        "git": ["commit", "branch", "merge", "push", "pull", "rebase"],
        "test": ["unittest", "pytest", "jest", "mock", "assert", "coverage"],
        "auth": ["token", "session", "password", "login", "logout", "permission"],
        "database": ["query", "table", "index", "migration", "schema"],
        "api": ["endpoint", "route", "request", "response", "middleware"],
        "rag": ["retrieval", "embedding", "chunk", "rerank", "vector"],
        "embedding": ["vector", "similarity", "cosine", "semantic"],
        "cache": ["redis", "memcached", "ttl", "invalidation"],
        "error": ["exception", "bug", "crash", "failure"],
        "performance": ["latency", "throughput", "optimization", "speed"],
    }

    def __init__(self, vocabulary: Optional[set[str]] = None):
        """
        Args:
            vocabulary: Vocabulaire connu pour la correction orthographique
        """
        self.vocabulary = vocabulary or set()
        self._build_reverse_synonyms()

    def _build_reverse_synonyms(self):
        """Construit les synonymes bidirectionnels."""
        # Deja bidirectionnel dans SYNONYMS
        pass

    def set_vocabulary(self, vocabulary: set[str]):
        """Met a jour le vocabulaire pour la correction."""
        self.vocabulary = vocabulary

    def expand_synonyms(self, term: str, max_synonyms: int = 3) -> list[str]:
        """
        Trouve les synonymes d'un terme.

        Args:
            term: Terme a etendre
            max_synonyms: Nombre max de synonymes

        Returns:
            Liste de synonymes
        """
        term_lower = term.lower().strip()

        if term_lower in self.SYNONYMS:
            return self.SYNONYMS[term_lower][:max_synonyms]

        return []

    def expand_acronym(self, term: str) -> Optional[str]:
        """
        Etend un acronyme en sa forme longue.

        Args:
            term: Acronyme potentiel

        Returns:
            Forme longue ou None
        """
        term_lower = term.lower().strip()
        return self.ACRONYMS.get(term_lower)

    def get_tech_associations(self, term: str, max_terms: int = 3) -> list[str]:
        """
        Trouve les termes techniques associes.

        Args:
            term: Terme technique
            max_terms: Nombre max de termes

        Returns:
            Liste de termes associes
        """
        term_lower = term.lower().strip()

        if term_lower in self.TECH_ASSOCIATIONS:
            return self.TECH_ASSOCIATIONS[term_lower][:max_terms]

        return []

    def correct_spelling(self, term: str, cutoff: float = 0.7) -> Optional[str]:
        """
        Corrige l'orthographe d'un terme.

        Args:
            term: Terme potentiellement mal orthographie
            cutoff: Seuil de similarite (0-1)

        Returns:
            Correction ou None si pas de correction
        """
        if not self.vocabulary:
            return None

        term_lower = term.lower().strip()

        # Si le terme existe dans le vocabulaire, pas de correction
        if term_lower in self.vocabulary:
            return None

        # Cherche les termes similaires
        matches = get_close_matches(
            term_lower,
            self.vocabulary,
            n=1,
            cutoff=cutoff
        )

        if matches and matches[0] != term_lower:
            return matches[0]

        return None

    def expand_query(
        self,
        query: str,
        include_synonyms: bool = True,
        include_acronyms: bool = True,
        include_tech: bool = True,
        include_spelling: bool = True,
        max_expansion: int = 3  # Reduit de 5 a 3 pour eviter le bruit
    ) -> dict:
        """
        Etend une requete complete.

        Args:
            query: Requete originale
            include_synonyms: Inclure les synonymes
            include_acronyms: Inclure les acronymes etendus
            include_tech: Inclure les associations techniques
            include_spelling: Inclure les corrections orthographiques
            max_expansion: Nombre max de termes ajoutes par type

        Returns:
            Dict avec:
            - original: requete originale
            - expanded: requete etendue
            - terms_added: termes ajoutes
            - corrections: corrections orthographiques
        """
        # Tokenize la query
        words = re.findall(r'\b\w+\b', query.lower())

        expanded_terms = set()
        corrections = {}

        for word in words:
            # 1. Synonymes
            if include_synonyms:
                synonyms = self.expand_synonyms(word, max_expansion)
                expanded_terms.update(synonyms)

            # 2. Acronymes
            if include_acronyms:
                acronym_expansion = self.expand_acronym(word)
                if acronym_expansion:
                    # Ajoute les mots significatifs de l'expansion
                    expansion_words = [
                        w.lower() for w in acronym_expansion.split()
                        if len(w) > 2
                    ]
                    expanded_terms.update(expansion_words[:max_expansion])

            # 3. Associations techniques
            if include_tech:
                tech_terms = self.get_tech_associations(word, max_expansion)
                expanded_terms.update(tech_terms)

            # 4. Corrections orthographiques
            if include_spelling:
                correction = self.correct_spelling(word)
                if correction:
                    corrections[word] = correction
                    expanded_terms.add(correction)

        # Retire les mots deja dans la query
        original_words = set(words)
        new_terms = expanded_terms - original_words

        # Construit la requete etendue
        if new_terms:
            expanded_query = f"{query} {' '.join(list(new_terms)[:max_expansion])}"
        else:
            expanded_query = query

        return {
            "original": query,
            "expanded": expanded_query,
            "terms_added": list(new_terms)[:max_expansion],
            "corrections": corrections
        }

    def get_alternative_queries(
        self,
        query: str,
        max_alternatives: int = 3
    ) -> list[str]:
        """
        Genere des requetes alternatives.

        Args:
            query: Requete originale
            max_alternatives: Nombre max d'alternatives

        Returns:
            Liste de requetes alternatives
        """
        words = re.findall(r'\b\w+\b', query.lower())
        alternatives = []

        for i, word in enumerate(words):
            # Remplace chaque mot par ses synonymes
            synonyms = self.expand_synonyms(word, max_alternatives)

            for syn in synonyms[:2]:
                new_words = words.copy()
                new_words[i] = syn
                alternatives.append(' '.join(new_words))

        # Deduplique et limite
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt not in seen and alt != query.lower():
                seen.add(alt)
                unique_alternatives.append(alt)

        return unique_alternatives[:max_alternatives]


class SpellChecker:
    """
    Correcteur orthographique simple base sur la distance de Levenshtein.
    """

    def __init__(self, dictionary: set[str] = None):
        """
        Args:
            dictionary: Dictionnaire de mots corrects
        """
        self.dictionary = dictionary or set()

    def update_dictionary(self, words: set[str]):
        """Ajoute des mots au dictionnaire."""
        self.dictionary.update(words)

    def build_from_documents(self, documents: list[str]):
        """
        Construit le dictionnaire a partir de documents.

        Args:
            documents: Liste de textes
        """
        for doc in documents:
            words = re.findall(r'\b\w{3,}\b', doc.lower())
            self.dictionary.update(words)

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calcule la distance de Levenshtein entre deux chaines."""
        if len(s1) < len(s2):
            return SpellChecker.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def suggest(self, word: str, max_suggestions: int = 3, max_distance: int = 2) -> list[str]:
        """
        Suggere des corrections pour un mot.

        Args:
            word: Mot a corriger
            max_suggestions: Nombre max de suggestions
            max_distance: Distance max acceptable

        Returns:
            Liste de suggestions
        """
        if word.lower() in self.dictionary:
            return []

        suggestions = []
        word_lower = word.lower()

        for dict_word in self.dictionary:
            if abs(len(dict_word) - len(word_lower)) > max_distance:
                continue

            distance = self.levenshtein_distance(word_lower, dict_word)
            if distance <= max_distance:
                suggestions.append((dict_word, distance))

        # Trie par distance et retourne les meilleurs
        suggestions.sort(key=lambda x: x[1])
        return [s[0] for s in suggestions[:max_suggestions]]
