"""
Auto-Indexer: File watcher + decouverte dynamique des fichiers.
Detecte automatiquement les nouveaux fichiers .md et les modifications.
"""

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

import sys


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class FileChangeHandler(FileSystemEventHandler):
    """Gestionnaire d'evenements pour les modifications de fichiers."""

    def __init__(self, callback: Callable[[str, str], None], debounce_seconds: float = 1.0):
        """
        Args:
            callback: Fonction appelee lors d'un changement (filepath, event_type)
            debounce_seconds: Delai anti-rebond pour eviter les appels multiples
        """
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._pending_events: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._debounce_thread: Optional[threading.Thread] = None
        self._running = True
        self._start_debounce_thread()

    def _start_debounce_thread(self):
        """Demarre le thread de debounce."""
        def debounce_loop():
            while self._running:
                time.sleep(0.1)
                with self._lock:
                    now = time.time()
                    to_process = []
                    for filepath, (event_type, timestamp) in list(self._pending_events.items()):
                        if now - timestamp >= self.debounce_seconds:
                            to_process.append((filepath, event_type))
                            del self._pending_events[filepath]

                for filepath, event_type in to_process:
                    try:
                        self.callback(filepath, event_type)
                    except Exception as e:
                        log(f"Erreur callback file change: {e}")

        self._debounce_thread = threading.Thread(target=debounce_loop, daemon=True)
        self._debounce_thread.start()

    def stop(self):
        """Arrete le thread de debounce."""
        self._running = False

    def _schedule_event(self, filepath: str, event_type: str):
        """Planifie un evenement avec debounce."""
        if not filepath.endswith('.md'):
            return

        with self._lock:
            self._pending_events[filepath] = (event_type, time.time())

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule_event(event.src_path, "modified")

    def on_created(self, event):
        if not event.is_directory:
            self._schedule_event(event.src_path, "created")

    def on_deleted(self, event):
        if not event.is_directory:
            self._schedule_event(event.src_path, "deleted")


class AutoIndexer:
    """
    Systeme d'indexation automatique avec:
    - Decouverte dynamique des fichiers .md
    - File watcher pour les modifications en temps reel
    - Hash tracking pour eviter les re-indexations inutiles
    """

    def __init__(
        self,
        knowledge_path: Path,
        state_path: Path,
        index_callback: Callable[[Path], int],
        delete_callback: Callable[[str], None],
        rebuild_callback: Optional[Callable[[], None]] = None
    ):
        """
        Args:
            knowledge_path: Chemin vers le dossier /knowledge
            state_path: Chemin vers le fichier d'etat (hashes)
            index_callback: Fonction d'indexation (filepath) -> nb_chunks
            delete_callback: Fonction de suppression (rel_path) -> None
            rebuild_callback: Fonction de rebuild index hybride (optionnel)
        """
        self.knowledge_path = knowledge_path
        self.state_path = state_path
        self.index_callback = index_callback
        self.delete_callback = delete_callback
        self.rebuild_callback = rebuild_callback

        self._file_hashes: dict[str, str] = {}
        self._observer: Optional[Observer] = None
        self._handler: Optional[FileChangeHandler] = None

        self._load_state()

    def _load_state(self):
        """Charge l'etat precedent (hashes des fichiers)."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._file_hashes = data.get("file_hashes", {})
                    log(f"  Etat charge: {len(self._file_hashes)} fichiers connus")
            except Exception as e:
                log(f"  Erreur chargement etat: {e}")
                self._file_hashes = {}

    def _save_state(self):
        """Sauvegarde l'etat actuel."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "file_hashes": self._file_hashes,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            log(f"  Erreur sauvegarde etat: {e}")

    def _compute_hash(self, filepath: Path) -> str:
        """Calcule le hash MD5 d'un fichier."""
        try:
            content = filepath.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def discover_files(self) -> list[Path]:
        """Decouvre tous les fichiers .md dans /knowledge (recursif)."""
        if not self.knowledge_path.exists():
            return []

        # Recursif : inclut les sous-dossiers
        files = list(self.knowledge_path.glob("**/*.md"))
        # Tri par chemin relatif pour consistance
        return sorted(files, key=lambda f: str(f.relative_to(self.knowledge_path)))

    def _get_relative_path(self, filepath: Path) -> str:
        """Retourne le chemin relatif d'un fichier par rapport à knowledge_path."""
        try:
            return str(filepath.relative_to(self.knowledge_path))
        except ValueError:
            return filepath.name

    def needs_reindex(self, filepath: Path) -> bool:
        """Verifie si un fichier doit etre reindexe."""
        rel_path = self._get_relative_path(filepath)
        current_hash = self._compute_hash(filepath)

        if rel_path not in self._file_hashes:
            return True

        return self._file_hashes[rel_path] != current_hash

    def mark_indexed(self, filepath: Path):
        """Marque un fichier comme indexe."""
        rel_path = self._get_relative_path(filepath)
        self._file_hashes[rel_path] = self._compute_hash(filepath)
        self._save_state()

    def mark_deleted(self, rel_path: str):
        """Marque un fichier comme supprime (par chemin relatif)."""
        if rel_path in self._file_hashes:
            del self._file_hashes[rel_path]
            self._save_state()

    def initial_index(self) -> dict[str, int]:
        """
        Indexation initiale intelligente.
        Ne reindexe que les fichiers modifies depuis la derniere fois.

        Returns:
            Dict {rel_path: nb_chunks}
        """
        stats = {}
        files = self.discover_files()

        for filepath in files:
            rel_path = self._get_relative_path(filepath)

            if self.needs_reindex(filepath):
                log(f"  Indexation: {rel_path}")
                try:
                    chunks = self.index_callback(filepath)
                    self.mark_indexed(filepath)
                    stats[rel_path] = chunks
                except Exception as e:
                    log(f"  Erreur indexation {rel_path}: {e}")
                    stats[rel_path] = 0
            else:
                log(f"  Skip (inchange): {rel_path}")
                # On compte quand meme les chunks existants
                stats[rel_path] = -1  # -1 = skipped

        # Nettoie les fichiers supprimes (compare par chemin relatif)
        known_paths = {self._get_relative_path(f) for f in files}
        for rel_path in list(self._file_hashes.keys()):
            if rel_path not in known_paths:
                log(f"  Supprime de l'index: {rel_path}")
                self.delete_callback(rel_path)
                self.mark_deleted(rel_path)

        return stats

    def force_reindex_all(self) -> dict[str, int]:
        """Force la reindexation de tous les fichiers."""
        self._file_hashes = {}
        return self.initial_index()

    def _on_file_change(self, filepath: str, event_type: str):
        """Callback pour les changements de fichiers."""
        path = Path(filepath)
        rel_path = self._get_relative_path(path)

        if event_type == "deleted":
            log(f"  [WATCH] Fichier supprime: {rel_path}")
            self.delete_callback(rel_path)
            self.mark_deleted(rel_path)
        else:
            log(f"  [WATCH] Fichier modifie: {rel_path}")
            if path.exists():
                try:
                    chunks = self.index_callback(path)
                    self.mark_indexed(path)
                    log(f"  [WATCH] Reindexe: {chunks} chunks")
                    # Appelle rebuild_callback si disponible (pour BM25)
                    if self.rebuild_callback:
                        self.rebuild_callback()
                        log(f"  [WATCH] Index hybride reconstruit")
                except Exception as e:
                    log(f"  [WATCH] Erreur reindexation: {e}")

    def start_watching(self):
        """Demarre le file watcher."""
        if self._observer is not None:
            return

        self._handler = FileChangeHandler(self._on_file_change)
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.knowledge_path),
            recursive=True  # Surveille les sous-dossiers
        )
        self._observer.start()
        log(f"  File watcher demarre sur {self.knowledge_path} (recursif)")

    def stop_watching(self):
        """Arrete le file watcher."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._handler is not None:
            self._handler.stop()
            self._handler = None

        log("  File watcher arrete")
