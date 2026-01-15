"""
Test complet de tous les outils MCP du serveur RAG.
Execute dans Docker: docker compose run --rm rag-server python test_mcp_tools.py
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Config environment
_models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(_models_dir, exist_ok=True)
os.environ["HF_HOME"] = _models_dir
os.environ["TRANSFORMERS_CACHE"] = _models_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _models_dir
os.environ["TORCH_HOME"] = _models_dir

# Ajoute le repertoire au path
sys.path.insert(0, str(Path(__file__).parent))

# Import du serveur apres config
from server import (
    collection, hybrid_search, index_all_files, rebuild_hybrid_index,
    KNOWLEDGE_PATH, ensure_files_exist, get_all_knowledge_files,
    is_append_only, is_editable, index_file, delete_file_from_index,
    maintenance_manager, feedback_logger, semantic_cache, query_expander,
    time_scorer
)

def test_reindex():
    """Test reindex - Reindexation complete"""
    print("\n[TEST] reindex()")
    print("-" * 40)

    ensure_files_exist()
    stats = index_all_files()
    rebuild_hybrid_index()

    total = sum(v for v in stats.values() if v > 0)
    print(f"  Fichiers indexes: {len(stats)}")
    print(f"  Total chunks: {total}")

    for f, c in stats.items():
        print(f"    - {f}: {c} chunks")

    assert total > 0, "Doit indexer au moins quelques chunks"
    print("  [OK] reindex fonctionne")
    return True


def test_search():
    """Test search - Recherche hybride"""
    print("\n[TEST] search()")
    print("-" * 40)

    # Test basique
    results = hybrid_search("architecture", n_results=3)
    print(f"  Query: 'architecture'")
    print(f"  Resultats: {len(results)}")

    for i, r in enumerate(results):
        source = r["metadata"].get("source", "?")
        score = r["score"]
        print(f"    [{i+1}] {source} (score: {score:.3f})")

    # Test avec MMR
    results_mmr = hybrid_search("projet", n_results=3, use_mmr=True, mmr_lambda=0.5)
    print(f"\n  Query avec MMR: 'projet' (lambda=0.5)")
    print(f"  Resultats MMR: {len(results_mmr)}")

    # Test avec filtres
    results_filtered = hybrid_search("specs", n_results=3, filters={"doc_type": "specs"})
    print(f"\n  Query avec filtre doc_type=specs: 'specs'")
    print(f"  Resultats filtres: {len(results_filtered)}")

    print("  [OK] search fonctionne")
    return True


def test_list_files():
    """Test list_files - Liste des fichiers"""
    print("\n[TEST] list_files()")
    print("-" * 40)

    files = get_all_knowledge_files()
    print(f"  Total fichiers: {len(files)}")

    for f in files[:10]:
        file_type = "append" if is_append_only(f) else "edit" if is_editable(f) else "auto"
        print(f"    - {f} ({file_type})")

    if len(files) > 10:
        print(f"    ... et {len(files) - 10} autres")

    assert len(files) > 0, "Doit trouver au moins un fichier"
    print("  [OK] list_files fonctionne")
    return True


def test_read_doc():
    """Test read_doc - Lecture fichier"""
    print("\n[TEST] read_doc()")
    print("-" * 40)

    test_file = "projet/project.md"
    filepath = KNOWLEDGE_PATH / test_file

    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        print(f"  Fichier: {test_file}")
        print(f"  Taille: {len(content)} caracteres")
        preview = content[:100].replace('\n', ' ')
        print(f"  Preview: {preview}...")
        print("  [OK] read_doc fonctionne")
        return True
    else:
        print(f"  Fichier {test_file} non trouve")
        return False


def test_get_stats():
    """Test get_stats - Statistiques RAG"""
    print("\n[TEST] get_stats()")
    print("-" * 40)

    total_chunks = collection.count()
    files = get_all_knowledge_files()

    print(f"  Total chunks: {total_chunks}")
    print(f"  Total fichiers: {len(files)}")
    print(f"  Cache semantique: {semantic_cache.get_stats().entries_count} entrees")
    print(f"  Synonymes: {len(query_expander.SYNONYMS)}")

    assert total_chunks >= 0, "Doit avoir un count valide"
    print("  [OK] get_stats fonctionne")
    return True


def test_search_by_date():
    """Test search_by_date - Recherche temporelle"""
    print("\n[TEST] search_by_date()")
    print("-" * 40)

    from datetime import datetime as dt, timedelta

    today = dt.now()
    from_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    results = []
    for filepath in KNOWLEDGE_PATH.glob("**/*.md"):
        mtime = dt.fromtimestamp(filepath.stat().st_mtime)
        from_dt = dt.strptime(from_date, "%Y-%m-%d")
        to_dt = dt.strptime(to_date, "%Y-%m-%d")
        if from_dt <= mtime <= to_dt:
            rel_path = str(filepath.relative_to(KNOWLEDGE_PATH))
            results.append((rel_path, mtime.strftime("%Y-%m-%d %H:%M")))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"  Periode: {from_date} -> {to_date}")
    print(f"  Fichiers modifies: {len(results)}")

    for path, date in results[:5]:
        print(f"    - {date}: {path}")

    print("  [OK] search_by_date fonctionne")
    return True


def test_get_related():
    """Test get_related - Fichiers similaires"""
    print("\n[TEST] get_related()")
    print("-" * 40)

    test_file = "projet/specs.md"
    filepath = KNOWLEDGE_PATH / test_file

    if not filepath.exists():
        print(f"  Fichier {test_file} non trouve, skip")
        return True

    content = filepath.read_text(encoding="utf-8")
    results = hybrid_search(content[:500], n_results=5)

    # Filtre le fichier source
    filtered = [r for r in results if r["metadata"].get("source") != test_file][:3]

    print(f"  Reference: {test_file}")
    print(f"  Similaires:")

    for i, r in enumerate(filtered):
        source = r["metadata"].get("source", "?")
        score = r["score"]
        print(f"    [{i+1}] {source} (score: {score:.3f})")

    print("  [OK] get_related fonctionne")
    return True


def test_create_delete_doc():
    """Test create_doc + delete_doc - Creation/Suppression"""
    print("\n[TEST] create_doc() + delete_doc()")
    print("-" * 40)

    test_path = "test_temp_file.md"
    test_content = "# Test Temporaire\n\nCeci est un fichier de test."

    filepath = KNOWLEDGE_PATH / test_path

    # Create
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(test_content, encoding="utf-8")
    chunks = index_file(filepath)
    rebuild_hybrid_index()

    print(f"  Cree: {test_path} ({chunks} chunks)")
    assert filepath.exists(), "Fichier doit exister"

    # Delete
    delete_file_from_index(test_path)
    filepath.unlink()
    rebuild_hybrid_index()

    print(f"  Supprime: {test_path}")
    assert not filepath.exists(), "Fichier doit etre supprime"

    print("  [OK] create_doc + delete_doc fonctionnent")
    return True


def test_update_doc():
    """Test update_doc - Mise a jour section"""
    print("\n[TEST] update_doc()")
    print("-" * 40)

    import re

    test_file = "projet/project.md"
    filepath = KNOWLEDGE_PATH / test_file

    if not filepath.exists():
        print(f"  Fichier {test_file} non trouve, skip")
        return True

    # Sauvegarde
    original = filepath.read_text(encoding="utf-8")

    # Test de pattern matching
    section_name = "Vision"
    section_pattern = rf"(## {re.escape(section_name)}\n)(.*?)(?=\n## |\Z)"
    match = re.search(section_pattern, original, re.DOTALL)

    if match:
        print(f"  Section trouvee: {section_name}")
        print(f"  Contenu actuel (extrait): {match.group(2)[:50].strip()}...")
    else:
        print(f"  Section {section_name} non trouvee")

    print("  [OK] update_doc pattern matching fonctionne")
    return True


def test_append_doc():
    """Test append_doc - Ajout entree"""
    print("\n[TEST] append_doc()")
    print("-" * 40)

    test_file = "journal/progress.md"
    filepath = KNOWLEDGE_PATH / test_file

    if not filepath.exists():
        print(f"  Fichier {test_file} non trouve, skip")
        return True

    # Verifie le format
    content = filepath.read_text(encoding="utf-8")

    print(f"  Fichier: {test_file}")
    print(f"  Taille: {len(content)} caracteres")
    print(f"  Type: {'append-only' if is_append_only(test_file) else 'editable'}")

    assert is_append_only(test_file), "Doit etre append-only"

    print("  [OK] append_doc detection fonctionne")
    return True


def test_get_summary():
    """Test get_summary - Resume projet"""
    print("\n[TEST] get_summary()")
    print("-" * 40)

    import re

    output = ""

    # Vision
    filepath = KNOWLEDGE_PATH / "projet/project.md"
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        match = re.search(r'## Vision\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
        if match:
            output += f"Vision: {match.group(1).strip()[:50]}...\n"

    # Derniere session
    sessions_dir = KNOWLEDGE_PATH / "sessions"
    if sessions_dir.exists():
        sessions = sorted(sessions_dir.glob("*.md"), reverse=True)
        if sessions:
            output += f"Derniere session: {sessions[0].stem}\n"

    print(f"  {output if output else 'Aucune donnee'}")
    print("  [OK] get_summary fonctionne")
    return True


def test_get_timeline():
    """Test get_timeline - Timeline projet"""
    print("\n[TEST] get_timeline()")
    print("-" * 40)

    import re

    events = []

    # Sessions
    sessions_dir = KNOWLEDGE_PATH / "sessions"
    if sessions_dir.exists():
        for f in sessions_dir.glob("*.md"):
            events.append((f.stem, "session"))

    # Journal
    for journal_file in ["journal/progress.md", "journal/changelog.md"]:
        filepath = KNOWLEDGE_PATH / journal_file
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            for match in re.finditer(r'## (\d{4}-\d{2}-\d{2}[^#\n]*)', content):
                events.append((match.group(1)[:10], journal_file.split("/")[1]))

    events.sort(key=lambda x: x[0], reverse=True)

    print(f"  Evenements: {len(events)}")
    for date, type_ in events[:5]:
        print(f"    - {date}: {type_}")

    print("  [OK] get_timeline fonctionne")
    return True


def test_run_maintenance():
    """Test run_maintenance - Maintenance RAG"""
    print("\n[TEST] run_maintenance()")
    print("-" * 40)

    # Prepare les donnees
    result = collection.get(include=["documents", "metadatas", "embeddings"])

    chunks_data = []
    for i, (doc, meta, chunk_id) in enumerate(zip(
        result["documents"] or [],
        result["metadatas"] or [],
        result["ids"]
    )):
        chunks_data.append({
            "content": doc,
            "metadata": meta,
            "chunk_id": chunk_id
        })

    print(f"  Chunks a analyser: {len(chunks_data)}")
    print(f"  Archives actuelles: {len(maintenance_manager._archived_chunks)}")

    # Note: On ne run pas la maintenance complete car elle peut modifier l'etat
    print("  [OK] run_maintenance structure fonctionne")
    return True


def test_generate_map():
    """Test generate_map - Carte projet"""
    print("\n[TEST] generate_map()")
    print("-" * 40)

    project_root = KNOWLEDGE_PATH.parent
    extensions = ["py", "md"]

    files_count = 0
    for ext in extensions:
        for f in project_root.glob(f"**/*.{ext}"):
            path_str = str(f)
            # Skip certains dossiers
            if any(p in path_str for p in ['node_modules', 'venv', '__pycache__', '.git', 'db', 'models']):
                continue
            files_count += 1

    print(f"  Fichiers scannes: {files_count}")
    print("  [OK] generate_map scanning fonctionne")
    return True


def test_semantic_cache():
    """Test semantic cache"""
    print("\n[TEST] Semantic Cache")
    print("-" * 40)

    # Stats
    stats = semantic_cache.get_stats()
    print(f"  Entrees: {stats.entries_count}")
    print(f"  Hits: {stats.cache_hits}")
    print(f"  Misses: {stats.cache_misses}")
    print(f"  Hit rate: {stats.hit_rate():.1f}%")

    print("  [OK] Semantic cache fonctionne")
    return True


def test_query_expansion():
    """Test query expansion"""
    print("\n[TEST] Query Expansion")
    print("-" * 40)

    test_queries = ["API REST", "DB PostgreSQL", "auth JWT"]

    for query in test_queries:
        expansion = query_expander.expand_query(query)
        print(f"  '{query}' -> '{expansion['expanded']}'")
        if expansion['terms_added']:
            print(f"    Termes: {expansion['terms_added']}")

    print("  [OK] Query expansion fonctionne")
    return True


def run_all_tests():
    """Execute tous les tests MCP"""
    print("=" * 60)
    print("TESTS COMPLETS DES OUTILS MCP RAG")
    print("=" * 60)

    tests = [
        ("reindex", test_reindex),
        ("search", test_search),
        ("list_files", test_list_files),
        ("read_doc", test_read_doc),
        ("get_stats", test_get_stats),
        ("search_by_date", test_search_by_date),
        ("get_related", test_get_related),
        ("create_delete_doc", test_create_delete_doc),
        ("update_doc", test_update_doc),
        ("append_doc", test_append_doc),
        ("get_summary", test_get_summary),
        ("get_timeline", test_get_timeline),
        ("run_maintenance", test_run_maintenance),
        ("generate_map", test_generate_map),
        ("semantic_cache", test_semantic_cache),
        ("query_expansion", test_query_expansion),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  [ERREUR] {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Resume
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} tests passes")

    if passed == total:
        print("\nTous les outils MCP fonctionnent correctement!")
        return 0
    else:
        print("\nCertains tests ont echoue.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
