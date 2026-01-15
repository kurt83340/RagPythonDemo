"""
Tests pour le systeme RAG optimise.
Execute: python test_rag.py
"""

import os
import sys
from pathlib import Path

# Stockage de TOUS les fichiers localement dans le projet
_models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(_models_dir, exist_ok=True)

os.environ["HF_HOME"] = _models_dir                    # Hugging Face general
os.environ["TRANSFORMERS_CACHE"] = _models_dir         # Transformers models
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _models_dir # Sentence-transformers
os.environ["TORCH_HOME"] = _models_dir                 # PyTorch hub
os.environ["XDG_CACHE_HOME"] = _models_dir             # Fallback Linux cache

# Ajoute le repertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

def test_chunker():
    """Test du chunking avec tokens"""
    print("\n[TEST] Chunker")
    print("-" * 40)

    from chunker import TokenChunker

    chunker = TokenChunker(chunk_size=512, chunk_overlap=100)

    # Document de test
    text = """# Document de Test

## Section 1

Ceci est le contenu de la section 1. Elle contient plusieurs phrases
pour tester le chunking intelligent. Le chunker doit decouper ce texte
en morceaux de 512 tokens maximum avec un overlap de 100 tokens.

## Section 2

Voici une autre section avec du contenu different.
Cette section permet de verifier que les separateurs markdown
sont bien pris en compte lors du decoupage.

### Sous-section 2.1

Une sous-section pour tester les separateurs de niveau 3.
"""

    chunks = chunker.chunk_document(text, "test.md")

    print(f"  Nombre de chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk['token_count']} tokens")
        preview = chunk['content'][:80].replace('\n', ' ')
        print(f"    Preview: {preview}...")

    assert len(chunks) > 0, "Doit produire au moins 1 chunk"
    print("  [OK] Chunker fonctionne")
    return True


def test_hybrid_search():
    """Test de la recherche hybride BM25"""
    print("\n[TEST] Hybrid Search (BM25)")
    print("-" * 40)

    from hybrid_search import HybridSearcher

    documents = [
        "Le systeme d'authentification utilise JWT pour les tokens.",
        "La base de donnees PostgreSQL stocke les utilisateurs.",
        "L'API REST expose les endpoints /users et /auth.",
        "Le frontend React communique avec le backend.",
        "Les tests unitaires couvrent 80% du code.",
    ]

    searcher = HybridSearcher(documents)

    # Test BM25
    query = "authentification JWT"
    results = searcher.search_bm25(query, top_k=3)

    print(f"  Query: '{query}'")
    print(f"  Resultats BM25:")
    for idx, score in results:
        print(f"    [{idx}] score={score:.3f}: {documents[idx][:50]}...")

    # Le premier resultat devrait etre sur JWT
    assert results[0][0] == 0, "JWT devrait etre le premier resultat"
    print("  [OK] BM25 fonctionne")

    # Test RRF Fusion
    ranking1 = [(0, 1.0), (1, 0.8), (2, 0.6)]
    ranking2 = [(2, 1.0), (0, 0.5), (3, 0.3)]

    fused = HybridSearcher.reciprocal_rank_fusion([ranking1, ranking2])
    print(f"  RRF Fusion: {fused[:3]}")
    print("  [OK] RRF Fusion fonctionne")

    return True


def test_reranker():
    """Test du cross-encoder reranking"""
    print("\n[TEST] Reranker (Cross-Encoder)")
    print("-" * 40)

    from reranker import Reranker

    reranker = Reranker()

    query = "Comment fonctionne l'authentification ?"
    documents = [
        "La meteo est ensoleille aujourd'hui.",
        "Le systeme d'authentification utilise des tokens JWT signes.",
        "Les utilisateurs se connectent via email et mot de passe.",
        "Le serveur valide les credentials avant d'emettre un token.",
    ]

    results = reranker.rerank(query, documents, top_k=3)

    print(f"  Query: '{query}'")
    print(f"  Reranking:")
    for doc, score in results:
        print(f"    score={score:.3f}: {doc[:50]}...")

    # Le document sur la meteo devrait etre en dernier
    assert "meteo" not in results[0][0].lower(), "Le doc meteo ne devrait pas etre premier"
    print("  [OK] Reranker fonctionne")

    return True


def test_embedding():
    """Test du modele d'embedding multilingue"""
    print("\n[TEST] Embedding Multilingue")
    print("-" * 40)

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    texts_fr = [
        "Le chat dort sur le canape.",
        "Un felin se repose sur le sofa.",
        "La voiture roule sur l'autoroute.",
    ]

    embeddings = model.encode(texts_fr)

    # Calcul similarite cosine
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_01 = cosine_sim(embeddings[0], embeddings[1])
    sim_02 = cosine_sim(embeddings[0], embeddings[2])

    print(f"  '{texts_fr[0]}'")
    print(f"  vs '{texts_fr[1]}': {sim_01:.3f}")
    print(f"  vs '{texts_fr[2]}': {sim_02:.3f}")

    assert sim_01 > sim_02, "Phrases similaires devraient avoir un score plus haut"
    print("  [OK] Embedding multilingue fonctionne")

    return True


def test_chromadb():
    """Test de ChromaDB"""
    print("\n[TEST] ChromaDB")
    print("-" * 40)

    import chromadb
    import tempfile

    # Utilise un repertoire temporaire pour les tests
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        collection = client.get_or_create_collection("test")

        # Ajoute des documents
        collection.add(
            ids=["doc1", "doc2"],
            documents=["Premier document", "Deuxieme document"],
            embeddings=[[0.1] * 384, [0.2] * 384]
        )

        assert collection.count() == 2, "Doit avoir 2 documents"

        # Suppression
        collection.delete(ids=["doc1"])
        assert collection.count() == 1, "Doit avoir 1 document apres suppression"

    print("  [OK] ChromaDB fonctionne")
    return True


def run_all_tests():
    """Execute tous les tests"""
    print("=" * 50)
    print("TESTS RAG OPTIMISE")
    print("=" * 50)

    tests = [
        ("ChromaDB", test_chromadb),
        ("Chunker", test_chunker),
        ("Embedding", test_embedding),
        ("Hybrid Search", test_hybrid_search),
        ("Reranker", test_reranker),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  [ERREUR] {e}")
            results.append((name, False))

    # Resume
    print("\n" + "=" * 50)
    print("RESUME")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} tests passes")

    if passed == total:
        print("\nTous les tests sont OK!")
        return 0
    else:
        print("\nCertains tests ont echoue.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
