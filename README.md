# RAG Python Demo

A production-ready **Retrieval-Augmented Generation (RAG)** server implementing the Model Context Protocol (MCP).

## Features

- **Hybrid Search**: Combines semantic embeddings with BM25 keyword search
- **Cross-Encoder Reranking**: Uses `mmarco-mMiniLMv2-L12-H384-v1` for relevance scoring
- **MMR Diversity**: Maximal Marginal Relevance to avoid redundant results
- **Query Expansion**: Automatic query reformulation for better recall
- **Contextual Retrieval**: Chunk enrichment with document context
- **Semantic Cache**: LRU cache with embedding similarity for faster responses
- **PDF Support**: Extract and index PDF documents with header metadata
- **Time-Weighted Scoring**: Boost recent documents when relevant

## Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector DB | ChromaDB (persistent) |
| Protocol | MCP (Model Context Protocol) |
| Runtime | Docker |

## Architecture

```
rag-server/
├── server.py              # MCP server & tool definitions
├── search.py              # Search orchestration
├── hybrid_search.py       # BM25 + semantic fusion
├── reranker.py            # Cross-encoder + MMR
├── chunker.py             # Document chunking strategies
├── query_expander.py      # Query expansion module
├── contextual_retrieval.py # Chunk enrichment
├── semantic_cache.py      # Embedding-based cache
├── pdf_processor.py       # PDF extraction
├── indexing.py            # Document indexing
├── maintenance.py         # Cleanup & optimization
└── config.py              # Configuration
```

## Quick Start

```bash
# Clone and run
docker compose up

# The MCP server will be available for integration
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search(query)` | Hybrid search with reranking |
| `search_by_date(from, to)` | Temporal filtering |
| `get_related(file)` | Find similar documents |
| `reindex()` | Rebuild search index |
| `get_stats()` | System statistics |

## Configuration

Key parameters in `config.py`:
- `TOP_K`: Number of results (default: 10)
- `MMR_LAMBDA`: Diversity vs relevance trade-off (0-1)
- `CACHE_TTL`: Cache expiration time
- `CHUNK_SIZE`: Document chunk size

## License

MIT
