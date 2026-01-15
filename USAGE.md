# Usage Guide

Complete guide to set up, configure, and use the RAG server.

## 1. Setup

### Prerequisites
- Docker & Docker Compose
- An MCP-compatible client (Claude Desktop, VS Code extension, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/kurt83340/RagPythonDemo.git
cd RagPythonDemo

# Create knowledge folder
mkdir -p knowledge

# Start the server (first run downloads models ~500MB)
docker compose up
```

The server will:
1. Download embedding model (`paraphrase-multilingual-MiniLM-L12-v2`)
2. Download reranker model (`mmarco-mMiniLMv2-L12-H384-v1`)
3. Initialize ChromaDB
4. Index all files in `knowledge/`

## 2. Adding Documents

### Supported Formats
| Format | Extension | Notes |
|--------|-----------|-------|
| Markdown | `.md` | Best for structured docs |
| PDF | `.pdf` | Auto-extracts text + headers |
| Text | `.txt` | Plain text |
| JSON | `.json` | Structured data |

### Folder Structure (Recommended)

```
knowledge/
├── projet/           # Project documentation
│   ├── project.md    # Vision, goals, context
│   ├── specs.md      # Specifications, features
│   └── architecture.md
│
├── journal/          # Chronological logs
│   ├── changelog.md  # Version history
│   ├── bugs.md       # Resolved bugs
│   └── learnings.md  # Technical learnings
│
└── references/       # Technical docs
    └── api.md
```

### Optimizing Your Documents

**Best practices for better retrieval:**

1. **Use clear headers** - The chunker extracts header hierarchy
   ```markdown
   # Main Topic
   ## Subtopic
   ### Detail
   Content here will be tagged with header context
   ```

2. **Keep sections focused** - One concept per section (~200-500 words)

3. **Use keywords** - Include relevant terms for BM25 matching

4. **Add dates** - For time-weighted scoring
   ```markdown
   ## 2024-01-15 - Feature X implemented
   ```

5. **Avoid huge files** - Split into logical documents

### Adding Files

**Option 1: Direct copy**
```bash
# Copy files to knowledge folder
cp my-docs/*.md knowledge/

# Reindex
# Use the reindex() MCP tool or restart server
```

**Option 2: MCP import tool**
```
import_file(path="/path/to/file.pdf", folder="references")
```

**Option 3: MCP create tool**
```
create_doc(path="projet/roadmap.md", content="# Roadmap\n\n...")
```

## 3. Connecting to MCP Client

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "docker",
      "args": ["exec", "-i", "rag-server", "python", "server.py"]
    }
  }
}
```

### VS Code (Cline/Continue)

Configure your MCP settings to connect to the running container.

## 4. Using the RAG

### Basic Search

```
search(query="authentication flow")
```

Returns ranked results with:
- Relevance score
- Source file
- Content snippet
- Header context

### Search with Filters

```
search(
  query="bug fix",
  source="bugs",           # Filter by file
  date_from="2024-01-01",  # Temporal filter
  n_results=10             # More results
)
```

### MMR Diversity

Avoid redundant results with MMR (Maximal Marginal Relevance):

```
search(
  query="API endpoints",
  use_mmr=true,
  mmr_lambda=0.7   # 0.5=balanced, 0.7=relevance, 1.0=no diversity
)
```

### Other Tools

| Tool | Usage |
|------|-------|
| `get_stats()` | View indexed files count, models, version |
| `reindex()` | Rebuild index after manual file changes |
| `get_related(file)` | Find similar documents |
| `search_by_date(from, to)` | Pure temporal search |
| `read_doc(file)` | Read entire file content |
| `list_files(folder)` | List files in folder |

## 5. Quick Test (Copy-Paste Examples)

Test the RAG immediately with these sample files.

### Step 1: Create test files

**knowledge/api-docs.md**
```markdown
# REST API Documentation

## Authentication

All API requests require a Bearer token in the Authorization header.

To obtain a token, call POST /auth/login with your credentials:
- email: your registered email
- password: your password

The token expires after 24 hours.

## Endpoints

### GET /users
Returns a list of all users. Requires admin role.

Parameters:
- limit (int): Maximum results (default: 50)
- offset (int): Pagination offset

### POST /users
Creates a new user account.

Required fields:
- email: Valid email address
- password: Minimum 8 characters
- name: Display name

### GET /products
Returns product catalog.

Query parameters:
- category: Filter by category
- min_price / max_price: Price range
- sort: "price_asc", "price_desc", "newest"
```

**knowledge/architecture.md**
```markdown
# System Architecture

## Tech Stack

- **Backend**: Python FastAPI
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **Search**: Elasticsearch 8
- **Queue**: RabbitMQ

## Services

### API Gateway
Handles authentication, rate limiting, request routing.
Port: 8000

### User Service
Manages user accounts, profiles, permissions.
Port: 8001

### Product Service
Product catalog, inventory, pricing.
Port: 8002

### Order Service
Shopping cart, checkout, order processing.
Port: 8003

## Database Schema

Users table:
- id (UUID, PK)
- email (unique)
- password_hash
- created_at

Products table:
- id (UUID, PK)
- name
- price (decimal)
- category_id (FK)
```

**knowledge/changelog.md**
```markdown
# Changelog

## v2.1.0 - 2024-01-15

### Added
- PDF document support with header extraction
- MMR diversity algorithm for search results
- Semantic caching for faster repeated queries

### Fixed
- Memory leak in embedding model
- Incorrect date parsing for European formats

## v2.0.0 - 2024-01-01

### Added
- Hybrid search (vector + BM25)
- Cross-encoder reranking
- Multi-language support (EN/FR/ES)

### Changed
- Switched from FAISS to ChromaDB
- New chunking strategy with header context

## v1.0.0 - 2023-12-01

### Added
- Initial release
- Basic semantic search
- Markdown file indexing
```

### Step 2: Start and index

```bash
# Start server
docker compose up -d

# Wait for models to download (first time only)
docker logs -f rag-server

# Once ready, reindex via MCP
reindex()
```

### Step 3: Test queries

| Query | Expected Result |
|-------|-----------------|
| `search(query="authentication")` | Returns API auth docs + token info |
| `search(query="database schema")` | Returns architecture with tables |
| `search(query="what's new in v2.1")` | Returns changelog v2.1.0 section |
| `search(query="POST endpoints")` | Returns POST /users from API docs |
| `search(query="Redis")` | Returns tech stack section |

### Step 4: Test advanced features

```python
# MMR diversity - get varied results
search(query="API", use_mmr=True, mmr_lambda=0.5)

# Filter by source
search(query="added", source="changelog")

# Boost recent content
search(query="features", boost_recent=True)

# Get related files
get_related(file="architecture.md")
```

### Step 5: Test PDF (optional)

Drop any PDF into `knowledge/` folder:
```bash
cp ~/Documents/manual.pdf knowledge/
# Then reindex()
```

The PDF processor extracts:
- Text content
- Header hierarchy
- Page numbers as metadata

## 6. Example Workflow

```bash
# 1. Start server
docker compose up -d

# 2. Add your documentation
cp ~/project-docs/*.md knowledge/projet/

# 3. Reindex (via MCP client)
reindex()

# 4. Check stats
get_stats()
# → "15 files indexed, 142 chunks"

# 5. Search
search(query="how to deploy")
# → Returns relevant chunks with scores

# 6. Get full context
read_doc(file="projet/architecture.md")
```

## 7. Performance Tips

### For Better Results

| Issue | Solution |
|-------|----------|
| Irrelevant results | Use more specific keywords |
| Too similar results | Enable MMR with lower lambda |
| Missing recent content | Check `boost_recent=true` |
| Slow first query | Normal - models loading |

### Indexing Large Collections

For 100+ files:
1. Add files in batches
2. Run `reindex()` once
3. First search may be slow (cache warming)

### Restarting

```bash
# Restart server (keeps index)
docker compose restart

# Full rebuild (clears index)
docker compose down -v
docker compose up
```

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| "No results found" | Check files are in `knowledge/`, run `reindex()` |
| Slow startup | First run downloads models, subsequent starts are fast |
| Container not starting | Check `docker logs rag-server` |
| MCP not connecting | Ensure container is running: `docker ps` |
