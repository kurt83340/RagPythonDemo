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

## 5. Example Workflow

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

## 6. Performance Tips

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

## 7. Troubleshooting

| Problem | Solution |
|---------|----------|
| "No results found" | Check files are in `knowledge/`, run `reindex()` |
| Slow startup | First run downloads models, subsequent starts are fast |
| Container not starting | Check `docker logs rag-server` |
| MCP not connecting | Ensure container is running: `docker ps` |
