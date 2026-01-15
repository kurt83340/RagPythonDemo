# Benchmarks

Performance evaluation of the RAG system on a 1000-point scale.

## Overall Score: 855.4 / 1000 (A+)

| Category | Score | Max | Percentage |
|----------|-------|-----|------------|
| Semantic Relevance | 249.4 | 300 | 83% |
| Lexical Precision | 150 | 150 | 100% |
| Reranker Quality | 206 | 300 | 69% |
| Discrimination | 150 | 150 | 100% |
| Latency | 100 | 100 | 100% |

## Category Details

### Semantic Relevance (83%)
Measures how well the system finds semantically similar content, even with different wording.

- Tests paraphrase matching
- Cross-language queries (FR/EN)
- Synonym handling

### Lexical Precision (100%)
Measures exact keyword matching via BM25.

- Technical terms
- Function names
- Exact phrases

### Reranker Quality (69%)
Measures cross-encoder reranking effectiveness.

- Re-ordering by true relevance
- Filtering out false positives
- Room for improvement with heavier models

### Discrimination (100%)
Measures ability to distinguish relevant from irrelevant content.

- No false positives in top results
- Clear separation between matches and non-matches

### Latency (100%)
Response time under threshold.

- First query: ~2s (model loading)
- Cached queries: <100ms
- Average: <500ms

## Fine-Tuning Results

8 configurations tested with varying parameters:

| Parameter | Values Tested |
|-----------|---------------|
| Query Expansion | ON / OFF |
| RRF Weights | 0.4/0.6, 0.5/0.5, 0.6/0.4, 0.8/0.2, 1.0/0.0 |

**Result**: All configurations scored ~800/1000

**Conclusion**: The cross-encoder reranker normalizes results across configurations. The final stage reranking is the dominant factor in result quality.

## Models Used

| Component | Model | Why |
|-----------|-------|-----|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Best multilingual balance |
| Reranker | `mmarco-mMiniLMv2-L12-H384-v1` | Fast + multilingual |

## Potential Improvements

To improve the 69% reranker score:
- Upgrade to `ms-marco-MiniLM-L-12-v2` (heavier, better quality)
- Use `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed/quality balance
- Fine-tune on domain-specific data

Current score of 855/1000 is sufficient for most knowledge base use cases.
