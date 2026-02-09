# Retrieval Pipeline Usage Guide

Quick reference for using the Phase 2 retrieval components.

---

## Basic Usage

### Initialize Retriever

```python
from src.config import BenchmarkConfig
from src.retrieval import Retriever

# Load config and unified graph (from Phase 1)
config = BenchmarkConfig()
# unified_graph = ... (from Phase 1)

# Build retriever (uses checkpoints if available)
retriever = Retriever.build_from_checkpoint_or_new(
    config=config,
    unified_graph=unified_graph
)
```

**First run:** ~5-10 minutes (builds embeddings and FAISS index)
**Subsequent runs:** ~10 seconds (loads from checkpoints)

---

### Retrieve Subgraph for a Question

```python
# Ask a question
question = "Who is Justin Bieber's brother?"

# Retrieve subgraph
result = retriever.retrieve(question)

# Access results
print(f"Question: {result.question}")
print(f"Subgraph: {result.num_nodes} nodes, {result.num_edges} edges")
print(f"Top entities: {result.seed_entities}")
print(f"Time: {result.retrieval_time_ms:.1f}ms")

# Use the subgraph
subgraph = result.subgraph  # NetworkX DiGraph
print(f"Nodes: {list(subgraph.nodes())[:5]}")
print(f"Edges: {list(subgraph.edges(data=True))[:3]}")
```

**Output:**
```
Question: Who is Justin Bieber's brother?
Subgraph: 42 nodes, 98 edges
Top entities: ['Justin Bieber', 'Jaxon Bieber', ...]
Time: 234.5ms
Nodes: ['Justin Bieber', 'Jaxon Bieber', 'Jazmyn Bieber', ...]
Edges: [('Justin Bieber', 'Jaxon Bieber', {'relation': 'people.person.sibling_s'}), ...]
```

---

## RetrievedSubgraph Object

```python
@dataclass
class RetrievedSubgraph:
    subgraph: nx.DiGraph           # Extracted subgraph
    question: str                  # Original question
    seed_entities: List[str]       # Top-k from k-NN
    similarity_scores: Dict[str, float]  # Entity → score
    num_nodes: int                 # Subgraph size
    num_edges: int                 # Edge count
    retrieval_time_ms: float       # Latency
    pcst_used: bool                # True if PCST, False if BFS
```

**Example:**
```python
result = retriever.retrieve("What movies did Tom Hanks star in?")

# Check if answer entity is in subgraph
answer_entity = "Forrest Gump"
if answer_entity in result.subgraph.nodes():
    print(f"✓ Found {answer_entity} in subgraph")

# Get similarity scores
top_entity = result.seed_entities[0]
score = result.similarity_scores[top_entity]
print(f"Top entity: {top_entity} (score: {score:.3f})")

# Check algorithm used
if result.pcst_used:
    print("PCST extraction successful")
else:
    print("Used BFS fallback")
```

---

## Component Usage

### TextEmbedder

```python
from src.retrieval import TextEmbedder

embedder = TextEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

# Embed a query
queries = ["Who is Justin Bieber?", "What is the capital of France?"]
embeddings = embedder.embed_texts(queries, batch_size=2)
print(embeddings.shape)  # (2, 384)

# Embed graph entities
entity_embeddings = embedder.embed_graph_entities(unified_graph)
print(len(entity_embeddings))  # 1023103

# Embed relations
relation_embeddings = embedder.embed_relations(unified_graph)
print(len(relation_embeddings))  # 5622
```

---

### EntityIndex

```python
from src.retrieval import EntityIndex
import numpy as np

# Build index
index = EntityIndex(embedding_dim=384)
index.build(entity_embeddings)

# Search
query_embedding = embedder.embed_texts(["Justin Bieber"])[0]
results = index.search(query_embedding, k=10)

for entity, score in results:
    print(f"{entity}: {score:.3f}")

# Save/load
index.save(
    index_path=Path("faiss_index.bin"),
    mapping_path=Path("entity_mapping.pkl")
)

index2 = EntityIndex(embedding_dim=384)
index2.load(
    index_path=Path("faiss_index.bin"),
    mapping_path=Path("entity_mapping.pkl")
)
```

---

### PCSTSolver

```python
from src.retrieval import PCSTSolver

solver = PCSTSolver(cost=1.0, budget=50)

# Seed nodes from k-NN search
seed_nodes = ["Justin Bieber", "Jaxon Bieber", "Pattie Mallette"]

# Prize assignment (rank-based)
prizes = {
    "Justin Bieber": 10,
    "Jaxon Bieber": 9,
    "Pattie Mallette": 8
}

# Extract subgraph
subgraph = solver.extract_subgraph(
    G=unified_graph,
    seed_nodes=seed_nodes,
    prizes=prizes
)

print(f"Extracted {len(subgraph)} nodes")

# Validate
is_valid = solver.validate_subgraph(subgraph)
print(f"Valid: {is_valid}")
```

---

## Configuration

All retrieval parameters in `BenchmarkConfig`:

```python
from src.config import BenchmarkConfig

config = BenchmarkConfig(
    # Retrieval settings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dim=384,
    top_k_entities=10,
    pcst_budget=50,

    # Other settings...
    seed=42,
    deterministic=True,
)
```

**Parameters:**
- `embedding_model`: Sentence-Transformers model name
- `embedding_dim`: Embedding dimension (384 for MiniLM)
- `top_k_entities`: Number of neighbors from k-NN search
- `pcst_budget`: Max nodes in extracted subgraph

---

## Batch Processing

Process multiple questions:

```python
questions = [
    "Who is Justin Bieber's brother?",
    "What movies did Tom Hanks star in?",
    "Where was Barack Obama born?",
]

results = []
for question in questions:
    result = retriever.retrieve(question)
    results.append(result)
    print(f"Q: {question}")
    print(f"   → {result.num_nodes} nodes, {result.retrieval_time_ms:.1f}ms")

# Aggregate statistics
avg_nodes = sum(r.num_nodes for r in results) / len(results)
avg_time = sum(r.retrieval_time_ms for r in results) / len(results)

print(f"\nAverage: {avg_nodes:.1f} nodes, {avg_time:.1f}ms")
```

---

## Validation Example

Check if retrieval finds answer entities:

```python
from datasets import load_dataset

# Load validation split
dataset = load_dataset("rmanluo/RoG-webqsp", split="validation[:100]")

hit_count = 0
for example in dataset:
    question = example["question"]
    answer_entities = example.get("a_entity", [])

    # Retrieve
    result = retriever.retrieve(question)
    subgraph_nodes = set(result.subgraph.nodes())

    # Check hit
    hit = any(ans in subgraph_nodes for ans in answer_entities)
    if hit:
        hit_count += 1

hit_rate = hit_count / len(dataset) * 100
print(f"Hit rate: {hit_rate:.1f}% ({hit_count}/{len(dataset)})")
```

**Expected:** Hit rate > 60%

---

## Troubleshooting

### CUDA Out of Memory

If embeddings fail with CUDA OOM:

```python
# Use CPU instead
embedder = TextEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # Force CPU
)
```

### Disconnected Subgraphs

If PCST produces disconnected graphs:

```python
import networkx as nx

result = retriever.retrieve(question)
if not nx.is_weakly_connected(result.subgraph):
    print("⚠ Disconnected subgraph (BFS fallback should prevent this)")
    # Check result.pcst_used
    print(f"PCST used: {result.pcst_used}")
```

### Slow k-NN Search

If search is slow (>1 second):

```python
# Check index size
print(f"Index size: {len(retriever.entity_index)} entities")

# Consider using IVF index (future optimization)
# Current: IndexFlatIP (exact search)
# Future: IndexIVFFlat (approximate search, faster)
```

### Empty Subgraphs

If retrieval returns empty graphs:

```python
result = retriever.retrieve(question)
if result.num_nodes == 0:
    print("⚠ No nodes retrieved")
    print(f"Seed entities: {result.seed_entities}")
    print(f"Similarity scores: {result.similarity_scores}")
    # Check if seed entities exist in unified graph
```

---

## Checkpoints

Checkpoints are saved to Google Drive:

```
/content/drive/MyDrive/arcOS_benchmark/checkpoints/
├── entity_embeddings.pkl        # 1M entity embeddings (~1.5 GB)
├── relation_embeddings.pkl      # 5.6K relation embeddings (~8 MB)
├── faiss_index.bin              # FAISS index (~1.5 GB)
└── entity_mapping.pkl           # ID ↔ name mapping (~50 MB)
```

### Manual Checkpoint Management

```python
from src.utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint

# Check if checkpoint exists
checkpoint_path = config.get_checkpoint_path("entity_embeddings.pkl")
if checkpoint_exists(checkpoint_path):
    print("Checkpoint exists")

# Load checkpoint
embeddings = load_checkpoint(checkpoint_path, format="pickle")

# Delete checkpoint (force rebuild)
# checkpoint_path.unlink()  # Delete file
```

---

## Performance Tips

**Cold Start (First Run):**
1. Ensure GPU is available (`torch.cuda.is_available()`)
2. Expect ~5-10 minutes for embedding computation
3. Checkpoints save automatically to Drive

**Warm Start (Checkpoints Exist):**
1. Loads in ~10 seconds
2. No recomputation needed

**Query Optimization:**
1. Batch multiple queries if possible
2. PCST is the bottleneck (~100-300ms)
3. Consider caching frequent queries

---

## Next Steps

Once retrieval works:
1. **Phase 3:** Convert subgraphs to PyG `Data` objects
2. **Phase 4:** Train GNN to learn attention weights
3. **Phase 5:** Use attention to verbalize subgraph as text
4. **Phase 6:** Send verbalized text to LLM for answer generation

The retrieval pipeline is the foundation for all downstream phases!

---

## References

- [Phase 2 Completion Doc](PHASE2_COMPLETE.md)
- [Sentence-Transformers Docs](https://www.sbert.net/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki)
- [PCST Fast](https://github.com/fraenkel-lab/pcst_fast)
