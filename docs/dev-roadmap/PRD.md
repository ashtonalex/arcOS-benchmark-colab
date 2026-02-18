# arcOS Benchmark - MVP Product Requirements Document

**Version:** 2.1 (Colab Edition)
**Date:** 2026-02-09
**Last Updated:** 2026-02-18
**Platform:** Google Colab

---

## 1. Product Overview

arcOS Benchmark is a causal question-answering system that combines Graph Neural Network (GNN) structural reasoning with Large Language Model (LLM) generation. It implements a simplified G-Retriever-inspired pipeline where a GNN learns the structure of a knowledge graph, guides verbalization of relevant subgraphs into text (hard prompts), and appends them to LLM queries for grounded causal reasoning.

**Key difference from full G-Retriever:** Instead of injecting continuous soft-prompt embeddings into a locally-hosted LLM, this system uses GNN-guided attention to select and verbalize the most relevant subgraph portions into discrete text. This text-based hard prompt is appended to the user query and sent to any LLM via the OpenRouter API. This makes the system API-compatible, Colab-friendly, and model-agnostic.

---

## 2. Goals

1. **Benchmark reproducibility** - Deterministic inference pipeline with seeded randomness for consistent evaluation across runs
2. **Colab-native** - Entire pipeline runs in a single Google Colab notebook with free-tier GPU (T4, ~15 GB VRAM)
3. **API-driven LLM** - All LLM calls route through OpenRouter, enabling model comparison without local hosting
4. **Graph-grounded answers** - LLM responses are constrained to knowledge graph context, reducing hallucination
5. **Measurable quality** - Quantitative evaluation of answer accuracy against gold-standard answers from RoG-webqsp

---

## 3. Dataset

**Source:** [rmanluo/RoG-webqsp](https://huggingface.co/datasets/rmanluo/RoG-webqsp) (HuggingFace)

| Split      | Rows  |
|------------|-------|
| Train      | 2,830 |
| Validation | 246   |
| Test       | 1,630 |
| **Total**  | **4,706** |

**Dev mode** (for faster iteration): 600 / 50 / 1,628 via `RoGWebQSPLoader.slice_dataset()`

**Schema per example:**

| Field      | Type           | Description                                    |
|------------|----------------|------------------------------------------------|
| `id`       | string         | Unique identifier (e.g., `WebQTrn-0`)          |
| `question` | string         | Natural language question                      |
| `answer`   | list[string]   | Gold-standard answer strings                   |
| `q_entity` | list[string]   | Topic entities mentioned in the question       |
| `a_entity` | list[string]   | Answer entities                                |
| `graph`    | list[triple]   | Knowledge subgraph as `[subject, relation, object]` triples |

**Graph characteristics:**
- Nodes are entity strings (e.g., `"Justin Bieber"`) — after noise filtering
- Raw data contains opaque Freebase MIDs (e.g., `"m.02mjmr"`) which are filtered as CVT nodes
- Edges are Freebase relation strings (e.g., `"people.person.sibling_s"`)
- Subgraphs contain multi-hop paths (up to 4 hops from topic entity)
- Derived from the Freebase knowledge graph (~88M entities, 20K relations)

**Noise filtering (applied at graph construction):**
- CVT node removal: opaque Freebase MIDs matching `^[mg]\.[0-9a-z_]+$`
- Junk relation removal: `freebase.valuenotation.*`, `freebase.type_profile.*`, `type.object.*`, `kg.object_profile.*`, `rdf-schema#*`
- Effect: drops ~15-20% of triples, keeps only semantically meaningful relations

---

## 4. Architecture

### 4.1 Pipeline Overview

```
Query ──> Retrieval ──> GNN Encoding ──> Attention-Guided Verbalization ──> LLM Generation
              |                                    |
         NetworkX graph                    Hard prompt (text)
         + k-NN search                     appended to query
```

### 4.2 Layer Breakdown

#### Layer 1: Graph Storage & Retrieval — IMPLEMENTED

| Aspect      | Detail |
|-------------|--------|
| **Graph DB**    | NetworkX (in-memory, replacing Memgraph) |
| **Embedding**   | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Index**       | FAISS `IndexFlatIP` (exact cosine similarity) |
| **Subgraph extraction** | PCST via `pcst_fast` with BFS localization + component bridging |
| **Max subgraph size** | 70 nodes (configurable via `pcst_budget`) |

**Process:**
1. Load RoG-webqsp dataset from HuggingFace
2. Build NetworkX graph from triple lists (with noise filtering)
3. Encode all node/edge text with Sentence-Transformers
   - Entity embeddings enriched with relation context (e.g., "Cleveland | containedby, time zone, adjoins")
4. Store embeddings in FAISS index
5. At query time: embed query → k-NN search (top 15) → BFS localization (500 nodes) → PCST optimization → component bridging

**PCST parameters:**
- Edge cost: 0.015 (low enough for multi-hop paths to be profitable)
- Budget: 70 nodes
- Local budget: 500 nodes (BFS neighborhood)
- Pruning: Goemans-Williamson ("gw")
- Edge weight alpha: 0.5 (query-aware edge cost scaling)
- Bridge max hops: 6 (for connecting disconnected components)

#### Layer 2: GNN Encoder — IMPLEMENTED

| Aspect      | Detail |
|-------------|--------|
| **Architecture** | GATv2 (primary) or GraphSAGE (alternative) |
| **Layers**       | 3 layers (configurable) |
| **Hidden dim**   | 256 |
| **Heads**        | 4 (GATv2 multi-head attention) |
| **Pooling**      | AttentionPooling (`GlobalAttention`) with gate + feature networks |
| **Framework**    | PyTorch Geometric |
| **Training**     | Focal loss (gamma=2.0) for answer node prediction |
| **Optimizer**    | AdamW (lr=1e-3, weight_decay=1e-4) |

**Process:**
1. Convert retrieved subgraph to PyG `Data` object (node features from Sentence-Transformer embeddings)
2. Broadcast query embedding to all nodes (query conditioning)
3. Pass through GATv2Conv layers with residual connections and LayerNorm
4. Attention pooling scores each node's relevance to the query
5. Output: `GNNOutput(node_embeddings, attention_scores, graph_embedding)`

**Training details:**
- Task: binary classification (answer node vs. non-answer node)
- Loss: focal loss (gamma=2.0) — handles ~2% positive rate
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience=5
- Gradient clipping: max_norm=1.0

#### Layer 3: Attention-Guided Graph Verbalization (Hard Prompt) — TODO

| Aspect      | Detail |
|-------------|--------|
| **Input**    | Subgraph + GNN attention weights |
| **Output**   | Natural language description of the most relevant subgraph portions |
| **Format**   | Ranked triples: `(subject) --[relation]--> (object)` |
| **Max tokens** | ~500 tokens of verbalized context |
| **Top-K triples** | 15 (configurable via `config.top_k_triples`) |

**Process:**
1. Rank nodes/edges by GNN attention scores
2. Select top-K triples (those involving highest-attention nodes)
3. Clean Freebase relation strings to human-readable form
4. Verbalize selected triples into structured text
5. Format as hard prompt:
   ```
   Relevant knowledge graph context:
   1. (Justin Bieber) --[sibling]--> (Jaxon Bieber)
   2. (Justin Bieber) --[born in]--> (London, Ontario)
   ...

   Question: What is the name of Justin Bieber's brother?
   ```

#### Layer 4: LLM Interpretation (via OpenRouter) — TODO

| Aspect      | Detail |
|-------------|--------|
| **API**        | OpenRouter (`https://openrouter.ai/api/v1`) |
| **SDK**        | `openai` Python client with custom `base_url` |
| **Default model** | `google/gemini-2.5-flash-preview` (cost-effective) |
| **Fallback**   | Configurable model fallback chain |
| **Retry**      | Exponential backoff with jitter (tenacity) |
| **Temperature** | 0.0 (deterministic) |

**Process:**
1. Construct prompt: system instruction + hard prompt (verbalized graph) + user question
2. Call OpenRouter API with `temperature=0`, `seed=42`
3. Parse response into structured `CausalExplanation`
4. Extract identified causal edges and reasoning steps

### 4.3 Simplified Pipeline (No-GNN Baseline)

A text-only baseline for comparison:
1. Retrieve subgraph (same as above)
2. Verbalize **all** triples equally (no attention weighting)
3. Send to LLM

This baseline isolates the GNN's contribution to answer quality.

---

## 5. Configuration

All configuration via a Python dataclass with `__post_init__` validation.

```python
@dataclass
class BenchmarkConfig:
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Google Drive
    drive_root: str = "/content/drive/MyDrive/arcOS_benchmark"

    # Dataset
    dataset_name: str = "rmanluo/RoG-webqsp"
    max_train_examples: int = 600       # None = use all (2,830)
    max_val_examples: int = 50          # None = use all (246)
    max_test_examples: int = None       # Keep full test set

    # Graph
    graph_directed: bool = True
    unified_graph_min_nodes: int = 2000
    unified_graph_min_edges: int = 6000

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    top_k_entities: int = 15
    pcst_budget: int = 70
    pcst_local_budget: int = 500
    pcst_cost: float = 0.015
    pcst_pruning: str = "gw"
    pcst_edge_weight_alpha: float = 0.5
    pcst_bridge_components: bool = True
    pcst_bridge_max_hops: int = 6

    # GNN
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1
    gnn_pooling: str = "attention"      # "attention" | "mean"

    # Verbalization
    top_k_triples: int = 15
    verbalization_format: str = "natural"  # "natural" | "structured"

    # LLM
    llm_provider: str = "openrouter"
    llm_model: str = "google/gemini-2.5-flash-preview"
    llm_api_base: str = "https://openrouter.ai/api/v1"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 10
    patience: int = 5
    gradient_clip: float = 1.0

    # Evaluation
    metrics: list = ["exact_match", "f1", "hits@1"]
```

---

## 6. Evaluation

### 6.1 Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Proportion of predictions exactly matching a gold answer |
| **F1 Score** | Token-level F1 between prediction and best-matching gold answer |
| **Hit@1** | Whether the top prediction is in the gold answer set |
| **Graph Grounding Rate** | Proportion of answer entities found in the retrieved subgraph |

### 6.2 Evaluation Protocol

1. Run pipeline on test split (1,630 questions, or configurable subset)
2. Compare generated answers against `answer` field
3. Compute metrics; report mean and standard deviation across 3 seeded runs
4. Compare GNN-guided pipeline vs. no-GNN baseline

---

## 7. Colab Environment

### 7.1 Runtime Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | T4 (free tier sufficient) |
| RAM | ~13 GB system RAM |
| VRAM | ~4 GB peak (GNN training + embeddings) |
| Disk (Drive) | ~5 GB (checkpoints across all phases) |
| Network | Required (OpenRouter API, HuggingFace downloads) |

### 7.2 Package Management

All installations use **uv** (not pip):

```python
# Cell 1: Setup uv
import os
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""
!pip install uv

# Cell 1: Install dependencies
!uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu121
!uv pip install --system torch_geometric
!uv pip install --system pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
!uv pip install --system \
    datasets sentence-transformers faiss-gpu \
    openai tenacity pydantic networkx \
    pcst_fast scikit-learn tqdm
```

### 7.3 Persistence Strategy

- **Google Drive mount** for checkpoints and results
- **HuggingFace cache** redirected to Drive
- **Idempotent cells** — each cell checks if work is already done before re-executing
- **Checkpoint saving** after each pipeline stage:

| Phase | Checkpoints | Size |
|-------|------------|------|
| 1 | `dataset.pkl`, `unified_graph.pkl` | ~0.7 GB |
| 2 | `entity_embeddings.pkl`, `faiss_index.bin`, `relation_embeddings.pkl` | ~3 GB |
| 3 | `gnn_model.pt`, `pyg_train_data.pkl`, `pyg_val_data.pkl` | ~1.7 GB |
| 6 | `batch_results_*.jsonl` (incremental) | TBD |
| 7 | `metrics.json`, `comparison.csv` | <1 MB |

### 7.4 Secrets Management

```python
from google.colab import userdata
OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
```

---

## 8. Non-Goals (MVP Exclusions)

- No local LLM hosting (vLLM, Ollama, etc.)
- No Docker or container orchestration
- No Memgraph or external graph database
- No web UI or API server
- No W&B/Weave observability integration (future phase)
- No GNN soft-prompt injection into LLM embedding space
- No multi-GPU or distributed training
- No real-time / streaming inference

---

## 9. Success Criteria

| Criterion | Target |
|-----------|--------|
| Pipeline runs end-to-end in Colab free tier | Yes |
| GNN-guided pipeline outperforms no-GNN baseline on F1 | >5% improvement |
| Reproducible results across 3 seeded runs | Variance < 2% |
| Full test set evaluation completes within Colab session | < 4 hours |
| Code organized as importable modules (not just notebook cells) | Yes |

---

## 10. Key Dependencies

| Package | Purpose | Version Constraint |
|---------|---------|--------------------|
| `torch` | Deep learning framework | >=2.1 |
| `torch_geometric` | GNN framework (GATv2Conv, SAGEConv, GlobalAttention) | >=2.5 |
| `datasets` | HuggingFace dataset loading | latest |
| `sentence-transformers` | Text embeddings (all-MiniLM-L6-v2) | latest |
| `faiss-gpu` | k-NN search (IndexFlatIP) | latest |
| `networkx` | In-memory graph storage | >=3.0 |
| `pcst_fast` | PCST subgraph solver | latest |
| `openai` | OpenRouter API client | >=1.0 |
| `tenacity` | Retry logic | latest |
| `uv` | Package installation (Colab workaround) | latest |

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Colab session timeout during evaluation | Lost progress | Checkpoint every 50 examples; resume from last checkpoint |
| OpenRouter rate limits | Slow evaluation | Batch requests; configurable delay; model fallback chain |
| OpenRouter API cost | Budget overrun | Default to cost-effective model (Gemini Flash); `max_samples` config |
| T4 VRAM insufficient for large GNN | Training failure | Cap subgraph at 70 nodes; focal loss; gradient clipping |
| PyG version mismatch in Colab | Import errors | Dynamic version detection; pin compatible wheels |
| Dataset noise (CVT nodes, junk relations) | Bad embeddings | Noise filtering at graph construction time |
| GNN class imbalance (~2% positive rate) | Model predicts all-negative | Focal loss (gamma=2.0) focuses on hard positives |
| Disconnected PCST subgraphs | Incomplete context | Component bridging via shortest paths (max 6 hops) |
