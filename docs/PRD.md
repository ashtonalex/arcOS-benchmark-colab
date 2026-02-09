# arcOS Benchmark - MVP Product Requirements Document

**Version:** 2.0 (Colab Edition)
**Date:** 2026-02-09
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
- Nodes are entity strings (e.g., `"Justin Bieber"`)
- Edges are Freebase relation strings (e.g., `"people.person.sibling_s"`)
- Subgraphs contain multi-hop paths (up to 4 hops from topic entity)
- Derived from the Freebase knowledge graph (~88M entities, 20K relations)

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

#### Layer 1: Graph Storage & Retrieval

| Aspect      | Detail |
|-------------|--------|
| **Graph DB**    | NetworkX (in-memory, replacing Memgraph) |
| **Embedding**   | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Index**       | FAISS flat index for k-NN semantic search |
| **Subgraph extraction** | PCST via `pcst_fast` library, or BFS fallback |
| **Max subgraph size** | 20 nodes (configurable) |

**Process:**
1. Load RoG-webqsp dataset from HuggingFace
2. Build NetworkX graph from triple lists
3. Encode all node/edge text with Sentence-Transformers into embeddings
4. Store embeddings in FAISS index
5. At query time: embed query -> k-NN search -> PCST subgraph extraction

#### Layer 2: GNN Encoder

| Aspect      | Detail |
|-------------|--------|
| **Architecture** | GATv2 (primary) or GraphSAGE (alternative) |
| **Layers**       | 2-4 layers, configurable |
| **Hidden dim**   | 256 (tunable) |
| **Pooling**      | Attention pooling (`GlobalAttention`) |
| **Framework**    | PyTorch Geometric |
| **Training**     | Supervised on train split; loss = answer generation quality |

**Process:**
1. Convert retrieved subgraph to PyG `Data` object (node features from Sentence-Transformer embeddings)
2. Pass through GATv2 layers to produce contextual node embeddings
3. Attention pooling scores each node's relevance to the query
4. Attention weights guide verbalization (next layer)

#### Layer 3: Attention-Guided Graph Verbalization (Hard Prompt)

| Aspect      | Detail |
|-------------|--------|
| **Input**    | Subgraph + GNN attention weights |
| **Output**   | Natural language description of the most relevant subgraph portions |
| **Format**   | Ranked triples: `(subject) --[relation]--> (object)` |
| **Max tokens** | ~500 tokens of verbalized context |

**Process:**
1. Rank nodes/edges by GNN attention scores
2. Select top-K triples (those involving highest-attention nodes)
3. Verbalize selected triples into structured text
4. Format as hard prompt:
   ```
   Relevant knowledge graph context:
   1. (Justin Bieber) --[sibling]--> (Jaxon Bieber)
   2. (Justin Bieber) --[born_in]--> (London, Ontario)
   ...

   Question: What is the name of Justin Bieber's brother?
   ```

#### Layer 4: LLM Interpretation (via OpenRouter)

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

All configuration via Python dataclasses with Pydantic validation.

```python
@dataclass
class BenchmarkConfig:
    # Dataset
    dataset_name: str = "rmanluo/RoG-webqsp"
    dataset_split: str = "test"
    max_samples: int = 100          # for quick runs

    # Retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_nodes: int = 10
    max_subgraph_nodes: int = 20
    pcst_cost: float = 1.0

    # GNN
    gnn_type: str = "gatv2"         # "gatv2" | "graphsage"
    gnn_layers: int = 2
    gnn_hidden_dim: int = 256
    gnn_heads: int = 4              # for GATv2
    gnn_dropout: float = 0.1

    # LLM
    openrouter_model: str = "google/gemini-2.5-flash-preview"
    openrouter_fallback: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 512

    # Reproducibility
    seed: int = 42
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
| VRAM | ~4 GB (GNN + embeddings) |
| Disk | ~2 GB (dataset + models) |
| Network | Required (OpenRouter API, HuggingFace downloads) |

### 7.2 Package Management

All installations use **uv** (not pip):

```python
# Cell 1: Setup uv
import os
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""
!pip install uv

# Cell 2: Install dependencies
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
- **HuggingFace cache** redirected to Drive: `os.environ["HF_HOME"] = "/content/drive/MyDrive/arcOS/hf_cache"`
- **Idempotent cells** - each cell checks if work is already done before re-executing
- **Checkpoint saving** after each pipeline stage (graph built, embeddings computed, GNN trained, results generated)

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
| `torch_geometric` | GNN framework | >=2.5 |
| `datasets` | HuggingFace dataset loading | latest |
| `sentence-transformers` | Text embeddings | latest |
| `faiss-gpu` | k-NN search | latest |
| `networkx` | In-memory graph storage | >=3.0 |
| `pcst_fast` | PCST subgraph solver | latest |
| `openai` | OpenRouter API client | >=1.0 |
| `tenacity` | Retry logic | latest |
| `pydantic` | Config validation | >=2.0 |
| `uv` | Package installation | latest |

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Colab session timeout during evaluation | Lost progress | Checkpoint after each batch; resume from last checkpoint |
| OpenRouter rate limits | Slow evaluation | Batch requests; configurable delay; model fallback chain |
| OpenRouter API cost | Budget overrun | Default to cost-effective model; `max_samples` config for quick runs |
| T4 VRAM insufficient for large GNN | Training failure | Cap subgraph at 20 nodes; use 2-layer GNN; monitor with `nvidia-smi` |
| PyG version mismatch in Colab | Import errors | Dynamic version detection; pin compatible wheels |
| Dataset too large for RAM | OOM | Stream dataset; process one example at a time |
