# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

arcOS Benchmark is a causal question-answering system combining Graph Neural Networks (GNN) with Large Language Models (LLM) for grounded reasoning. It runs entirely in Google Colab and implements a simplified G-Retriever-inspired pipeline where GNNs learn graph structure and guide verbalization of knowledge graph subgraphs into text (hard prompts) that are sent to LLMs via OpenRouter API.

**Key Architecture Decision:** Instead of soft-prompt embeddings injected into LLMs, this system uses GNN attention to select and verbalize relevant subgraph triples into discrete text appended to queries. This makes it API-compatible and model-agnostic.

## Critical Environment Setup

This project uses **uv** for package management in Google Colab. The uv installation requires clearing Colab's broken constraint environment variables:

```python
import os
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""
!pip install uv
!uv pip install --system <packages>
```

Always clear these environment variables before installing uv or any dependencies.

## Development Workflow

### Running the Benchmark in Colab

1. Upload `src/` folder to `/content/arcOS-benchmark-colab/src/`
2. Open `notebooks/arcOS_benchmark.ipynb` in Colab
3. Set runtime to GPU (T4 or better)
4. Run cells 1-8 sequentially for Phase 1
5. Verify Cell 8 shows "PHASE 1 COMPLETE"

**Timing:**
- Cold start (first run): ~4-5 minutes
- Warm start (with checkpoints): ~1-2 minutes
- Re-run (runtime active): ~15 seconds

### Local Testing (No GPU Required)

```bash
python test_phase1_imports.py
```

This validates all module imports and basic functionality without requiring Colab or GPU.

## Architecture

### Pipeline Overview

```
Query → Retrieval → GNN Encoding → Attention-Guided Verbalization → LLM Generation
           |                              |
      NetworkX graph                  Hard prompt (text)
      + k-NN search                   appended to query
```

### Key Components

**Layer 1: Graph Storage & Retrieval**
- Graph DB: NetworkX (in-memory, not Memgraph)
- Embeddings: Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim)
- Index: FAISS for k-NN semantic search
- Subgraph extraction: PCST via `pcst_fast` library

**Layer 2: GNN Encoder**
- Architecture: GATv2 (primary) or GraphSAGE (alternative)
- Framework: PyTorch Geometric
- Pooling: Attention pooling (`GlobalAttention`)
- Output: Node embeddings + attention weights

**Layer 3: Graph Verbalization**
- GNN attention scores guide which triples to verbalize
- Format: Ranked triples as natural language text
- Max budget: ~500 tokens
- Baseline: Unweighted verbalization (no GNN) for comparison

**Layer 4: LLM Integration**
- API: OpenRouter (`https://openrouter.ai/api/v1`)
- Default model: `google/gemini-2.5-flash-preview`
- Temperature: 0.0 (deterministic)
- Seed: 42 (for reproducibility)

### Data Flow

1. **Dataset:** HuggingFace `rmanluo/RoG-webqsp` (4,706 examples)
   - Train: 2,830 | Validation: 246 | Test: 1,630
   - Each example: question, answer, graph triples (subject-relation-object)

2. **Graph Construction:**
   - Unified graph: All training triples merged (~15K-25K nodes, ~40K-80K edges)
   - Per-example graphs: Individual subgraphs for each question

3. **Retrieval:** Query → k-NN search → PCST subgraph (max 20 nodes)

4. **GNN Processing:** Subgraph → GATv2 layers → Attention scores

5. **Verbalization:** Top-K attention-weighted triples → Natural language text

6. **LLM Generation:** Hard prompt + question → Answer

## Configuration

All hyperparameters are centralized in `src/config.py` using the `BenchmarkConfig` dataclass:

```python
from src.config import BenchmarkConfig

config = BenchmarkConfig(
    seed=42,
    deterministic=True,
    dataset_name="rmanluo/RoG-webqsp",
    gnn_hidden_dim=256,
    gnn_num_layers=3,
    llm_model="anthropic/claude-3.5-sonnet",
    # ... see config.py for all options
)
```

**Key Paths:**
- Checkpoints: `/content/drive/MyDrive/arcOS_benchmark/checkpoints`
- Results: `/content/drive/MyDrive/arcOS_benchmark/results`

## Dataset Schema

```python
{
    "id": str,                    # Unique identifier (e.g., "WebQTest-1234")
    "question": str,              # Natural language question
    "answer": List[str],          # List of answer entities
    "q_entity": str,              # Question topic entity (Freebase ID)
    "a_entity": str,              # Answer entity (Freebase ID)
    "graph": List[List[str]],     # Triples: [subject, relation, object]
}
```

**Triple Format:** `[subject, relation, object]` where:
- Subject/Object: Freebase entity IDs (e.g., `"m.02mjmr"`)
- Relation: Freebase dot-notation (e.g., `"people.person.sibling_s"`)

## Module Structure

```
src/
├── config.py              # Central configuration (BenchmarkConfig)
├── utils/
│   ├── seeds.py          # Determinism (set_seeds function)
│   └── checkpoints.py    # Google Drive persistence
├── data/
│   ├── dataset_loader.py # RoGWebQSPLoader class (Freebase KG)
│   ├── graph_builder.py  # GraphBuilder class (NetworkX, Freebase)
│   ├── agqa_loader.py    # AGQALoader class (video QA, download, split)
│   ├── ag_converter.py   # AG annotation converter (pivot relations to pairwise tuples)
│   └── scene_graph_builder.py # SceneGraphBuilder (Action Genome → PyG HeteroData)
├── retrieval/
│   ├── embeddings.py     # TextEmbedder (sentence-transformers)
│   ├── faiss_index.py    # EntityIndex (FAISS, Freebase pipeline)
│   ├── pcst_solver.py    # PCSTSolver (Freebase pipeline)
│   ├── retriever.py      # Retriever orchestration (Freebase pipeline)
│   ├── video_index.py    # VideoIndex (per-video FAISS for scene graphs)
│   ├── hetero_pcst.py    # HeteroPCST (PCST for PyG HeteroData)
│   └── video_retriever.py # VideoRetriever orchestration (scene graph pipeline)
├── gnn/
│   ├── encoder.py        # GATv2Encoder / GraphSAGEEncoder (homogeneous)
│   ├── hetero_encoder.py # HeteroGATv2Encoder (HeteroConv, scene graphs)
│   ├── hetero_trainer.py # HeteroGNNTrainer (training loop for HeteroGATv2)
│   ├── hetero_model_wrapper.py # HeteroGNNModel high-level API
│   ├── pooling.py        # Attention pooling
│   ├── trainer.py        # GNN training loop
│   ├── model_wrapper.py  # GNNModel high-level API
│   └── data_utils.py     # PyG data conversion utilities
├── verbalization/
│   └── scene_verbalizer.py # SceneVerbalizer (attention-weighted triple formatting)
└── evaluation/
    └── benchmark.py      # BenchmarkEvaluator (EM, F1, hit rate, attention precision)
```

### Key Classes

**`BenchmarkConfig`** (`src/config.py`)
- Manages all hyperparameters with validation
- Properties: `checkpoint_dir`, `results_dir`
- Methods: `get_checkpoint_path()`, `print_summary()`
- Video scene graph fields: `agqa_subset_size`, `ag_frame_sample_rate`, `top_k_seeds`, `pcst_temporal_cost_weight`, `gnn_encoder_type`

**`RoGWebQSPLoader`** (`src/data/dataset_loader.py`)
- `load(dataset_name, split)` → DatasetDict or Dataset
- `inspect_schema(dataset)` → Print schema and examples
- `validate_split_counts(dataset)` → Verify expected sizes

**`AGQALoader`** (`src/data/agqa_loader.py`)
- `parse_sample(raw)` → Standardized QA dict
- `get_unique_video_ids(samples)` → Set of video IDs
- `subsample(samples)` → Randomly subsample to configured size
- `download_agqa(target_dir)` → Download AGQA 2.0 balanced JSON files
- `load_from_file(filepath)` → Load AGQA JSON, parse all entries
- `split(samples, train_ratio, val_ratio)` → Deterministic split by video_id (no leakage)

**`AGConverter`** (`src/data/ag_converter.py`)
- `load_ag_annotations(pkl_path)` → Load Action Genome pickle
- `convert_video(video_id, raw_annotations)` → Pivot AG relations to SceneGraphBuilder format
- `convert_all(raw_annotations, video_ids)` → Batch convert multiple videos

**`SceneGraphBuilder`** (`src/data/scene_graph_builder.py`)
- `build(ag_annotations)` → PyG HeteroData with object nodes, spatial/temporal edges
- Stores `data.object_names` and `data.spatial_predicates` for SceneVerbalizer
- Supports optional embedder for class name encoding (falls back to random)

**`GraphBuilder`** (`src/data/graph_builder.py`)
- `build_from_triples(triples)` → NetworkX graph for single example
- `build_unified_graph(dataset)` → Merged graph from all examples
- `compute_graph_statistics(G)` → Comprehensive metrics
- `validate_graph_size(G, min_nodes, min_edges)` → Size validation

**`VideoRetriever`** (`src/retrieval/video_retriever.py`)
- `retrieve(question, scene_graph)` → RetrievalResult (subgraph + metadata)
- Orchestrates: embed query → per-video FAISS k-NN → PCST subgraph extraction

**`HeteroGATv2Encoder`** (`src/gnn/hetero_encoder.py`)
- Forward: HeteroData + query_embedding → (node_embeddings, attention_scores, graph_embedding)
- Uses HeteroConv with per-edge-type GATv2Conv (spatial + temporal)
- Query-conditioned input projection, residual + LayerNorm

**`HeteroGNNTrainer`** (`src/gnn/hetero_trainer.py`)
- Training loop for HeteroGATv2Encoder with focal loss
- `train(train_data, val_data)` → Training history
- `save_checkpoint(path)` / `load_checkpoint(path)`
- Early stopping with patience, LR scheduling

**`HeteroGNNModel`** (`src/gnn/hetero_model_wrapper.py`)
- `build_from_checkpoint_or_train(config, retriever, train_samples, val_samples, scene_graphs)` → Factory
- `encode(subgraph, question)` → (node_embeddings, attention_scores, graph_embedding)
- `set_embedder(embedder)` → Set text embedder for inference

**`SceneVerbalizer`** (`src/verbalization/scene_verbalizer.py`)
- `verbalize(data, attention_scores)` → Attention-ranked triple text
- `verbalize_unweighted(data)` → Baseline verbalization (no GNN)

**`BenchmarkEvaluator`** (`src/evaluation/benchmark.py`)
- `exact_match`, `f1`, `retrieval_hit_rate`, `attention_precision`
- `aggregate(results)` → Mean metrics across examples

### Utility Functions

**`set_seeds(seed, deterministic)`** (`src/utils/seeds.py`)
- Sets random, numpy, torch, PYTHONHASHSEED
- Enables CUDNN deterministic mode for GPU reproducibility

**Checkpoint Functions** (`src/utils/checkpoints.py`)
- `save_checkpoint(obj, filepath, format)` → Serialize to Drive
- `load_checkpoint(filepath, format)` → Deserialize from Drive
- `checkpoint_exists(filepath)` → Check if checkpoint exists
- Supports pickle, JSON, GraphML formats

## Determinism and Reproducibility

All random operations are seeded for bit-exact reproducibility:

```python
from src.utils.seeds import set_seeds

set_seeds(seed=42, deterministic=True)
# Sets: random, numpy, torch, PYTHONHASHSEED, CUDNN
```

When `deterministic=True`:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- GPU operations become deterministic but slower

## Checkpoint Strategy

Google Drive persistence prevents data loss on Colab disconnects:

```python
from src.utils.checkpoints import save_checkpoint, load_checkpoint

# Save
save_checkpoint(dataset, config.get_checkpoint_path("dataset.pkl"), format="pickle")

# Load (returns None if not exists)
dataset = load_checkpoint(config.get_checkpoint_path("dataset.pkl"), format="pickle")
```

**Idempotency Pattern:**
```python
checkpoint_path = config.get_checkpoint_path("unified_graph.pkl")
if checkpoint_exists(checkpoint_path):
    G = load_checkpoint(checkpoint_path)
else:
    G = builder.build_unified_graph(dataset)
    save_checkpoint(G, checkpoint_path)
```

## Implementation Phases

The project follows an 8-phase roadmap (see `docs/ROADMAP.md`):

1. **Environment & Data** ✓ COMPLETE
   - Colab setup, dataset loading, graph construction

2. **Retrieval Pipeline** ✓ COMPLETE
   - Sentence-Transformers embeddings, FAISS index, PCST extraction

3. **GNN Encoder** ✓ COMPLETE
   - GATv2/GraphSAGE implementation, attention pooling, training

4. **Graph Verbalization**
   - Attention-guided triple selection, natural language formatting

5. **LLM Integration**
   - OpenRouter client, retry logic, response parsing

6. **End-to-End Pipeline**
   - Component orchestration, batch processing, checkpointing

7. **Evaluation & Benchmarking**
   - Metrics (EM, F1, Hit@1), baseline comparison, reproducibility

8. **Hardening & Polish**
   - Error handling, logging, documentation

### Video Scene Graph Pipeline (feat/video-scene-graph)

Parallel implementation replacing Freebase KG with video scene graphs:

- [x] Config: Video scene graph fields in BenchmarkConfig (+ ag_annotations_dir, agqa_data_dir)
- [x] AGQA Loader: Parse, subsample, download, load_from_file, split by video_id
- [x] AG Converter: Pivot AG annotations to SceneGraphBuilder format
- [x] Scene Graph Builder: Action Genome → PyG HeteroData (+ object_names, spatial_predicates)
- [x] Video Index: Per-video FAISS for k-NN seed selection
- [x] HeteroPCST: PCST adapter for HeteroData subgraph extraction
- [x] Video Retriever: Orchestrates embed → k-NN → PCST pipeline
- [x] HeteroGATv2 Encoder: HeteroConv with per-edge-type GATv2Conv
- [x] HeteroGNN Trainer: Training loop with focal loss, early stopping
- [x] HeteroGNN Model Wrapper: High-level API with checkpoint factory
- [x] Scene Verbalizer: Attention-weighted triple formatting for LLM prompts
- [x] Benchmark Evaluator: EM, F1, retrieval hit rate, attention precision
- [x] Integration test: End-to-end smoke test with mock data
- [x] Notebook: Rewritten for video scene graph pipeline (36 cells, Phases 0-5)

**Current Status:** Video scene graph pipeline complete, notebook rewritten, awaiting Colab validation

## Validation and Testing

### Phase 1 Success Criteria

All validated in Cell 8 of the notebook:

- [x] GPU available (`torch.cuda.is_available()`)
- [x] All imports successful
- [x] Dataset splits valid (2830/246/1630)
- [x] Unified graph size valid (>10K nodes, >30K edges)
- [x] Checkpoint round-trip works

### Local Validation

```bash
python test_phase1_imports.py
```

Verifies imports, configuration validation, and basic graph construction without Colab.

## Common Patterns

### Loading Dataset with Caching

```python
from src.data.dataset_loader import RoGWebQSPLoader
from src.config import BenchmarkConfig

config = BenchmarkConfig()
loader = RoGWebQSPLoader(cache_dir=config.checkpoint_dir / "huggingface_cache")
dataset = loader.load("rmanluo/RoG-webqsp")
loader.validate_split_counts(dataset)
```

### Building Unified Graph

```python
from src.data.graph_builder import GraphBuilder

builder = GraphBuilder(directed=True)
G = builder.build_unified_graph(dataset["train"])
builder.validate_graph_size(G, min_nodes=10000, min_edges=30000)
```

### Configuration Management

```python
from src.config import BenchmarkConfig

config = BenchmarkConfig(
    seed=123,
    gnn_num_layers=4,
    llm_temperature=0.1
)
config.print_summary()

# Access paths
checkpoint_path = config.get_checkpoint_path("my_model.pt")
results_path = config.get_results_path("metrics.json")
```

## Known Constraints

- **Graph Database:** NetworkX only (no Memgraph or external DB)
- **LLM Access:** OpenRouter API only (no local vLLM/Ollama)
- **Package Manager:** uv required (with environment variable workaround)
- **Platform:** Google Colab with T4 GPU (15GB VRAM)
- **Dataset:** Fixed at 4,706 examples (not scalable to millions)

## Important Notes

1. **Always clear `UV_CONSTRAINT` variables** before installing dependencies in Colab
2. **Mount Google Drive** before running any checkpoint operations
3. **Set runtime to GPU** for PyTorch/PyG operations
4. **Use checkpointing** to prevent data loss on session timeout
5. **Validate Phase 1** before proceeding to Phase 2

## Documentation

- `docs/PRD.md` - Product requirements and architecture
- `docs/ROADMAP.md` - 8-phase implementation plan
- `docs/QUICKSTART.md` - Step-by-step Colab setup
- `docs/PHASE1_COMPLETE.md` - Phase 1 implementation details
- `docs/IMPLEMENTATION_SUMMARY.md` - Technical summary and decisions
