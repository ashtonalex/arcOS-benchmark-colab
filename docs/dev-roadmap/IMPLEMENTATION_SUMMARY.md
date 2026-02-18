# arcOS Benchmark - Implementation Summary

**Covers:** Phases 1-3 (Environment & Data, Retrieval, GNN Encoder)
**Last Updated:** 2026-02-18
**Status:** Phases 1-3 complete, awaiting full Colab validation

---

## What Has Been Built

Three of eight phases are fully implemented. The system can load a dataset, build a knowledge graph, retrieve query-relevant subgraphs, and encode them with a GNN to produce attention-weighted node embeddings. What remains is verbalization (Phase 4), LLM integration (Phase 5), and the end-to-end wiring, evaluation, and polish (Phases 6-8).

---

## Phase 1: Environment & Data Foundation

### Modules

| File | Class/Function | Purpose | LOC |
|------|---------------|---------|-----|
| `src/config.py` | `BenchmarkConfig` | Central hyperparameter dataclass for all 8 phases | 154 |
| `src/utils/seeds.py` | `set_seeds()` | Deterministic random state (torch, numpy, random, CUDNN) | 39 |
| `src/utils/checkpoints.py` | `save_checkpoint()`, `load_checkpoint()`, etc. | Google Drive persistence (pickle/JSON/GraphML) | 144 |
| `src/data/dataset_loader.py` | `RoGWebQSPLoader` | HuggingFace dataset loading with caching and validation | 262 |
| `src/data/graph_builder.py` | `GraphBuilder` | NetworkX graph construction with noise filtering | 329 |

### Key Implementation Decisions

**Noise filtering at graph construction time.**
RoG-WebQSP contains significant noise from Freebase: opaque CVT (Compound Value Type) nodes like `m.0rqp4h0` and administrative relations like `freebase.valuenotation.*`. These produce meaningless embeddings and pollute PCST retrieval. `GraphBuilder` filters them at construction time:
- CVT node pattern: `^[mg]\.[0-9a-z_]+$`
- Junk relation prefixes: `freebase.valuenotation`, `freebase.type_profile`, `type.object`, `kg.object_profile`, `rdf-schema#`
- Effect: drops ~15-20% of triples, keeps only semantically meaningful relations

**Dataset slicing for development.**
`RoGWebQSPLoader.slice_dataset()` reduces the dataset to 600/50/1628 (train/val/test) for faster iteration. Config thresholds are adjusted accordingly (min 2K nodes, 6K edges).

### Notebook Cells
- Cell 1: Environment setup (uv + dependencies)
- Cell 2: Module imports
- Cell 3: Configuration initialization
- Cell 4: Seed setup
- Cell 5: Google Drive mount
- Cell 6: Dataset loading
- Cell 7: Unified graph construction
- Cell 8: Phase 1 validation (automated pass/fail)

---

## Phase 2: Retrieval Pipeline

### Modules

| File | Class | Purpose | LOC |
|------|-------|---------|-----|
| `src/retrieval/embeddings.py` | `TextEmbedder` | Sentence-Transformers encoding with relation enrichment | ~200 |
| `src/retrieval/faiss_index.py` | `EntityIndex` | FAISS IndexFlatIP for exact cosine similarity search | ~150 |
| `src/retrieval/pcst_solver.py` | `PCSTSolver` | Localize-then-optimize subgraph extraction | ~300 |
| `src/retrieval/retriever.py` | `Retriever`, `RetrievedSubgraph` | Orchestration with factory pattern | ~250 |

### Key Implementation Decisions

**Relation-enriched entity embeddings.**
Instead of embedding bare entity names (e.g., "Cleveland"), `TextEmbedder` enriches them with relation context (e.g., "Cleveland | containedby, time zone, adjoins"). This disambiguates entities that share names and improves k-NN retrieval quality.

**Localize-then-optimize PCST strategy.**
Running PCST on the full unified graph (10K+ nodes) is too slow. The solver first extracts a 500-node BFS neighborhood from seed entities, then runs PCST on that local subgraph. This keeps query time under 1 second.

**Component bridging.**
PCST can produce disconnected components when seed entities are graph-distant. The solver bridges disconnected components via shortest paths (max 6 hops), ensuring the output subgraph is always connected.

**Local prize computation.**
When k-NN seeds are in a different connected component than the root entity, the root's component would have no prized nodes. The solver computes local cosine similarity prizes for nodes in the root's component, ensuring it's included in the output.

### Checkpoint Budget
| Artifact | Size |
|----------|------|
| `entity_embeddings.pkl` | ~1.5 GB |
| `faiss_index.bin` | ~1.5 GB |
| `relation_embeddings.pkl` | ~8 MB |
| **Total** | **~3 GB** |

### Notebook Cells
- Cell 9: Build retriever (embeddings + FAISS index, or load from checkpoint)
- Cell 10: Validate retrieval on 10 examples
- Cell 11: Success criteria check

---

## Phase 3: GNN Encoder

### Modules

| File | Class | Purpose | LOC |
|------|-------|---------|-----|
| `src/gnn/data_utils.py` | `SubgraphConverter`, `GNNOutput` | NetworkX → PyG conversion | ~200 |
| `src/gnn/encoder.py` | `GATv2Encoder`, `GraphSAGEEncoder` | GNN architectures | ~230 |
| `src/gnn/pooling.py` | `AttentionPooling`, `MeanPooling`, `MaxPooling` | Graph-level aggregation | ~150 |
| `src/gnn/trainer.py` | `GNNTrainer`, `FocalLoss` | Training with focal loss and early stopping | ~280 |
| `src/gnn/model_wrapper.py` | `GNNModel` | High-level API with checkpoint management | ~180 |

### Key Implementation Decisions

**Focal loss for class imbalance.**
Each subgraph has ~50 nodes but only 1-3 answer nodes. Standard cross-entropy loss overwhelms the signal from rare positive examples. Focal loss (`gamma=2.0`) down-weights easy negatives and focuses learning on hard examples: `FL(p_t) = -(1 - p_t)^gamma * log(p_t)`.

**Query conditioning in GATv2.**
The encoder broadcasts the question embedding to all nodes before GATv2Conv layers. This lets the attention mechanism learn query-dependent node relevance, not just structural importance.

**Residual connections + LayerNorm.**
Each GATv2Conv layer has a residual connection and layer normalization, preventing gradient degradation in deeper networks (3 layers default).

**Factory pattern matching Phase 2.**
`GNNModel.build_from_checkpoint_or_train()` mirrors `Retriever.build_from_checkpoint_or_new()`, providing a consistent API across phases. Both check for existing checkpoints before doing expensive computation.

### GNN Architecture (GATv2Encoder)
```
Input: node_features [N, 384] + edge_features [E, 384] + query [384]

→ InputProjection (384 → 256)
→ QueryCondition (concatenate query → project back to 256)
→ GATv2Conv Layer 1 (4 heads, 256-dim) + Residual + LayerNorm + Dropout(0.1)
→ GATv2Conv Layer 2 (4 heads, 256-dim) + Residual + LayerNorm + Dropout(0.1)
→ GATv2Conv Layer 3 (4 heads, 256-dim) + Residual + LayerNorm + Dropout(0.1)
→ AttentionPooling (gate + feature networks)

Output: node_embeddings [N, 256], attention_scores {node: float}, graph_embedding [256]
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Loss | Focal loss (gamma=2.0) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early stopping | patience=5 epochs |
| Gradient clipping | max_norm=1.0 |
| Batch size | 16 |
| Epochs | 10 (with early stopping) |

### Checkpoint Budget
| Artifact | Size |
|----------|------|
| `pyg_train_data.pkl` | ~1.5 GB |
| `pyg_val_data.pkl` | ~150 MB |
| `gnn_model.pt` | ~50 MB |
| **Total** | **~1.7 GB** |

### Notebook Cells
- Cell 12: Build/train GNN model (or load from checkpoint)
- Cell 13: Test inference on example subgraph
- Cell 14: Validate metrics (F1 > 0.5)
- Cell 15: Visualize attention weights
- Cell 16: GPU memory check

---

## Architecture Decisions (Cross-Phase)

### 1. Hard Prompts vs. Soft Prompts
**Decision:** Text-based hard prompts (verbalized triples).
- Works with any LLM API (OpenRouter, Anthropic, OpenAI)
- No need to modify LLM internals or embedding spaces
- Easier to debug, interpret, and iterate on
- Trade-off: Less expressive than continuous soft prompts

### 2. NetworkX vs. External Graph DB
**Decision:** In-memory NetworkX.
- Colab-native, no external services
- Sufficient for 4,706 examples (~15K nodes, ~40K edges after filtering)
- Easy checkpoint to Google Drive (pickle)
- Trade-off: won't scale to millions of nodes

### 3. FAISS IndexFlatIP vs. Approximate Methods
**Decision:** Exact inner product search.
- Guarantees correct nearest neighbors
- Acceptable latency for ~10K entities (~200ms per query)
- Trade-off: slower than IVF/HNSW for larger indices

### 4. Focal Loss vs. Standard Cross-Entropy
**Decision:** Focal loss with gamma=2.0.
- Severe class imbalance: ~2% positive nodes per subgraph
- Standard CE converges to "always predict negative"
- Focal loss forces model to learn from hard positive examples

### 5. Noise Filtering at Source
**Decision:** Filter CVT nodes and junk relations during graph construction.
- Prevents garbage from propagating through embeddings, FAISS, PCST, and GNN
- Reduces graph size by ~15-20%, saving memory and compute
- No downstream modules need their own filtering logic

---

## Performance Characteristics

### Timing

| Phase | Operation | Cold Start | Warm Start |
|-------|-----------|------------|------------|
| 1 | Environment + data | ~5 min | ~2.5 min |
| 2 | Embeddings + FAISS build | ~5-10 min | ~10 sec |
| 2 | Single query retrieval | - | 100-500 ms |
| 3 | PyG data preparation | ~15 min | ~10 sec |
| 3 | GNN training (10 epochs) | ~15-20 min | skipped |
| 3 | Single inference | - | <100 ms |
| **Total cold start** | **(Phases 1-3)** | **~40-50 min** | - |
| **Total warm start** | **(Phases 1-3)** | - | **~3 min** |

### Memory

| Resource | Usage |
|----------|-------|
| Google Drive (checkpoints) | ~4.7 GB |
| System RAM (runtime) | ~2-3 GB |
| GPU VRAM (peak training) | ~800 MB |
| GPU VRAM (inference) | ~200 MB |

---

## Code Metrics

### Lines of Code by Module

| Module | LOC |
|--------|-----|
| `src/config.py` | 154 |
| `src/utils/` | 183 |
| `src/data/` | 591 |
| `src/retrieval/` | ~900 |
| `src/gnn/` | ~1,040 |
| **Total source** | **~2,870** |
| Scripts + tests | ~400 |
| **Total** | **~3,270** |

### Quality
- All public classes and functions have docstrings
- Type hints used throughout
- Consistent patterns across phases (factory methods, checkpoint-or-build, dataclass outputs)
- Local validation scripts for import testing without GPU

---

## Alignment with PRD

| PRD Section | Status | Notes |
|-------------|--------|-------|
| 1. Overview (GNN + LLM pipeline) | Partial | GNN complete, LLM pending (Phase 5) |
| 2. Goals (reproducibility, Colab-native) | Met | Seeds, checkpoints, idempotent cells |
| 3. Dataset (RoG-WebQSP) | Met | 4,706 examples, schema validated |
| 4. Architecture (4 layers) | 2/4 | Retrieval + GNN done; Verbalization + LLM pending |
| 5. Configuration | Met | `BenchmarkConfig` covers all phases |
| 6. Evaluation | Pending | Phase 7 |
| 7. Colab Environment | Met | uv setup, Drive persistence, GPU validation |
| 8. Non-Goals | Respected | No Docker, no local LLM, no Memgraph |
| 9. Success Criteria | Pending | Requires full pipeline (Phase 6+) |

---

## What's Next

### Immediate (can start in parallel)
- **Phase 4:** Graph Verbalization — convert GNN attention scores to ranked triple text
- **Phase 5:** LLM Integration — OpenRouter client with retry/fallback

### After Phases 4+5
- **Phase 6:** Wire everything into `BenchmarkPipeline`
- **Phase 7:** Evaluate with EM, F1, Hit@1 metrics
- **Phase 8:** Error handling, logging, polish

### Critical Path
```
[Phase 4] ──┐
            ├──> Phase 6 → Phase 7 → Phase 8
[Phase 5] ──┘
```
Phase 4 is likely the bottleneck since it requires careful attention-to-text conversion and relation cleaning. Phase 5 is more mechanical (API client + retry logic).
