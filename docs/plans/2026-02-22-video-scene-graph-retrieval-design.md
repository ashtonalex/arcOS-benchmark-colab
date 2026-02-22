# Video Scene Graph Retrieval Design

## Overview

Refactor the G-retrieval layer from static Freebase knowledge graph QA (RoG-WebQSP) to video scene graph QA using AGQA + Action Genome. The goal is benchmarking GNN-augmented LLM inference vs pure LLM inference on structured video understanding tasks.

## Motivation

The current PCST implementation constantly falls back to BFS due to noisy node embeddings and k-NN selection from the Freebase-based unified graph (~1M nodes, CVT noise, disconnected components). Switching to Action Genome scene graphs eliminates these issues: graphs are per-video (1K-5K nodes), cleanly labeled (36 object classes, 26 relations), and temporally connected.

## Architecture

```
AGQA QA pair (question, answer, video_id, program)
    ↓
Action Genome scene graph (per video_id)
    ↓
PyG HeteroData (object nodes, spatial + temporal edges)
    ↓
Query embedding (sentence-transformer, 384-dim)
    ↓
k-NN seed selection (per-video FAISS index)
    ↓
PCST subgraph extraction (pcst_fast on flattened edge arrays)
    ↓
HeteroGATv2 encoding (HeteroConv, per-edge-type attention)
    ↓
Attention-weighted verbalization (ranked triples, 500 token budget)
    ↓
LLM generation (OpenRouter, hard prompt)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data source | AGQA + Action Genome | Pre-annotated scene graphs + 192M QA pairs. No custom video processing. Ground truth alignment guaranteed via `program` field. |
| Graph storage | PyG HeteroData (no NetworkX) | Native heterogeneous support. PCST only needs numpy arrays, extractable from edge_index. Eliminates conversion overhead. |
| Temporal structure | Temporal edges | Same object across frames connected by identity edges. Preserves motion/state changes. GNN learns temporal patterns. |
| GNN architecture | HeteroConv + GATv2Conv | Explicit per-edge-type convolutions. Direct attention weight access for verbalization. Per-type architecture tuning. |
| Auto-conversion | HeteroConv over to_hetero() | to_hetero() makes attention extraction opaque. HeteroConv returns attention weights directly per edge type. |
| Batch size | 128 | Subgraphs are small (≤50 nodes). Total GPU memory ~70-80MB per batch. Well within T4's 15GB. |
| Compute target | Google Colab T4 (15GB VRAM) | Same as current project. Sufficient for scene-graph-scale GNN work. |

## Layer 1: Data

### Sources

- **Action Genome**: 10K Charades videos, frame-level scene graph annotations. 36 object classes, 26 relationship types, ~1.7M object instances.
- **AGQA**: 192M balanced QA pairs derived from Action Genome scene graphs. Each pair includes `video_id`, `question`, `answer`, and `program` (functional derivation from graph nodes).

### Subsampling

- Use AGQA balanced split (1.7M QA pairs)
- Subsample to 50K QA pairs (~3K unique videos)
- Checkpoint converted HeteroData objects to Google Drive

### PyG HeteroData Schema

```
Node types:
    "object":
        x: [num_objects, 384]  # sentence-transformer embedding of object class
        frame_id: [num_objects]  # which frame this instance appears in
        object_class: [num_objects]  # integer class label

Edge types:
    ("object", "spatial_rel", "object"):
        edge_index: [2, num_spatial_edges]
        edge_attr: [num_spatial_edges, 384]  # relation type embedding
        relation_type: [num_spatial_edges]  # integer relation label

    ("object", "temporal", "object"):
        edge_index: [2, num_temporal_edges]
        edge_attr: [num_temporal_edges, 2]  # [frame_delta, bbox_iou]
```

### Graph Scale (per video, ~30s clip)

- ~100-300 frame snapshots (sampled at 1-3 fps)
- ~5-15 objects per frame → 500-4,500 object nodes
- ~10-30 spatial edges + ~5-15 temporal edges per frame
- Total per video: ~1K-5K nodes, ~3K-12K edges

## Layer 2: Retrieval

### Pipeline

1. **Query embedding**: Sentence-transformer (all-MiniLM-L6-v2, 384-dim)
2. **k-NN seed selection**: FAISS IndexFlatIP per-video index. Top-10 object nodes by cosine similarity. Per-video index is small (1K-5K nodes) so exact search is fast.
3. **PCST subgraph extraction**:
   - Prizes: cosine similarity scores from k-NN
   - Edge costs: uniform for spatial, distance-weighted for temporal
   - Budget: 50 nodes max
   - Input: flatten HeteroData edge types into single edge array with type mapping
   - Output: node mask applied back to HeteroData, preserving types
4. **Output**: HeteroData subgraph (≤50 nodes, connected)

### PCST Adapter (HeteroData to pcst_fast)

```
HeteroData.edge_index (per type) → concatenate into flat edge array
    → track (edge_idx → original edge type) mapping
    → run pcst_fast(edges, prizes, costs)
    → selected node indices → mask HeteroData
    → return subgraph HeteroData with types preserved
```

### Why PCST Works Better Here

- Per-video graphs are 100-1000x smaller than current unified graph (no localization step needed)
- No CVT/MID noise — Action Genome uses clean object class labels
- Temporal edges provide natural connectivity (objects persist across frames)
- Disconnected component problem eliminated by temporal links
- BFS fallback retained but expected to trigger rarely

## Layer 3: GNN Encoder

### Architecture: HeteroGATv2

```
HeteroData subgraph (≤50 nodes)
    ↓
[1] Per-type input projection
    - object: Linear(384 → 256)
    ↓
[2] Query conditioning
    - query_embedding: Linear(384 → 256)
    - Broadcast + residual add to all node features
    ↓
[3] HeteroConv layers (×3)
    - GATv2Conv for ("object", "spatial_rel", "object"): 4 heads
    - GATv2Conv for ("object", "temporal", "object"): 4 heads
    - 256-dim hidden, dropout 0.1
    - Attention weights extracted per edge type
    ↓
[4] Node attention aggregation
    - Average incoming attention weights per node (across edge types, layers)
    - Normalize to [0, 1]
    ↓
[5] Output:
    - node_embeddings: (num_nodes, 256)
    - attention_scores: (num_nodes,) per-node importance
    - graph_embedding: attention-pooled (256,)
```

### Training

- **Task**: Answer node prediction. Nodes participating in AGQA `program` labeled positive (1), others negative (0).
- **Loss**: Focal loss (gamma=2.0) for class imbalance
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Early stopping**: patience=5 on validation F1
- **Batch size**: 128
- **Estimated training time**: 10-20 min on T4 (50K examples, small subgraphs)
- **Model params**: ~2M

### Memory Budget (T4, 15GB VRAM)

| Component | Memory |
|-----------|--------|
| Batch data (128 subgraphs) | ~40MB |
| Model params | ~8MB |
| Optimizer state (AdamW) | ~16MB |
| Gradients | ~8MB |
| Sentence-transformer (if co-loaded) | ~500MB |
| **Total** | **~570MB (<4% of T4)** |

## Layer 4: Verbalization

1. Rank nodes by GNN attention score (descending)
2. For each high-attention node, collect its edges:
   - Spatial: "person holding cup (frame 42)"
   - Temporal: "person appears in frames 38-45"
3. Format as ranked triple list with attention scores
4. Truncate to ~500 token budget

## Layer 5: LLM Integration

- API: OpenRouter (same as current)
- Default model: configurable via BenchmarkConfig
- Temperature: 0.0, seed: 42

## Benchmark Evaluation

### Three-Way Comparison

| Condition | Graph input | GNN | What it tests |
|-----------|------------|-----|---------------|
| Pure LLM | None (raw frame descriptions) | No | LLM baseline capability |
| Graph-verbalized | PCST subgraph triples (unweighted) | No | Value of graph structure |
| GNN-augmented | Attention-weighted triples | Yes | Value of GNN attention |

### Metrics

| Metric | What it measures |
|--------|-----------------|
| Exact Match (EM) | Answer string equality |
| F1 | Token-level overlap |
| Hit@1 | Top answer is correct |
| Retrieval Hit Rate | PCST subgraph contains answer-relevant nodes (from `program`) |
| Attention Precision | High-attention nodes overlap with ground truth program nodes |

### Ground Truth

- **Video selection**: Deterministic — each AGQA QA pair specifies `video_id`
- **Answer correctness**: Exact match against AGQA `answer` field
- **Retrieval correctness**: AGQA `program` field identifies which scene graph nodes/edges participate in the answer derivation
- **No ambiguity**: All mappings are explicit in the dataset

## What Gets Replaced

| Current (RoG-WebQSP) | New (AGQA + Action Genome) |
|----------------------|---------------------------|
| `src/data/dataset_loader.py` | New AGQA/Action Genome loader |
| `src/data/graph_builder.py` (NetworkX) | New PyG HeteroData builder |
| `src/retrieval/embeddings.py` (entity enrichment) | Simplified object class embeddings |
| `src/retrieval/faiss_index.py` (unified 1M index) | Per-video FAISS indices |
| `src/retrieval/pcst_solver.py` (localization + bridging) | Simplified PCST (no localization needed) |
| `src/retrieval/retriever.py` (orchestration) | New orchestration for per-video retrieval |
| `src/gnn/encoder.py` (homogeneous GATv2) | HeteroConv + GATv2Conv |
| `src/gnn/data_utils.py` (NetworkX → PyG) | Direct HeteroData slicing |

## What Stays the Same

- `src/config.py` — extended with new parameters
- `src/utils/seeds.py` — unchanged
- `src/utils/checkpoints.py` — unchanged
- `src/gnn/pooling.py` — attention pooling reusable
- `src/gnn/trainer.py` — training loop mostly reusable (adapt for HeteroData batches)
- LLM integration pattern (Phase 5)
- Verbalization pattern (Phase 4)
- Google Drive checkpointing strategy
- OpenRouter API integration
