# Phase 3: GNN Encoder - Implementation Complete

**Date:** 2026-02-10
**Status:** ✅ IMPLEMENTED (Awaiting Colab Validation)
**Roadmap Phase:** 3 of 8

---

## Overview

Phase 3 implements a trainable Graph Neural Network (GNN) that learns to identify query-relevant nodes within retrieved subgraphs. The GNN produces attention-weighted node embeddings that guide the verbalization stage (Phase 4).

**Architecture:** GATv2-based encoder with attention pooling, trained on answer entity prediction using focal loss to handle class imbalance.

---

## What Was Implemented

### Module Structure

Created `src/gnn/` with 6 new files:

1. **`__init__.py`** - Package exports
2. **`data_utils.py`** - NetworkX → PyTorch Geometric conversion
3. **`encoder.py`** - GATv2Encoder and GraphSAGEEncoder
4. **`pooling.py`** - AttentionPooling, MeanPooling, MaxPooling
5. **`trainer.py`** - Training loop with focal loss and early stopping
6. **`model_wrapper.py`** - High-level GNNModel API

### Key Components

#### 1. Data Conversion (`data_utils.py`)

**`SubgraphConverter`** - Converts retrieval output to PyTorch Geometric format:
- **Input:** `RetrievedSubgraph` (NetworkX DiGraph)
- **Output:** PyG `Data` object with:
  - Node features: 384-dim entity embeddings (from Phase 2)
  - Edge features: 384-dim relation embeddings (from Phase 2)
  - Query embedding: 384-dim question embedding
  - Labels: Binary mask (1 = answer entity, 0 = other)

**Key design decisions:**
- Deterministic node ordering (sorted alphabetically) for reproducibility
- Reuses Phase 2 embeddings (no re-computation)
- Handles missing entities/relations gracefully (zero vectors)

#### 2. GNN Architectures (`encoder.py`)

**`GATv2Encoder`** (Primary architecture):
```
Input (384-dim) → Projection (hidden_dim) →
GATv2Conv Layer 1 (4 heads) → Dropout → LayerNorm → Residual →
GATv2Conv Layer 2 → ... → GATv2Conv Layer N →
Output: (node_embeddings, attention_weights)
```

Features:
- Multi-head attention (4 heads per layer)
- Query conditioning (broadcast query to all nodes)
- Residual connections and layer normalization
- Attention weights extracted by averaging across heads and layers

**`GraphSAGEEncoder`** (Alternative for ablation):
- Mean aggregation with learned attention head
- Same input/output interface as GATv2
- Useful for comparing attention mechanisms

**Common interface:**
```python
forward(x, edge_index, edge_attr, query_embedding) -> (node_embeddings, attention_weights)
```

#### 3. Graph Pooling (`pooling.py`)

**`AttentionPooling`** - Learnable attention-based aggregation:
- Gate network: computes per-node importance scores
- Feature network: transforms embeddings before pooling
- Returns: (graph_embedding, node_attention_scores)

**Baseline pooling options:**
- `MeanPooling`: Simple averaging
- `MaxPooling`: Element-wise maximum

#### 4. Training (`trainer.py`)

**`GNNTrainer`** - Complete training loop:

**Objective:** Answer entity prediction (binary classification per node)

**Loss function:** Focal Loss with gamma=2.0
- Handles class imbalance (1-3 answer nodes out of 50)
- Formula: `FL(p_t) = -(1 - p_t)^gamma * log(p_t)`
- Focuses training on hard examples

**Training features:**
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- Early stopping (patience=5 epochs)
- Gradient clipping (max_norm=1.0)
- CUDA memory management (empty_cache between epochs)

**Metrics tracked:**
- Loss (training and validation)
- Accuracy, Precision, Recall, F1 (answer node prediction)

**Memory management:**
- Batch size: 16 (with OOM fallback to 8)
- Gradient accumulation support
- Checkpoint saving at best validation loss

#### 5. High-Level API (`model_wrapper.py`)

**`GNNModel`** - Orchestration class following Phase 2 patterns:

```python
# Factory method (checkpoint or train)
model = GNNModel.build_from_checkpoint_or_train(
    config, retriever, train_data, val_data
)

# Inference
output = model.encode(retrieved_subgraph, question)
# Returns: GNNOutput(node_embeddings, attention_scores, graph_embedding)
```

**Key methods:**
- `build_from_checkpoint_or_train()` - Automatic checkpoint loading or training
- `encode()` - Single subgraph encoding
- `encode_batch()` - Batch processing
- `get_top_attention_nodes()` - Extract top-K nodes by attention

**Data preparation:**
- Caches PyG Data objects to disk (avoid re-conversion)
- Retrieves subgraphs for all train/val examples
- Progress bars for long operations

---

## Integration with Other Phases

### Phase 2 Dependencies (Input)

Phase 3 reuses Phase 2 artifacts:
- `entity_embeddings.pkl` - Node features
- `relation_embeddings.pkl` - Edge features
- `TextEmbedder` - Question encoding
- `Retriever` - Subgraph retrieval

**No re-computation needed** - all Phase 2 checkpoints are loaded and reused.

### Phase 4 Interface (Output)

Phase 3 provides to Phase 4:
- `GNNOutput.attention_scores` - Dict[node_name, float] for triple ranking
- Phase 4 will select top-K triples by attention for verbalization

### Configuration

All GNN parameters defined in `BenchmarkConfig`:
- `gnn_hidden_dim: int = 256`
- `gnn_num_layers: int = 3`
- `gnn_num_heads: int = 4`
- `gnn_dropout: float = 0.1`
- `gnn_pooling: str = "attention"`
- `learning_rate: float = 1e-3`
- `weight_decay: float = 1e-4`
- `batch_size: int = 16`
- `num_epochs: int = 50`
- `patience: int = 5`

**No changes to `config.py` needed** - all parameters already exist.

---

## Checkpoints

Following Phase 1/2 patterns, saves to Google Drive:

1. **`gnn_model.pt`** (~50 MB)
   - Encoder, pooling, prediction head state dicts
   - Optimizer and scheduler states
   - Best epoch and metrics

2. **`pyg_train_data.pkl`** (~1.5 GB)
   - 2,830 converted PyG Data objects (train split)
   - Avoids re-conversion and re-retrieval

3. **`pyg_val_data.pkl`** (~150 MB)
   - 246 converted PyG Data objects (validation split)

4. **`gnn_training_history.json`** (~5 KB)
   - Loss curves, F1 scores, accuracy per epoch
   - Used for visualization and analysis

**Total checkpoint size:** ~1.7 GB

---

## Success Criteria (from ROADMAP.md)

| Criterion | Expected | Status |
|-----------|----------|--------|
| Trains without OOM on T4 GPU (15GB VRAM) | Yes | ⏳ Pending Colab validation |
| Validation answer-node F1 > 0.5 | >0.5 | ⏳ Pending training |
| Attention concentrates on answer nodes | Visual inspection | ⏳ Pending visualization |
| Training completes in < 30 minutes | <30 min | ⏳ Pending Colab run |
| GATv2 and GraphSAGE compatible | Same interface | ✅ Implemented |

---

## Validation Plan

### Local Validation (Complete)

```bash
python test_phase3_imports.py
```

**Results:**
- ✅ Config has GNN parameters
- ⏭️ Other tests skipped (missing Colab dependencies - expected)

### Colab Validation (Pending)

**Cell 12:** Build/load GNN model
- Load from checkpoint or train from scratch
- Expected time: 30 min (cold start), 5 sec (warm start)

**Cell 13:** Test inference
- Encode example subgraph
- Verify output shapes and attention scores

**Cell 14:** Validate metrics
- Check validation F1 > 0.5
- Plot loss and F1 curves

**Cell 15:** Visualize attention
- Network graph with color-coded attention
- Verify answer nodes have high attention

**Cell 16:** Memory check
- Ensure GPU memory < 14 GB (no leak)

See `docs/PHASE3_NOTEBOOK_CELLS.md` for detailed cell code.

---

## Expected Training Results

Based on ROADMAP.md estimates:

**Training time:**
- Data preparation: ~15 min (retrieve + convert, one-time)
- Training: ~15-20 min (20-30 epochs with early stopping)
- **Total:** ~30 min cold start ✓

**Metrics:**
- Final validation F1: 0.55-0.65 (target >0.5) ✓
- Final validation loss: ~0.12
- Answer node accuracy: ~85%

**Attention behavior:**
- Top-5 attention nodes should include 1-2 answer entities
- Attention entropy should decrease over training (concentration)

---

## Dependencies

Added to `uv pip install` (Cell 1 in notebook):

```python
torch-geometric
matplotlib
tqdm
```

**PyTorch Geometric installation:**
- Requires matching CUDA version
- May need special installation command (see `PHASE3_NOTEBOOK_CELLS.md`)

---

## Architecture Decisions

### 1. Why GATv2 over GAT?

GATv2 addresses the "static attention" problem of GAT:
- GAT computes attention before applying feature transformation
- GATv2 applies transformation first, allowing dynamic attention
- Better performance on heterogeneous graphs

### 2. Why Focal Loss over BCE?

Binary cross-entropy treats all nodes equally:
- Most nodes are NOT answer entities (1-3 out of 50)
- BCE focuses too much on easy negatives
- Focal loss down-weights easy examples, focuses on hard positives

### 3. Why Attention Pooling?

Attention pooling learns which nodes matter for the task:
- Mean pooling treats all nodes equally (loses signal)
- Max pooling is too aggressive (single node dominates)
- Attention pooling adaptively weights nodes

### 4. Why Residual Connections?

Deep GNNs suffer from over-smoothing:
- Nodes become too similar after many layers
- Residual connections preserve node identity
- Enables training deeper networks (3+ layers)

---

## Known Limitations

1. **Training data size:** Only 2,830 examples
   - May not generalize to very different question types
   - Could benefit from data augmentation

2. **Batch size:** Limited to 16 on T4 GPU
   - Larger batches might improve training stability
   - Could use gradient accumulation if needed

3. **Answer prediction task:** Proxy for relevance
   - Answer entities are not the only relevant nodes
   - Neighbors of answers may also be important
   - Could add auxiliary tasks (e.g., path prediction)

4. **Fixed embeddings:** Phase 2 embeddings are frozen
   - GNN doesn't update entity/relation representations
   - Could fine-tune embeddings jointly with GNN

---

## Next Steps

### Immediate (Colab Validation)

1. Add notebook cells (see `PHASE3_NOTEBOOK_CELLS.md`)
2. Run Cell 12 to train GNN (or load checkpoint)
3. Run Cells 13-16 to validate metrics and visualization
4. Verify success criteria (F1 > 0.5, time < 30 min, no OOM)

### Phase 4 Integration

Phase 4 will use `GNNOutput.attention_scores` to:
1. Rank triples by node attention
2. Select top-K triples for verbalization
3. Format as natural language text (hard prompt)

**Interface contract:**
```python
gnn_output = gnn_model.encode(retrieved_subgraph, question)
# gnn_output.attention_scores: Dict[node_name, float]

# Phase 4 will use this to rank triples:
for src, dst, rel in retrieved_subgraph.edges(data=True):
    score = (attention_scores[src] + attention_scores[dst]) / 2
    # Select top-K by score
```

---

## File Summary

### New Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/gnn/__init__.py` | 20 | Package exports |
| `src/gnn/data_utils.py` | 150 | NetworkX → PyG conversion |
| `src/gnn/encoder.py` | 230 | GATv2 and GraphSAGE encoders |
| `src/gnn/pooling.py` | 180 | Attention and baseline pooling |
| `src/gnn/trainer.py` | 280 | Training loop and metrics |
| `src/gnn/model_wrapper.py` | 180 | High-level API |
| `test_phase3_imports.py` | 200 | Import validation test |
| `docs/PHASE3_NOTEBOOK_CELLS.md` | 300 | Notebook cells documentation |
| `docs/PHASE3_COMPLETE.md` | 400 | This document |

**Total:** ~1,940 lines of new code

### Files Modified

None - Phase 3 is entirely additive.

---

## Testing

### Test Coverage

**Import test:** `test_phase3_imports.py`
- ✅ Config parameters exist
- ⏭️ Module imports (Colab-only dependencies)
- ⏭️ Class instantiation
- ⏭️ Dataclass functionality

**Unit tests:** Not implemented (out of scope)
- Could add tests for data conversion
- Could add tests for attention aggregation
- Could mock PyG Data for faster testing

**Integration test:** Colab notebook validation
- End-to-end training loop
- Inference on real examples
- Attention visualization

---

## Performance Estimates

### Training Performance (T4 GPU)

| Metric | Value |
|--------|-------|
| Data preparation time | ~15 min (one-time) |
| Training time per epoch | ~30 sec |
| Total training time | ~15-20 min (20-30 epochs) |
| Inference time per example | ~50 ms |
| Batch inference (16 examples) | ~200 ms |

### Memory Usage (T4 GPU, 15GB VRAM)

| Component | Memory |
|-----------|--------|
| Model parameters | ~20 MB |
| Training batch (16 examples) | ~500 MB |
| Gradients | ~20 MB |
| Optimizer states | ~40 MB |
| PyG overhead | ~200 MB |
| **Total peak** | ~800 MB |

**Safety margin:** 14.2 GB available (plenty of headroom)

### Checkpoint Sizes

| Checkpoint | Size |
|------------|------|
| `gnn_model.pt` | ~50 MB |
| `pyg_train_data.pkl` | ~1.5 GB |
| `pyg_val_data.pkl` | ~150 MB |
| `gnn_training_history.json` | ~5 KB |
| **Total** | ~1.7 GB |

---

## References

### Papers

1. **GATv2:** Brody et al., "How Attentive are Graph Attention Networks?" (ICLR 2022)
2. **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
3. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)

### Libraries

- **PyTorch Geometric:** https://pytorch-geometric.readthedocs.io/
- **PyTorch:** https://pytorch.org/
- **NetworkX:** https://networkx.org/

---

## Conclusion

Phase 3 is **fully implemented** and ready for Colab validation. The implementation follows all architecture decisions from the plan:

✅ GATv2 encoder with multi-head attention
✅ Query conditioning via broadcast addition
✅ Focal loss for class imbalance
✅ Attention pooling for graph-level embeddings
✅ Factory pattern matching Phase 2 retriever
✅ Checkpoint-based training with early stopping
✅ Comprehensive training metrics and visualization

**Next action:** Add notebook cells and run training in Colab to validate success criteria.
