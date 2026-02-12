# Dataset Reduction to 1/5 Size

This document explains changes made to reduce the dataset size from full (2826 train / 246 val) to approximately 1/5 size (600 train / 50 val) for faster iteration during development.

## Changes Made

### 1. Config Parameters (`src/config.py`)

Added dataset size limits:
```python
# Dataset size limits (for faster iteration during development)
max_train_examples: Optional[int] = 600  # ~1/5 of 2826, None = use all
max_val_examples: Optional[int] = 50     # ~1/5 of 246, None = use all
max_test_examples: Optional[int] = None  # Keep test set full for final eval
```

Reduced graph size requirements:
```python
# Reduced min sizes for 1/5 dataset (was 10000/30000 for full)
unified_graph_min_nodes: int = 2000
unified_graph_min_edges: int = 6000
```

### 2. Notebook Updates

**Cell 7 (Graph Construction):**
- Checks `config.max_train_examples` before building unified graph
- Limits train data using `.select(range(max_train_examples))`

**Cell 12 (GNN Training):**
- Limits both train and val data before passing to `GNNModel.build_from_checkpoint_or_train()`
- Prints dataset sizes for clarity

## Impact

### Before (Full Dataset)
- **Train:** 2,826 examples
- **Validation:** 246 examples
- **Unified Graph:** ~1M nodes, ~2.9M edges
- **Embedding Time:** ~10-15 minutes
- **GNN Training:** ~25-30 minutes

### After (1/5 Dataset)
- **Train:** 600 examples (~21% of full)
- **Validation:** 50 examples (~20% of full)
- **Unified Graph:** ~200K-300K nodes (estimated)
- **Embedding Time:** ~2-3 minutes
- **GNN Training:** ~5-7 minutes

## Checkpoint Invalidation

When switching dataset sizes, **you must delete old checkpoints** to force rebuilding:

```bash
# Delete these checkpoints from Google Drive:
checkpoints/unified_graph.pkl           # Built from old full dataset
checkpoints/entity_embeddings.pkl        # Built from old graph
checkpoints/faiss_index.bin              # Built from old embeddings
checkpoints/entity_mapping.pkl           # FAISS mappings
checkpoints/relation_embeddings.pkl      # Built from old graph
checkpoints/gnn_model.pt                 # Trained on old data
checkpoints/pyg_train_data.pkl           # Old training examples
checkpoints/pyg_val_data.pkl             # Old validation examples
checkpoints/gnn_training_history.json    # Old training metrics
```

Keep these:
```bash
checkpoints/dataset.pkl                  # Full dataset (we .select() from it)
checkpoints/huggingface_cache/           # Raw HF files
```

## Restoring Full Dataset

To switch back to full dataset for final evaluation:

1. Edit `src/config.py`:
   ```python
   max_train_examples: Optional[int] = None  # None = use all
   max_val_examples: Optional[int] = None
   ```

2. Delete checkpoints (same list as above)

3. Re-run notebook from Cell 7 onwards

## Notes

- Test set is always kept at full size (1,628 examples) for final evaluation
- The `dataset.pkl` checkpoint contains the **full** dataset — limiting is done at graph/training time
- When debugging retrieval/GNN, 600 examples is usually sufficient
- For final benchmarking, use the full dataset
