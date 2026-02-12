"""
Delete checkpoints to force rebuild after dataset size changes.

Based on DATASET_REDUCTION.md - deletes all checkpoints that depend on dataset size.

Run this in a notebook cell with:
    !python delete_checkpoints.py

Or copy the code directly into a notebook cell.
"""

from pathlib import Path
from src.config import BenchmarkConfig

def delete_checkpoints():
    """Delete all checkpoints that depend on dataset size."""
    config = BenchmarkConfig()
    checkpoint_dir = config.checkpoint_dir

    # Files to delete (from DATASET_REDUCTION.md)
    files_to_delete = [
        # Phase 1: Graph
        "unified_graph.pkl",

        # Phase 2: Retrieval
        "entity_embeddings.pkl",
        "faiss_index.bin",
        "entity_mapping.pkl",
        "relation_embeddings.pkl",

        # Phase 3: GNN
        "gnn_model.pt",
        "pyg_train_data.pkl",
        "pyg_val_data.pkl",
        "gnn_training_history.json",
    ]

    # Files to KEEP
    keep_files = [
        "dataset.pkl",  # Full dataset (we .select() from it)
        # huggingface_cache/ directory is also kept
    ]

    print("🗑️  Deleting Dataset-Dependent Checkpoints")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()

    deleted_count = 0
    missing_count = 0
    total_size = 0

    for filename in files_to_delete:
        filepath = checkpoint_dir / filename

        if filepath.exists():
            try:
                file_size = filepath.stat().st_size / (1024**2)  # MB
                filepath.unlink()
                print(f"✓ Deleted: {filename} ({file_size:.1f} MB)")
                deleted_count += 1
                total_size += file_size
            except Exception as e:
                print(f"✗ Error deleting {filename}: {e}")
        else:
            print(f"- Skipped: {filename} (not found)")
            missing_count += 1

    print("=" * 60)
    print(f"Summary: {deleted_count} deleted ({total_size:.1f} MB freed), {missing_count} not found")
    print()
    print("Kept (these are safe):")
    for filename in keep_files:
        filepath = checkpoint_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
    print()
    print("Next steps:")
    print("  1. Re-run Cell 7 (unified graph)")
    print("  2. Re-run Cell 9 (retrieval pipeline)")
    print("  3. Re-run Cell 12 (GNN training)")

if __name__ == "__main__":
    delete_checkpoints()
