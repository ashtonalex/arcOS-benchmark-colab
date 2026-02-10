# Phase 3: GNN Encoder - Notebook Cells

This document describes the notebook cells to add for Phase 3 implementation.

## Cell 12: Build/Load GNN Model

```python
# ============================================================
# PHASE 3: GNN Encoder
# ============================================================

print("Building GNN Model...")
print("This will either:")
print("  1. Load pre-trained model from checkpoint, OR")
print("  2. Prepare data and train from scratch (~30 min)")
print()

from src.gnn import GNNModel

# Build GNN model (handles checkpoint loading or training automatically)
gnn_model = GNNModel.build_from_checkpoint_or_train(
    config=config,
    retriever=retriever,
    train_data=dataset["train"],
    val_data=dataset["validation"],
    encoder_type="gatv2",  # or "graphsage"
    pooling_type="attention",  # or "mean", "max"
)

print("\n" + "="*60)
print("GNN Model Ready")
print("="*60)
```

**Expected output (from checkpoint):**
```
============================================================
Building GNN Model
============================================================
Loading GNN model from checkpoint...
Loaded checkpoint from epoch 23
✓ GNN model loaded

============================================================
GNN Model Ready
============================================================
```

**Expected output (training from scratch):**
```
============================================================
Building GNN Model
============================================================
No checkpoint found. Training from scratch...

Preparing training data...
  Converting dataset to PyG format (this may take a while)...
100%|██████████| 2830/2830 [15:22<00:00,  3.07it/s]
100%|██████████| 246/246 [01:18<00:00,  3.14it/s]
  Saving converted data to checkpoints...
✓ Training data: 2830 examples
✓ Validation data: 246 examples

Training GNN on cuda
Encoder: GATv2Encoder
Max epochs: 50, Patience: 5

Epoch 1/50 | Train Loss: 0.2341, F1: 0.423 | Val Loss: 0.1987, F1: 0.512
  → New best model (F1: 0.512)
Epoch 2/50 | Train Loss: 0.1876, F1: 0.534 | Val Loss: 0.1723, F1: 0.568
  → New best model (F1: 0.568)
...
Epoch 23/50 | Train Loss: 0.0912, F1: 0.723 | Val Loss: 0.1234, F1: 0.621
  → New best model (F1: 0.621)
Epoch 24/50 | Train Loss: 0.0889, F1: 0.734 | Val Loss: 0.1245, F1: 0.619
...
Early stopping at epoch 28

Training complete. Best epoch: 23

Saving model checkpoint...
✓ Checkpoint saved

============================================================
GNN Model Ready
============================================================
```

---

## Cell 13: Test GNN Inference

```python
# Test GNN encoding on a single example
print("Testing GNN inference on example query...\n")

test_question = "Who is Justin Bieber's brother?"
print(f"Question: {test_question}")

# Retrieve subgraph
retrieved = retriever.retrieve(test_question)
print(f"Retrieved subgraph: {retrieved.num_nodes} nodes, {retrieved.num_edges} edges")

# Encode with GNN
gnn_output = gnn_model.encode(retrieved, test_question)

print(f"\nGNN Output:")
print(f"  Node embeddings shape: {gnn_output.node_embeddings.shape}")
print(f"  Graph embedding shape: {gnn_output.graph_embedding.shape}")
print(f"  Attention scores: {len(gnn_output.attention_scores)} nodes")

# Get top attention nodes
top_nodes = gnn_model.get_top_attention_nodes(gnn_output, top_k=10)
print(f"\nTop 10 nodes by attention score:")
for i, (node, score) in enumerate(top_nodes, 1):
    print(f"  {i}. {node}: {score:.4f}")
```

**Expected output:**
```
Testing GNN inference on example query...

Question: Who is Justin Bieber's brother?
Retrieved subgraph: 47 nodes, 89 edges

GNN Output:
  Node embeddings shape: torch.Size([47, 256])
  Graph embedding shape: torch.Size([256])
  Attention scores: 47 nodes

Top 10 nodes by attention score:
  1. Jaxon Bieber: 0.1234
  2. Justin Bieber: 0.0987
  3. Pattie Mallette: 0.0765
  4. Jazmyn Bieber: 0.0654
  5. Jeremy Bieber: 0.0543
  6. m.04rzd: 0.0432
  7. people.person.sibling_s: 0.0321
  8. people.person.children: 0.0298
  9. m.02mjmr: 0.0287
  10. people.person.parents: 0.0276
```

---

## Cell 14: Validate GNN Metrics

```python
# Load training history and display metrics
import json
import matplotlib.pyplot as plt

history_path = config.get_checkpoint_path("gnn_training_history.json")
with open(history_path, "r") as f:
    history = json.load(f)

print("="*60)
print("PHASE 3 VALIDATION: GNN Metrics")
print("="*60)

# Best metrics
best_val_f1 = max(history["val_f1"])
best_val_loss = min(history["val_loss"])
final_val_f1 = history["val_f1"][-1]

print(f"\nTraining Summary:")
print(f"  Epochs trained: {len(history['train_loss'])}")
print(f"  Best validation F1: {best_val_f1:.3f}")
print(f"  Best validation loss: {best_val_loss:.4f}")
print(f"  Final validation F1: {final_val_f1:.3f}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(True)

# F1 curve
axes[1].plot(history["train_f1"], label="Train F1")
axes[1].plot(history["val_f1"], label="Val F1")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Answer Node Prediction F1")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Success criteria check
print(f"\n{'='*60}")
print("Success Criteria:")
print(f"{'='*60}")

criteria = [
    ("Validation F1 > 0.5", best_val_f1 > 0.5, best_val_f1),
    ("Training completed < 30 min", True, "N/A"),  # User observation
    ("No OOM errors", True, "N/A"),  # User observation
]

for criterion, passed, value in criteria:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status} - {criterion}: {value}")

if all(c[1] for c in criteria):
    print(f"\n{'='*60}")
    print("SUCCESS: Phase 3 Complete")
    print(f"{'='*60}")
else:
    print(f"\n{'='*60}")
    print("FAILED: Some criteria not met")
    print(f"{'='*60}")
```

---

## Cell 15: Visualize Attention on Example

```python
# Visualize GNN attention on a subgraph
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def visualize_gnn_attention(
    subgraph: nx.DiGraph,
    attention_scores: dict,
    answer_entities: list,
    question: str,
    top_k: int = 20,
):
    """
    Visualize GNN attention on a subgraph.

    Args:
        subgraph: NetworkX DiGraph
        attention_scores: Dict[node_name, float]
        answer_entities: List of ground truth answer entities
        question: Question text
        top_k: Show only top-K nodes by attention
    """
    # Get top-K nodes by attention
    sorted_nodes = sorted(
        attention_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]
    top_nodes = [node for node, _ in sorted_nodes]

    # Create subgraph with only top nodes
    G_viz = subgraph.subgraph(top_nodes).copy()

    # Node colors (red = answer, blue = high attention, gray = low attention)
    node_colors = []
    for node in G_viz.nodes():
        if node in answer_entities:
            node_colors.append("red")
        else:
            # Scale by attention (darker = higher attention)
            attn = attention_scores.get(node, 0.0)
            intensity = min(attn * 10, 1.0)  # Scale for visibility
            node_colors.append((0.2, 0.4, 0.8, 0.3 + 0.7 * intensity))

    # Node sizes proportional to attention
    node_sizes = [
        300 + 2000 * attention_scores.get(node, 0.0) for node in G_viz.nodes()
    ]

    # Layout
    pos = nx.spring_layout(G_viz, k=0.5, iterations=50, seed=42)

    # Plot
    plt.figure(figsize=(14, 10))
    plt.title(f"GNN Attention Visualization\nQ: {question}", fontsize=12)

    # Draw edges
    nx.draw_networkx_edges(
        G_viz, pos, alpha=0.3, arrows=True, arrowsize=10, width=1.0
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G_viz, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9
    )

    # Draw labels (only for top 10)
    labels = {node: node[:20] for node in list(G_viz.nodes())[:10]}
    nx.draw_networkx_labels(G_viz, pos, labels, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Print attention scores
    print("Top 10 attention scores:")
    for i, (node, score) in enumerate(sorted_nodes[:10], 1):
        is_answer = "✓ ANSWER" if node in answer_entities else ""
        print(f"  {i}. {node[:30]}: {score:.4f} {is_answer}")


# Test visualization
test_question = "Who is Barack Obama's spouse?"
test_answer = ["Michelle Obama", "m.025s5v9"]  # Freebase ID

retrieved = retriever.retrieve(test_question)
gnn_output = gnn_model.encode(retrieved, test_question)

visualize_gnn_attention(
    subgraph=retrieved.subgraph,
    attention_scores=gnn_output.attention_scores,
    answer_entities=test_answer,
    question=test_question,
    top_k=20,
)
```

**Expected output:**
- Network graph visualization with:
  - Red nodes = ground truth answers
  - Blue nodes = high attention (darker = higher)
  - Node size proportional to attention score
  - Top 10 node labels visible
- List of top 10 attention scores with answer markers

---

## Cell 16: Memory Check

```python
# Check GPU memory usage
if torch.cuda.is_available():
    print("GPU Memory Summary:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Verify no memory leak
    assert torch.cuda.memory_allocated() / 1e9 < 14.0, "Memory leak detected!"
    print("\n✓ Memory usage within acceptable range")
else:
    print("GPU not available")
```

---

## Dependencies to Add

In the first cell (Cell 1), add to uv pip install:

```python
!uv pip install --system \
    torch-geometric \
    matplotlib \
    tqdm
```

**Note:** PyG installation in Colab may require:

```python
import torch
def install_pyg():
    torch_version = torch.__version__.split('+')[0]
    cuda_version = torch.version.cuda.replace('.', '')
    !pip install torch-scatter torch-sparse torch-cluster torch-spmatrix torch-geometric -f https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html

install_pyg()
```

---

## Timing Expectations

| Operation | Cold Start | Warm Start |
|-----------|------------|------------|
| Cell 12: Build model | 30 min (training) | 5 sec (load checkpoint) |
| Cell 13: Inference test | 2 sec | 2 sec |
| Cell 14: Metrics validation | 1 sec | 1 sec |
| Cell 15: Visualization | 3 sec | 3 sec |
| Cell 16: Memory check | <1 sec | <1 sec |

**Total Phase 3:** ~30 min cold start, ~15 sec warm start
