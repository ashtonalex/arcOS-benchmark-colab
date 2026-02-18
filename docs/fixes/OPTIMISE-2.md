  Logging & Visualization Recommendations for the Notebook

  The trainer already records history with 6 metrics per epoch. Here's what to add:

  1. Training Curves (essential)

  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 3, figsize=(15, 4))

  # Loss
  axes[0].plot(history['train_loss'], label='Train')
  axes[0].plot(history['val_loss'], label='Val')
  axes[0].axvline(trainer.best_epoch, color='r', linestyle='--', alpha=0.5, label='Best')
  axes[0].set_title('Loss'); axes[0].legend()

  # F1
  axes[1].plot(history['train_f1'], label='Train')
  axes[1].plot(history['val_f1'], label='Val')
  axes[1].set_title('F1 Score'); axes[1].legend()

  # Accuracy
  axes[2].plot(history['train_acc'], label='Train')
  axes[2].plot(history['val_acc'], label='Val')
  axes[2].set_title('Accuracy'); axes[2].legend()

  plt.tight_layout(); plt.show()

  2. Class Imbalance Monitoring

  Log positive/negative ratio per batch. With PCST subgraphs of ~20 nodes and typically 1-2 answer nodes, expect ~5-10% positive rate. If F1 stays near 0, the model may be      
  predicting all-negative.

  # Add to train_epoch, inside the batch loop:
  pos_rate = batch.y.mean().item()

  Aggregate and plot across epochs as a sanity check.

  3. Attention Distribution Visualization

  After inference on a sample, visualize which nodes the GNN attends to:

  import networkx as nx

  output = model.encode(retrieved, question)
  top_nodes = model.get_top_attention_nodes(output, top_k=10)

  # Color nodes by attention score
  node_colors = [output.attention_scores.get(n, 0) for n in retrieved.subgraph.nodes()]
  nx.draw(retrieved.subgraph, node_color=node_colors, cmap='YlOrRd',
          with_labels=True, font_size=6, node_size=300)
  plt.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd'), label='Attention')

  4. Learning Rate Tracking

  The ReduceLROnPlateau scheduler silently adjusts LR. Log it:

  # Add to history dict
  "learning_rate": []

  # In train loop, after scheduler.step():
  current_lr = self.optimizer.param_groups[0]['lr']
  self.history["learning_rate"].append(current_lr)

  5. Gradient Norm Monitoring

  Already clipping at 1.0, but log the pre-clip norm to detect exploding gradients:

  # Before clip_grad_norm_, compute and log:
  total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
  # total_norm is the actual norm before clipping

  6. Per-Epoch Confusion Matrix (compact)

  # Add to _compute_metrics return:
  "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(((preds==0)&(labels==0)).sum())

  Then plot the TP/FP/FN/TN breakdown over epochs as a stacked bar chart to see if recall is improving or precision is degrading.