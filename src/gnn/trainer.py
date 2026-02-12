"""
GNN training loop with focal loss and early stopping.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from tqdm.auto import tqdm
import numpy as np

from ..config import BenchmarkConfig
from ..utils.checkpoints import save_checkpoint
from .encoder import GATv2Encoder, GraphSAGEEncoder
from .pooling import AttentionPooling, MeanPooling, MaxPooling


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in answer node prediction.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the correct class.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits [num_nodes]
            targets: Binary labels [num_nodes]

        Returns:
            loss: Scalar loss
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Weighted loss
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class GNNTrainer:
    """
    Trains GNN encoder on answer entity prediction task.

    Uses focal loss to handle class imbalance (most nodes are not answers).
    Implements early stopping and learning rate scheduling.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_type: str = "gatv2",
        pooling_type: str = "attention",
    ):
        """
        Args:
            config: BenchmarkConfig instance
            encoder_type: 'gatv2' or 'graphsage'
            pooling_type: 'attention', 'mean', or 'max'
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build encoder
        if encoder_type == "gatv2":
            self.encoder = GATv2Encoder(config).to(self.device)
        elif encoder_type == "graphsage":
            self.encoder = GraphSAGEEncoder(config).to(self.device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Build pooling layer
        if pooling_type == "attention":
            self.pooling = AttentionPooling(config).to(self.device)
        elif pooling_type == "mean":
            self.pooling = MeanPooling(config).to(self.device)
        elif pooling_type == "max":
            self.pooling = MaxPooling(config).to(self.device)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Prediction head (graph embedding -> answer score per node)
        self.pred_head = nn.Linear(config.gnn_hidden_dim, 1).to(self.device)

        # Loss function
        self.criterion = FocalLoss(gamma=2.0)

        # Optimizer
        self.optimizer = AdamW(
            list(self.encoder.parameters())
            + list(self.pooling.parameters())
            + list(self.pred_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

        # Training state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_f1": [],
            "val_f1": [],
            "train_acc": [],
            "val_acc": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: PyG DataLoader for training data

        Returns:
            metrics: Dict with loss, accuracy, precision, recall, F1
        """
        self.encoder.train()
        self.pooling.train()
        self.pred_head.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            # Forward pass through encoder
            node_embeddings, attention_weights = self.encoder(
                batch.x, batch.edge_index, batch.edge_attr, batch.query_embedding[0]
            )

            # Pool to graph embedding
            graph_embedding, _ = self.pooling(node_embeddings, batch.batch)

            # Predict answer scores per node
            # Combine node embedding with graph context
            node_with_context = node_embeddings + graph_embedding[batch.batch]
            logits = self.pred_head(node_with_context).squeeze(-1)

            # Compute loss
            loss = self.criterion(logits, batch.y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters())
                + list(self.pooling.parameters())
                + list(self.pred_head.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.detach().cpu())
            all_labels.append(batch.y.detach().cpu())

        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: PyG DataLoader for validation data

        Returns:
            metrics: Dict with loss, accuracy, precision, recall, F1
        """
        self.encoder.eval()
        self.pooling.eval()
        self.pred_head.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)

                # Forward pass
                node_embeddings, attention_weights = self.encoder(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.query_embedding[0],
                )

                graph_embedding, _ = self.pooling(node_embeddings, batch.batch)
                node_with_context = node_embeddings + graph_embedding[batch.batch]
                logits = self.pred_head(node_with_context).squeeze(-1)

                # Compute loss
                loss = self.criterion(logits, batch.y)

                # Track metrics
                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        # Compute metrics
        avg_loss = total_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss

        return metrics

    def _compute_metrics(
        self, preds: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            preds: Binary predictions [num_nodes]
            labels: Binary labels [num_nodes]

        Returns:
            metrics: Dict with accuracy, precision, recall, F1
        """
        preds = preds.numpy()
        labels = labels.numpy()

        # Accuracy
        acc = (preds == labels).mean()

        # Precision, Recall, F1
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: PyG DataLoader for training
            val_loader: PyG DataLoader for validation
            num_epochs: Max epochs (defaults to config.num_epochs)
            patience: Early stopping patience (defaults to config.patience)

        Returns:
            history: Training history dict
        """
        num_epochs = num_epochs or self.config.num_epochs
        patience = patience or self.config.patience

        print(f"Training GNN on {self.device}")
        print(f"Encoder: {self.encoder.__class__.__name__}")
        print(f"Max epochs: {num_epochs}, Patience: {patience}\n")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.3f}"
            )

            # Early stopping check
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  → New best model (F1: {val_metrics['f1']:.3f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nTraining complete. Best epoch: {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, path: Path):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
            "pooling_state_dict": self.pooling.state_dict(),
            "pred_head_state_dict": self.pred_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": {
                "gnn_hidden_dim": self.config.gnn_hidden_dim,
                "gnn_num_layers": self.config.gnn_num_layers,
                "gnn_num_heads": self.config.gnn_num_heads,
                "gnn_dropout": self.config.gnn_dropout,
            },
        }
        save_checkpoint(checkpoint, path, format="pickle")

    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.pooling.load_state_dict(checkpoint["pooling_state_dict"])
        self.pred_head.load_state_dict(checkpoint["pred_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint from epoch {self.best_epoch+1}")
