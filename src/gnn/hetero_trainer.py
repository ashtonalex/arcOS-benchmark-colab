"""
Training loop for HeteroGATv2Encoder with focal loss and early stopping.
"""

from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
import numpy as np

from ..config import BenchmarkConfig
from .hetero_encoder import HeteroGATv2Encoder


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced answer node prediction."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HeteroGNNTrainer:
    """Trains HeteroGATv2Encoder on answer node prediction."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = HeteroGATv2Encoder(config).to(self.device)
        self.pred_head = nn.Linear(config.gnn_hidden_dim, 1).to(self.device)
        self.criterion = FocalLoss(gamma=2.0)

        self.optimizer = AdamW(
            list(self.encoder.parameters()) + list(self.pred_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_acc": [], "val_acc": [],
        }

    def _forward_single(self, data: HeteroData, query_embedding: torch.Tensor):
        """Forward pass for a single HeteroData graph."""
        node_emb, attn_scores, graph_emb = self.encoder(data, query_embedding)
        # Add graph context to node embeddings
        node_with_context = node_emb + graph_emb.unsqueeze(0).expand_as(node_emb)
        logits = self.pred_head(node_with_context).squeeze(-1)
        return logits, attn_scores

    def train_epoch(self, train_data: List[dict]) -> Dict[str, float]:
        """Train for one epoch over list of (data, query_emb, labels) dicts."""
        self.encoder.train()
        self.pred_head.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for item in tqdm(train_data, desc="Training", leave=False):
            data = item["data"].to(self.device)
            query_emb = item["query_embedding"].to(self.device)
            labels = item["labels"].to(self.device)

            logits, _ = self._forward_single(data, query_emb)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.pred_head.parameters()),
                max_norm=self.config.gradient_clip,
            )
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / max(len(train_data), 1)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss
        return metrics

    def validate(self, val_data: List[dict]) -> Dict[str, float]:
        """Validate without backward pass."""
        self.encoder.eval()
        self.pred_head.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for item in tqdm(val_data, desc="Validating", leave=False):
                data = item["data"].to(self.device)
                query_emb = item["query_embedding"].to(self.device)
                labels = item["labels"].to(self.device)

                logits, _ = self._forward_single(data, query_emb)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / max(len(val_data), 1)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss
        return metrics

    def _compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        preds_np = preds.numpy()
        labels_np = labels.numpy()
        acc = (preds_np == labels_np).mean()
        tp = ((preds_np == 1) & (labels_np == 1)).sum()
        fp = ((preds_np == 1) & (labels_np == 0)).sum()
        fn = ((preds_np == 0) & (labels_np == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"accuracy": float(acc), "precision": float(precision),
                "recall": float(recall), "f1": float(f1)}

    def train(
        self,
        train_data: List[dict],
        val_data: List[dict],
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        num_epochs = num_epochs or self.config.num_epochs
        patience = patience or self.config.patience

        print(f"Training HeteroGNN on {self.device}")
        print(f"Encoder: {self.encoder.__class__.__name__}")
        print(f"Max epochs: {num_epochs}, Patience: {patience}\n")

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_data)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            val_metrics = self.validate(val_data)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.scheduler.step(val_metrics["loss"])

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.3f}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  -> New best model (F1: {val_metrics['f1']:.3f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nTraining complete. Best epoch: {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, path: Path):
        checkpoint = {
            "encoder_state_dict": self.encoder.state_dict(),
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
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.pred_head.load_state_dict(checkpoint["pred_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.history = checkpoint["history"]
        print(f"Loaded checkpoint from epoch {self.best_epoch+1}")
