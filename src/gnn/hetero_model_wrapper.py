"""
High-level HeteroGNN model API for video scene graph pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gc
import torch
import numpy as np
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

from ..config import BenchmarkConfig
from ..retrieval.video_retriever import VideoRetriever, RetrievalResult
from ..utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint
from .hetero_trainer import HeteroGNNTrainer


class HeteroGNNModel:
    """
    High-level HeteroGNN model API matching the GNNModel pattern.

    Usage:
        model = HeteroGNNModel.build_from_checkpoint_or_train(
            config, retriever, train_data, val_data, scene_graphs
        )
        node_emb, attn_scores, graph_emb = model.encode(subgraph, question)
    """

    def __init__(self, config: BenchmarkConfig, trainer: HeteroGNNTrainer):
        self.config = config
        self.trainer = trainer
        self.device = trainer.device
        self.trainer.encoder.eval()
        self.trainer.pred_head.eval()

    @classmethod
    def build_from_checkpoint_or_train(
        cls,
        config: BenchmarkConfig,
        retriever: VideoRetriever,
        train_samples: List[Dict],
        val_samples: List[Dict],
        scene_graphs: Dict[str, HeteroData],
    ) -> "HeteroGNNModel":
        """Factory: load from checkpoint or train from scratch.

        Args:
            config: BenchmarkConfig instance.
            retriever: Trained VideoRetriever.
            train_samples: List of AGQA sample dicts (question, answer, video_id).
            val_samples: Validation samples.
            scene_graphs: Dict mapping video_id to built HeteroData scene graphs.

        Returns:
            HeteroGNNModel instance.
        """
        print("=" * 60)
        print("Building HeteroGNN Model")
        print("=" * 60)

        model_path = config.get_checkpoint_path("hetero_gnn_model.pt")
        history_path = config.get_checkpoint_path("hetero_gnn_history.json")
        train_data_path = config.get_checkpoint_path("hetero_pyg_train.pkl")
        val_data_path = config.get_checkpoint_path("hetero_pyg_val.pkl")

        trainer = HeteroGNNTrainer(config)

        if checkpoint_exists(model_path):
            print("Loading HeteroGNN from checkpoint...")
            trainer.load_checkpoint(model_path)
        else:
            print("No checkpoint found. Training from scratch...\n")

            if checkpoint_exists(train_data_path) and checkpoint_exists(val_data_path):
                print("  Loading cached PyG training data...")
                train_pyg = load_checkpoint(train_data_path, format="pickle")
                val_pyg = load_checkpoint(val_data_path, format="pickle")
            else:
                print("  Preparing training data...")
                train_pyg = cls._prepare_training_data(
                    train_samples, retriever, scene_graphs, config
                )
                val_pyg = cls._prepare_training_data(
                    val_samples, retriever, scene_graphs, config
                )
                save_checkpoint(train_pyg, train_data_path, format="pickle")
                save_checkpoint(val_pyg, val_data_path, format="pickle")

            print(f"  Training data: {len(train_pyg)} examples")
            print(f"  Validation data: {len(val_pyg)} examples\n")

            # Free embedder GPU memory before training
            try:
                retriever.embedder.model.to("cpu")
                retriever.embedder.device = "cpu"
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            history = trainer.train(train_pyg, val_pyg)

            print("\nSaving model checkpoint...")
            trainer.save_checkpoint(model_path)
            save_checkpoint(history, history_path, format="json")
            print("Checkpoint saved")

        print("\n" + "=" * 60)
        print("HeteroGNN Model Ready")
        print("=" * 60 + "\n")

        return cls(config, trainer)

    @classmethod
    def _prepare_training_data(
        cls,
        samples: List[Dict],
        retriever: VideoRetriever,
        scene_graphs: Dict[str, HeteroData],
        config: BenchmarkConfig,
    ) -> List[dict]:
        """Prepare list of {data, query_embedding, labels} dicts for training."""
        pyg_items = []
        skipped = 0

        # Batch embed all questions
        print("  Pre-computing query embeddings...")
        questions = [s["question"] for s in samples]
        query_embeddings = retriever.embedder.embed_texts(questions, show_progress=True)

        for idx, sample in enumerate(tqdm(samples, desc="Preparing PyG data")):
            video_id = sample["video_id"]
            if video_id not in scene_graphs:
                skipped += 1
                continue

            scene_graph = scene_graphs[video_id]
            answer_text = sample["answer"]
            question = sample["question"]

            try:
                result = retriever.retrieve(question, scene_graph)
                subgraph = result.subgraph

                # Label answer nodes by matching answer text to object_names
                labels = cls._label_answer_nodes(subgraph, answer_text)
                if labels.sum().item() == 0:
                    skipped += 1
                    continue

                query_emb = torch.tensor(query_embeddings[idx], dtype=torch.float32)

                pyg_items.append({
                    "data": subgraph,
                    "query_embedding": query_emb,
                    "labels": labels,
                })
            except Exception as e:
                skipped += 1
                continue

            if idx % 50 == 0 and idx > 0:
                gc.collect()

        if skipped > 0:
            total = skipped + len(pyg_items)
            print(f"  Skipped {skipped}/{total} examples ({skipped/max(total,1):.1%})")

        return pyg_items

    @staticmethod
    def _label_answer_nodes(subgraph: HeteroData, answer_text: str) -> torch.Tensor:
        """Label nodes whose object_name matches the answer text."""
        num_nodes = subgraph["object"].x.shape[0]
        labels = torch.zeros(num_nodes)
        names = getattr(subgraph, "object_names", None)
        if names is None:
            return labels

        answer_lower = answer_text.strip().lower()
        for i, name in enumerate(names):
            if name.strip().lower() == answer_lower:
                labels[i] = 1.0
        return labels

    def encode(
        self,
        subgraph: HeteroData,
        question: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a retrieved subgraph with query conditioning.

        Args:
            subgraph: HeteroData from VideoRetriever.
            question: Natural language question.

        Returns:
            (node_embeddings, attention_scores, graph_embedding)
        """
        query_emb = self.trainer.encoder.input_proj.weight.new_zeros(self.config.embedding_dim)

        # Use embedder if available
        if hasattr(self, '_embedder') and self._embedder is not None:
            emb = self._embedder.embed_texts([question])[0]
            query_emb = torch.tensor(emb, dtype=torch.float32).to(self.device)
        else:
            query_emb = query_emb.to(self.device)

        subgraph = subgraph.to(self.device)

        with torch.no_grad():
            node_emb, attn_scores, graph_emb = self.trainer.encoder(subgraph, query_emb)

        return node_emb.cpu(), attn_scores.cpu(), graph_emb.cpu()

    def set_embedder(self, embedder):
        """Set the text embedder for query encoding at inference time."""
        self._embedder = embedder
