"""
High-level GNN model API for integration with the pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional
import gc
import torch
import numpy as np
from torch_geometric.data import DataLoader
from datasets import Dataset

from ..config import BenchmarkConfig
from ..retrieval.retriever import Retriever, RetrievedSubgraph
from ..retrieval.embeddings import TextEmbedder
from ..utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint
from .data_utils import SubgraphConverter, GNNOutput
from .trainer import GNNTrainer


class GNNModel:
    """
    High-level GNN model API following the Phase 2 Retriever pattern.

    Provides a simple interface:
        model = GNNModel.build_from_checkpoint_or_train(config, retriever, train_data, val_data)
        output = model.encode(retrieved_subgraph, question)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        trainer: GNNTrainer,
        converter: SubgraphConverter,
    ):
        """
        Args:
            config: BenchmarkConfig instance
            trainer: Trained GNNTrainer instance
            converter: SubgraphConverter for data preprocessing
        """
        self.config = config
        self.trainer = trainer
        self.converter = converter
        self.device = trainer.device

        # Set to eval mode
        self.trainer.encoder.eval()
        self.trainer.pooling.eval()
        self.trainer.pred_head.eval()

    @classmethod
    def build_from_checkpoint_or_train(
        cls,
        config: BenchmarkConfig,
        retriever: Retriever,
        train_data: Dataset,
        val_data: Dataset,
        encoder_type: str = "gatv2",
        pooling_type: str = "attention",
    ) -> "GNNModel":
        """
        Factory method: load from checkpoint or train from scratch.

        Args:
            config: BenchmarkConfig instance
            retriever: Trained Retriever instance (provides embeddings and retrieval)
            train_data: HuggingFace Dataset (train split)
            val_data: HuggingFace Dataset (validation split)
            encoder_type: 'gatv2' or 'graphsage'
            pooling_type: 'attention', 'mean', or 'max'

        Returns:
            GNNModel instance
        """
        print("="*60)
        print("Building GNN Model")
        print("="*60)

        # Paths
        model_checkpoint_path = config.get_checkpoint_path("gnn_model.pt")
        history_path = config.get_checkpoint_path("gnn_training_history.json")
        train_data_path = config.get_checkpoint_path("pyg_train_data.pkl")
        val_data_path = config.get_checkpoint_path("pyg_val_data.pkl")

        # Build converter
        converter = SubgraphConverter(
            entity_embeddings=retriever.entity_embeddings,
            relation_embeddings=retriever.relation_embeddings,
            text_embedder=retriever.text_embedder,
            config=config,
        )

        # Build trainer
        trainer = GNNTrainer(config, encoder_type, pooling_type)

        # Check if checkpoint exists
        if checkpoint_exists(model_checkpoint_path):
            print(f"Loading GNN model from checkpoint...")
            trainer.load_checkpoint(model_checkpoint_path)
            print("✓ GNN model loaded")
        else:
            print("No checkpoint found. Training from scratch...\n")

            # Prepare training data
            print("Preparing training data...")
            if checkpoint_exists(train_data_path) and checkpoint_exists(val_data_path):
                print("  Loading cached PyG data...")
                train_pyg_data = load_checkpoint(train_data_path, format="pickle")
                val_pyg_data = load_checkpoint(val_data_path, format="pickle")
            else:
                print("  Converting dataset to PyG format (this may take a while)...")
                # Suppress per-example PCST output to avoid flooding cell output
                old_verbose = retriever.pcst_solver.verbose
                retriever.pcst_solver.verbose = False
                try:
                    train_pyg_data = cls._prepare_training_data(
                        train_data, retriever, converter, config
                    )
                    val_pyg_data = cls._prepare_training_data(
                        val_data, retriever, converter, config
                    )
                finally:
                    retriever.pcst_solver.verbose = old_verbose

                print("  Saving converted data to checkpoints...")
                save_checkpoint(train_pyg_data, train_data_path, format="pickle")
                save_checkpoint(val_pyg_data, val_data_path, format="pickle")

            print(f"✓ Training data: {len(train_pyg_data)} examples")
            print(f"✓ Validation data: {len(val_pyg_data)} examples\n")

            # After PyG data is saved, free the sentence transformer from GPU.
            # It's only needed for query embedding (already baked into PyG data)
            # and won't be used again until inference. Moving to CPU frees ~90 MB
            # of GPU VRAM before the GNN model is initialized for training.
            try:
                retriever.text_embedder.model.to("cpu")
                retriever.text_embedder.device = "cpu"
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create data loaders
            train_loader = DataLoader(
                train_pyg_data,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # Colab doesn't support multiprocessing well
            )
            val_loader = DataLoader(
                val_pyg_data,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )

            # Train
            history = trainer.train(train_loader, val_loader)

            # Save checkpoint
            print(f"\nSaving model checkpoint...")
            trainer.save_checkpoint(model_checkpoint_path)
            save_checkpoint(history, history_path, format="json")
            print("✓ Checkpoint saved")

        print("\n" + "="*60)
        print("GNN Model Ready")
        print("="*60 + "\n")

        return cls(config, trainer, converter)

    @classmethod
    def _prepare_training_data(
        cls,
        dataset: Dataset,
        retriever: Retriever,
        converter: SubgraphConverter,
        config: BenchmarkConfig,
    ) -> List:
        """
        Prepare PyG Data objects for training.

        For each example:
        1. Retrieve subgraph
        2. Convert to PyG Data with labels

        Args:
            dataset: HuggingFace Dataset
            retriever: Retriever instance
            converter: SubgraphConverter instance
            config: BenchmarkConfig

        Returns:
            List of PyG Data objects
        """
        from tqdm.auto import tqdm

        pyg_data_list = []
        skipped_no_answer = 0

        # Pre-compute all question embeddings in a single batched call.
        # This avoids 2x per-example GPU inference (retrieve + convert both
        # call embed_texts individually, fragmenting GPU memory over thousands
        # of single-sample forward passes).
        print("  Pre-computing query embeddings in batch...")
        all_questions = [ex["question"] for ex in dataset]
        query_embeddings = retriever.text_embedder.embed_texts(
            all_questions, batch_size=64, show_progress=True
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for idx, example in enumerate(tqdm(dataset, desc="Converting to PyG")):
            question = example["question"]
            answer_entities = example.get("a_entity", [])
            if isinstance(answer_entities, str):
                answer_entities = [answer_entities]
            q_entities = example.get("q_entity", [])
            if isinstance(q_entities, str):
                q_entities = [q_entities]

            precomputed_qe = query_embeddings[idx]

            # Pre-initialize so the finally block can safely del it even if
            # retrieve() raises before the assignment.
            retrieved = None
            try:
                retrieved = retriever.retrieve(question, q_entity=q_entities)

                # Convert to PyG Data, reusing the pre-computed query embedding
                data = converter.convert(
                    retrieved.subgraph,
                    question,
                    answer_entities=answer_entities,
                    precomputed_query_embedding=precomputed_qe,
                )

                # Skip examples where no answer node landed in the subgraph.
                # All-zero labels teach the GNN that no node matters, which
                # actively suppresses recall and poisons attention scores.
                if data.y is not None and data.y.sum().item() == 0:
                    skipped_no_answer += 1
                else:
                    pyg_data_list.append(data)

            except Exception as e:
                print(f"Warning: Failed to process example '{question}': {e}")
            finally:
                # Explicitly release the retrieved subgraph. NetworkX DiGraph
                # objects have cyclic dict references that refcounting alone
                # cannot free — only the cyclic GC can. Deleting here ensures
                # the reference is dropped immediately so the next gc.collect()
                # can actually reclaim the memory.
                del retrieved

            # Periodic GC to release accumulated NetworkX subgraph objects.
            # Every 20 examples (not 200) to keep the heap spike small.
            if idx % 20 == 0 and idx > 0:
                gc.collect()
            if idx % 100 == 0 and idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if skipped_no_answer > 0:
            total = skipped_no_answer + len(pyg_data_list)
            print(f"  Skipped {skipped_no_answer}/{total} examples "
                  f"with no answer node in subgraph "
                  f"({skipped_no_answer/total:.1%} miss rate)")

        return pyg_data_list

    def encode(
        self, retrieved_subgraph: RetrievedSubgraph, question: str
    ) -> GNNOutput:
        """
        Encode a retrieved subgraph with query conditioning.

        Args:
            retrieved_subgraph: RetrievedSubgraph from Phase 2 retriever
            question: Natural language question text

        Returns:
            GNNOutput with node embeddings, attention scores, and graph embedding
        """
        # Convert to PyG Data
        data = self.converter.convert(
            retrieved_subgraph.subgraph, question, answer_entities=None
        )
        data = data.to(self.device)

        # Forward pass
        with torch.no_grad():
            # Encoder (single graph: query_embedding is [1, 384], broadcasts via batch=None)
            node_embeddings, attention_weights = self.trainer.encoder(
                data.x, data.edge_index, data.edge_attr, data.query_embedding
            )

            # Pooling
            # Create batch tensor (single graph)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            graph_embedding, pooling_attention = self.trainer.pooling(
                node_embeddings, batch
            )

            # Use encoder attention weights (more informative than pooling attention)
            attention_scores_tensor = attention_weights

        # Convert attention scores to dict
        node_names = data.node_names
        attention_scores = {}
        for i, node_name in enumerate(node_names):
            attention_scores[node_name] = float(attention_scores_tensor[i].item())

        return GNNOutput(
            node_embeddings=node_embeddings.cpu(),
            attention_scores=attention_scores,
            graph_embedding=graph_embedding.squeeze(0).cpu(),
        )

    def encode_batch(
        self, retrieved_subgraphs: List[RetrievedSubgraph], questions: List[str]
    ) -> List[GNNOutput]:
        """
        Encode multiple subgraphs in a single batched forward pass.

        Args:
            retrieved_subgraphs: List of RetrievedSubgraph objects
            questions: List of question texts

        Returns:
            List of GNNOutput objects
        """
        from torch_geometric.data import Batch

        # Convert all subgraphs to PyG Data
        data_list = []
        for subgraph, question in zip(retrieved_subgraphs, questions):
            data = self.converter.convert(
                subgraph.subgraph, question, answer_entities=None
            )
            data_list.append(data)

        # Batch and forward pass
        batched = Batch.from_data_list(data_list).to(self.device)

        with torch.no_grad():
            node_embeddings, attention_weights = self.trainer.encoder(
                batched.x, batched.edge_index, batched.edge_attr,
                batched.query_embedding, batched.batch,
            )
            graph_embedding, _ = self.trainer.pooling(
                node_embeddings, batched.batch
            )

        # Split results back per graph
        outputs = []
        for i, data in enumerate(data_list):
            mask = (batched.batch == i)
            node_emb_i = node_embeddings[mask].cpu()
            attn_i = attention_weights[mask]
            graph_emb_i = graph_embedding[i].cpu()

            attention_scores = {}
            for j, name in enumerate(data.node_names):
                attention_scores[name] = float(attn_i[j].item())

            outputs.append(GNNOutput(
                node_embeddings=node_emb_i,
                attention_scores=attention_scores,
                graph_embedding=graph_emb_i,
            ))

        return outputs

    def get_top_attention_nodes(
        self, gnn_output: GNNOutput, top_k: int = 10
    ) -> List[tuple]:
        """
        Get top-K nodes by attention score.

        Args:
            gnn_output: GNNOutput from encode()
            top_k: Number of top nodes to return

        Returns:
            List of (node_name, attention_score) tuples, sorted by score descending
        """
        sorted_nodes = sorted(
            gnn_output.attention_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_nodes[:top_k]
