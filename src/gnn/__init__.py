"""
GNN Encoder module for arcOS benchmark.

Implements GATv2-based Graph Neural Networks for learning query-relevant
node representations in knowledge graph subgraphs.
"""

from .data_utils import SubgraphConverter, GNNOutput
from .encoder import GATv2Encoder, GraphSAGEEncoder
from .pooling import AttentionPooling, MeanPooling, MaxPooling
from .trainer import GNNTrainer
from .model_wrapper import GNNModel

__all__ = [
    "SubgraphConverter",
    "GNNOutput",
    "GATv2Encoder",
    "GraphSAGEEncoder",
    "AttentionPooling",
    "MeanPooling",
    "MaxPooling",
    "GNNTrainer",
    "GNNModel",
]
