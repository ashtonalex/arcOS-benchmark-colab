"""
NetworkX graph construction from RoG-WebQSP triples.

Builds both unified graphs (all training examples) and per-example subgraphs.
"""

from collections import Counter
from typing import List, Tuple, Optional
import networkx as nx


class GraphBuilder:
    """
    Builder for NetworkX graphs from triple lists.

    Each triple is [subject, relation, object] where:
    - subject: Entity string (e.g., "m.0d05w3")
    - relation: Freebase relation (e.g., "people.person.sibling_s")
    - object: Entity string (e.g., "m.02w_b5r")
    """

    def __init__(self, directed: bool = True):
        """
        Initialize the graph builder.

        Args:
            directed: If True, create directed graphs (default), else undirected
        """
        self.directed = directed
        print(f"✓ GraphBuilder initialized (directed={directed})")

    def build_from_triples(
        self,
        triples: List[List[str]],
        graph_id: Optional[str] = None
    ) -> nx.Graph:
        """
        Build a NetworkX graph from a list of triples.

        Args:
            triples: List of [subject, relation, object] triples
            graph_id: Optional identifier for the graph

        Returns:
            NetworkX Graph or DiGraph
        """
        G = nx.DiGraph() if self.directed else nx.Graph()

        for triple in triples:
            if len(triple) != 3:
                print(f"Warning: Skipping invalid triple: {triple}")
                continue

            subject, relation, obj = triple

            # Add nodes with entity names as attributes
            G.add_node(subject, entity_name=subject)
            G.add_node(obj, entity_name=obj)

            # Add edge with relation as attribute
            G.add_edge(subject, obj, relation=relation)

        if graph_id:
            G.graph["id"] = graph_id

        return G

    def build_unified_graph(
        self,
        dataset,
        max_examples: Optional[int] = None
    ) -> nx.Graph:
        """
        Build a unified graph from all examples in a dataset.

        This merges all triples from all examples into a single large graph,
        used for building the global entity/relation index in Phase 2.

        Args:
            dataset: HuggingFace Dataset object with 'graph' field
            max_examples: If set, only use first N examples (for testing)

        Returns:
            NetworkX Graph or DiGraph containing all triples
        """
        print("Building unified graph from dataset...")

        G = nx.DiGraph() if self.directed else nx.Graph()
        total_triples = 0

        num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

        for i, example in enumerate(dataset):
            if max_examples and i >= max_examples:
                break

            triples = example['graph']
            for triple in triples:
                if len(triple) != 3:
                    continue

                subject, relation, obj = triple

                # Add nodes
                G.add_node(subject, entity_name=subject)
                G.add_node(obj, entity_name=obj)

                # Add edge (will merge if duplicate)
                G.add_edge(subject, obj, relation=relation)

                total_triples += 1

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{num_examples} examples...")

        print(f"✓ Unified graph built from {num_examples} examples")
        print(f"  Total triples processed: {total_triples}")
        print(f"  Unique nodes: {G.number_of_nodes()}")
        print(f"  Unique edges: {G.number_of_edges()}")

        return G

    def compute_graph_statistics(self, G: nx.Graph) -> dict:
        """
        Compute comprehensive graph statistics.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "is_directed": G.is_directed(),
            "density": nx.density(G),
        }

        # Compute connectivity for directed graphs
        if G.is_directed():
            stats["is_weakly_connected"] = nx.is_weakly_connected(G)
            if not stats["is_weakly_connected"]:
                stats["num_weakly_connected_components"] = nx.number_weakly_connected_components(G)
        else:
            stats["is_connected"] = nx.is_connected(G)
            if not stats["is_connected"]:
                stats["num_connected_components"] = nx.number_connected_components(G)

        # Relation distribution
        relations = []
        for u, v, data in G.edges(data=True):
            relations.append(data.get("relation", "unknown"))

        relation_counts = Counter(relations)
        stats["num_unique_relations"] = len(relation_counts)
        stats["top_relations"] = relation_counts.most_common(10)

        # Degree statistics
        if G.is_directed():
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
            stats["avg_in_degree"] = sum(in_degrees) / len(in_degrees) if in_degrees else 0
            stats["avg_out_degree"] = sum(out_degrees) / len(out_degrees) if out_degrees else 0
        else:
            degrees = [d for n, d in G.degree()]
            stats["avg_degree"] = sum(degrees) / len(degrees) if degrees else 0

        return stats

    def print_graph_info(self, G: nx.Graph, name: str = "Graph"):
        """
        Print human-readable graph information.

        Args:
            G: NetworkX graph
            name: Name/label for the graph
        """
        print("\n" + "=" * 60)
        print(f"{name} Information")
        print("=" * 60)

        stats = self.compute_graph_statistics(G)

        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Directed: {stats['is_directed']}")
        print(f"Density: {stats['density']:.6f}")

        if "is_weakly_connected" in stats:
            print(f"Weakly connected: {stats['is_weakly_connected']}")
            if not stats['is_weakly_connected']:
                print(f"  Weakly connected components: {stats['num_weakly_connected_components']}")
        elif "is_connected" in stats:
            print(f"Connected: {stats['is_connected']}")
            if not stats['is_connected']:
                print(f"  Connected components: {stats['num_connected_components']}")

        print(f"\nRelation Statistics:")
        print(f"Unique relations: {stats['num_unique_relations']}")
        print(f"Top 10 relations:")
        for relation, count in stats['top_relations']:
            print(f"  - {relation}: {count}")

        print(f"\nDegree Statistics:")
        if "avg_in_degree" in stats:
            print(f"Average in-degree: {stats['avg_in_degree']:.2f}")
            print(f"Average out-degree: {stats['avg_out_degree']:.2f}")
        else:
            print(f"Average degree: {stats['avg_degree']:.2f}")

        print("=" * 60)

    def validate_graph_size(
        self,
        G: nx.Graph,
        min_nodes: int = 10000,
        min_edges: int = 30000
    ) -> bool:
        """
        Validate that graph meets minimum size requirements.

        Args:
            G: NetworkX graph to validate
            min_nodes: Minimum required nodes
            min_edges: Minimum required edges

        Returns:
            True if graph meets requirements, False otherwise
        """
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        nodes_ok = num_nodes >= min_nodes
        edges_ok = num_edges >= min_edges

        print("\n" + "=" * 60)
        print("Graph Size Validation")
        print("=" * 60)
        print(f"Nodes: {num_nodes} {'✓' if nodes_ok else f'✗ (expected >={min_nodes})'}")
        print(f"Edges: {num_edges} {'✓' if edges_ok else f'✗ (expected >={min_edges})'}")

        valid = nodes_ok and edges_ok
        if valid:
            print("\n✓ Graph meets size requirements")
        else:
            print("\n✗ Graph does not meet size requirements")

        print("=" * 60)
        return valid
