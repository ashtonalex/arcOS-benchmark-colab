
import numpy as np
import networkx as nx
import pcst_fast

def test_pcst_reproduction():
    print(" reproducing PCST failure scenario...")
    
    # 1. Create a graph with 308 nodes (0 to 307)
    num_nodes = 308
    root_idx = 0
    
    # Edges: Let's make a few components
    # Component 1 (Root component): Nodes 0-100 linear chain + some random edges
    edges = []
    for i in range(100):
        edges.append([i, i+1])
    # Add some random edges in root component
    for i in range(50):
        u = np.random.randint(0, 101)
        v = np.random.randint(0, 101)
        if u != v:
            edges.append([u, v])
            
    # Component 2 (Disconnected): Nodes 102-307
    # Let's say there are high prizes here
    for i in range(102, 307):
        edges.append([i, i+1])
        
    edges = np.array(edges, dtype=np.int64)
    
    # Prizes:
    # Root has 1.0
    prizes = np.zeros(num_nodes, dtype=np.float64)
    prizes[root_idx] = 1.0
    
    # "High prize-to-cost ratios (up to 66x)" -> 66 * 0.015 = 0.99
    # Add some high prizes in disconnected component
    prizes[200] = 0.9  # Node 200 is in Component 2
    prizes[205] = 0.8
    prizes[50] = 0.1 # In root component
    
    # Costs
    cost = 0.015
    costs = np.full(len(edges), cost, dtype=np.float64)
    
    print(f"Num nodes: {num_nodes}")
    print(f"Num edges: {len(edges)}")
    print(f"Root: {root_idx}")
    print(f"Max prize: {np.max(prizes)}")
    
    # Run pcst_fast
    # run(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)
    try:
        result_nodes, result_edges = pcst_fast.pcst_fast(
            edges, prizes, costs,
            root_idx, 1, "gw", 0
        )
        print(f"\nRaw result_nodes type: {type(result_nodes)}")
        print(f"Raw result_nodes shape: {result_nodes.shape}")
        print(f"Raw result_nodes content (first 20): {result_nodes[:20]}")
        
        # Check if it returns labels or indices
        if len(result_nodes) == num_nodes:
            print("Format: LABELS (length equals num_nodes)")
            # Standard label logic
            root_label = result_nodes[root_idx]
            print(f"Root label: {root_label}")
            if root_label < 0:
                 print("Root label is negative! (Pruned?)")
            else:
                 selected = np.where(result_nodes == root_label)[0]
                 print(f"Selected count (matching root label): {len(selected)}")
        else:
            print(f"Format: INDICES (length {len(result_nodes)} != {num_nodes})")
            print(f"Indices: {result_nodes}")
            
    except Exception as e:
        print(f"Error running pcst_fast: {e}")

if __name__ == "__main__":
    test_pcst_reproduction()
