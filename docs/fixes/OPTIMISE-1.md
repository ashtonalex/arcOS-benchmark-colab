# PCST Retrieval Optimisations

Fixes and optimisations for the PCST subgraph extraction pipeline.
Addresses 0% hit rate caused by broken prize scaling, incorrect root selection,
and faulty labels-vs-indices detection in `pcst_fast`.

## Status: Ready for Implementation

**Affected files:**
- `src/retrieval/retriever.py` (prize logic, seed filtering)
- `src/retrieval/pcst_solver.py` (root selection, format detection)

**Depends on:** PCST solver cleanup (already applied — zero-base prizes, cost=0.3)

---

## Problem Statement

The retrieval pipeline produces a 0% answer hit rate. PCST consistently returns
1 node (the root only), triggering BFS fallback which fills to budget with
topology-blind expansion. Three root causes have been identified:

1. **Prize scale mismatch**: Prizes range 3-100, edge cost is 0.3. PCST never
   prunes anything — every node appears worth keeping. The algorithm is
   effectively disabled.

2. **Wrong root selection**: PCST roots at the highest-prize k-NN result, not
   the query's topic entity. For "what is Henry Clay known for", the root is
   "John F. Kennedy" (a noisy k-NN match), so PCST explores the wrong
   neighborhood.

3. **Labels-vs-indices misdetection**: `pcst_fast` returns cluster labels on
   some installations. The detection heuristic (`len == num_nodes`) fails when
   the graph is disconnected — labels for a 77-node component are mistaken for
   77 indices, mapping to 1 unique node.

---

## Fix 1: Prize Scale Alignment

### Current behaviour

```python
# retriever.py lines 113-125
prizes[q_entity]  = 100.0                          # fixed
prizes[knn_entity] = log1p(top_k - rank + 1) * 20  # 14 to 55
prizes[neighbor]   = 3.0                            # 1-hop
```

With `pcst_cost=0.3`, every node has prize >> cost. PCST objective
`max(Σ prizes - Σ costs)` is maximised by including all nodes.
No selectivity occurs.

### Fix

Use raw cosine similarity from FAISS as prizes. Scores are already in
[0, 1] (IndexFlatIP with L2-normalised vectors = cosine similarity).
q_entity gets 1.0 (perfect relevance).

```python
prizes = {}

# Topic entities: perfect relevance
for entity in q_entity_names:
    prizes[entity] = 1.0

# k-NN entities: raw cosine similarity, filtered by threshold
similarity_threshold = 0.4
for entity, score in top_k_results:
    if entity not in prizes and score >= similarity_threshold:
        prizes[entity] = float(score)
```

### Rationale

- Prizes and costs on the same scale (0-1 vs 0.3) gives PCST meaningful
  signal. A node with cosine sim 0.2 doesn't justify a 0.3 edge cost.
- Threshold filtering removes noisy k-NN matches ("Henry Clay Frick" for
  a query about "Henry Clay") before they become seeds.
- No 1-hop neighbor prizes. With zero-base prizes, PCST keeps relay nodes
  organically when they cheaply connect high-prize targets. Manual
  assignment at 3.0 was a workaround for the old broken scale.

### Success criteria

- Prize values in [0, 1] range
- `max(prize) / pcst_cost` ratio between 2x and 5x (not 300x)
- Seeds list shorter (noisy low-similarity matches filtered out)

---

## Fix 2: Root Selection — Prefer Topic Entity

### Current behaviour

```python
# pcst_solver.py _pcst_extract()
best_seed = max(valid_seeds_in_local, key=lambda s: prizes.get(s, 0.0))
root = node_to_idx[best_seed]
```

Root is whichever seed has the highest prize. With noisy k-NN results,
this can be a false positive (JFK instead of Henry Clay). PCST then
builds a tree rooted in the wrong part of the graph.

### Fix

Accept an optional `root_entities` parameter in `extract_subgraph`.
The retriever passes `q_entity` (the dataset's known topic entities).
PCST prefers these as root over k-NN seeds.

**pcst_solver.py:**

```python
def extract_subgraph(self, G, seed_nodes, prizes, root_entities=None):
    ...
    # Pass root_entities through to _pcst_extract
    subgraph = self._pcst_extract(local_graph, valid_seeds, prizes,
                                   root_entities=root_entities)

def _pcst_extract(self, G, seed_nodes, prizes, pruning_override=None,
                  root_entities=None):
    ...
    # Root selection: prefer topic entity, fall back to highest-prize seed
    root_node = None
    if root_entities:
        for entity in root_entities:
            if entity in node_to_idx:
                root_node = entity
                break

    if root_node is None:
        root_node = max(valid_seeds_in_local,
                        key=lambda s: prizes.get(s, 0.0))

    root = int(node_to_idx[root_node])
```

**retriever.py:**

```python
subgraph = self.pcst_solver.extract_subgraph(
    self.unified_graph,
    seed_entities,
    prizes,
    root_entities=list(q_entity_names) if q_entity_names else None
)
```

### Rationale

- The dataset provides `q_entity` (the question's topic entity) which is
  the correct anchor for subgraph extraction.
- k-NN search is useful for discovering related entities but should not
  override the known topic entity as root.
- Fallback to highest-prize seed preserves behaviour when `q_entity` is
  not in the graph.

### Success criteria

- Root is the topic entity when it exists in the local graph
- Subgraph is centred on the correct entity, not a k-NN false positive

---

## Fix 3: Labels-vs-Indices Detection

### Current behaviour

```python
# pcst_solver.py _pcst_extract()
if len(result_nodes) == num_nodes and num_nodes > self.budget:
    # Labels format
    ...
else:
    # Indices format
    selected = result_nodes
```

This assumes labels format always returns exactly `num_nodes` entries.
When the graph is disconnected, `pcst_fast` returns labels for only the
root's connected component (e.g. 77 labels for 77 reachable nodes out
of 300 total). Since `77 != 300`, the code treats them as indices.
Labels `[0, 0, 0, ...]` then map to `nodes[0]` repeated 77 times = 1
unique node.

### Fix

Detect format by checking for duplicate values. Indices are always
unique (each selected node appears once). Labels have many duplicates
(cluster membership IDs like 0, 0, 0, 1, 0, ...).

```python
result_nodes = np.asarray(result_nodes, dtype=np.int64)
n_unique = len(np.unique(result_nodes))

if n_unique < len(result_nodes):
    # Labels format: many duplicates, extract root's cluster
    root_label = result_nodes[root] if root < len(result_nodes) else 0
    selected = np.where(result_nodes == root_label)[0]
    print(f"  PCST returned labels format, "
          f"extracted {len(selected)} nodes in root cluster")
else:
    # Indices format: all unique values, use directly
    selected = result_nodes
    print(f"  PCST output: {len(selected)} nodes (indices format)")
```

### Rationale

- Indices are node IDs — inherently unique (you can't select the same
  node twice). Any duplicates mean labels format.
- This is invariant regardless of graph connectivity, component size,
  or pcst_fast version.
- No runtime probe or diagnostic function needed.

### Success criteria

- Labels format detected correctly for disconnected graphs
- No single-node PCST results from format misdetection
- Works for both pcst_fast return formats without configuration

---

## Implementation Order

1. **Fix 3** (labels detection) — unblocks PCST from returning 1 node
2. **Fix 2** (root selection) — ensures PCST explores the right neighborhood
3. **Fix 1** (prize scale) — gives PCST meaningful selectivity signal

Fixes 2 and 3 are in `pcst_solver.py`. Fix 1 is in `retriever.py`.
All three are independent and can be applied in any order, but the
above order reflects debugging priority (fix the crash, fix the root,
then tune the signal).

---

## Tuning Guide (Post-Fix)

After applying all fixes, the `pcst_cost` parameter controls selectivity:

| `pcst_cost` | Behaviour | When to use |
|-------------|-----------|-------------|
| 0.1 | Keeps most nodes (prize > 0.1 justifies a hop) | Under-retrieving, small subgraphs |
| 0.3 | Balanced (default) | Starting point |
| 0.5 | Aggressive pruning (only high-similarity nodes) | Over-retrieving, noisy subgraphs |

The `similarity_threshold` in Fix 1 controls seed quality:

| Threshold | Effect |
|-----------|--------|
| 0.3 | Permissive — more seeds, more coverage, more noise |
| 0.4 | Balanced (default) |
| 0.6 | Strict — fewer seeds, less noise, risk of missing relevant entities |

Monitor hit rate and subgraph size to tune. Target: hit rate > 60%,
subgraph size 15-50 nodes (not always hitting budget).
