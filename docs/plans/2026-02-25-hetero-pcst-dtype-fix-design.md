# HeteroPCST dtype Fix + Diagnostic Logging Design

**Date:** 2026-02-25
**Status:** Approved
**Scope:** `src/retrieval/hetero_pcst.py` only

## Problem

`HeteroPCST` returns 1 node regardless of graph size. The symptom is identical to a
previously documented bug in `PCSTSolver` (the Freebase pipeline), where passing
`np.int64` edges to `pcst_fast` caused "169 items collapsing to 1 node" due to a C++
type mismatch. That solver was fixed by switching to `np.int32`. `HeteroPCST` was
written after that fix and did not inherit it.

## Root Cause

`HeteroPCST._flatten_edges()` returns `np.array(edges, dtype=np.int64)`.
`HeteroPCST._run_pcst()` calls `pcst_fast.pcst_fast(edges.astype(np.int64), ...)`.

`pcst_fast` on the Colab build expects `int32` edge indices. When it receives `int64`,
the C++ layer reads garbage values, the effective graph has no valid edges, and PCST
returns only the single highest-prize node (optimal solution for an edgeless graph).

## Hypotheses Ruled Out

| Hypothesis | Assessment |
|------------|-----------|
| Index mismatch (FAISS → prize array → edge_index) | Low risk — all use 0-indexed `data["object"].x` positions |
| Prize/cost ratio imbalance | Secondary concern — with real embeddings, prizes (0.3–0.7) dwarf edge costs |
| Static costs | By design, not a bug |
| Disconnected graph | Low risk — person hub (id=0) appears every frame, bridging all tracks |
| Polluted embeddings | Low risk — cell 14 passes `TextEmbedder` to `SceneGraphBuilder` |

## Fix

### 1. `_flatten_edges()` — change edge array dtype

```python
# Before
np.array(edges, dtype=np.int64)

# After
np.array(edges, dtype=np.int32)
```

### 2. `_run_pcst()` — use int32 consistently

```python
# Before
pcst_fast.pcst_fast(edges.astype(np.int64), ...)

# After
pcst_fast.pcst_fast(edges.astype(np.int32), ...)
```

### 3. Diagnostic logging

Add `verbose: bool = False` to `HeteroPCST.__init__`. When `True`, `extract()` prints:

- Input: `num_nodes`, edge count per type, total edges
- Prizes: count of prized nodes, min/max/mean prize value
- PCST raw output: vertex count before budget trim
- Final output: `selected_nodes` count

No interface changes to callers (`VideoRetriever`, notebook cells).

## Files Changed

| File | Change |
|------|--------|
| `src/retrieval/hetero_pcst.py` | dtype fix (2 lines) + verbose diagnostic block |

## Success Criteria

- Cell 21 diagnostic shows `Subgraph nodes > 1`
- Diagnostic log confirms prizes are in range and edges > 0 passed to PCST
- If still 1 node after fix, diagnostic reveals secondary cause (follow-up: Approach B/C)
