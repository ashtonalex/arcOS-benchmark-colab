# HeteroPCST dtype Fix + Diagnostic Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `HeteroPCST` returning 1 node on Colab by correcting the `pcst_fast` edge
array dtype from `int64` → `int32`, and add verbose diagnostic logging so failures are
visible in notebook output.

**Architecture:** Two-line dtype fix in `_flatten_edges` and `_run_pcst`. Verbose flag
added to `__init__` — off by default, enabling it in cell 18 of the notebook for
on-Colab confirmation. No caller interface changes.

**Tech Stack:** `pcst_fast`, `numpy`, `torch_geometric.data.HeteroData`

**Design doc:** `docs/plans/2026-02-25-hetero-pcst-dtype-fix-design.md`

---

## Context for the implementer

`pcst_fast` is a C++ Prize-Collecting Steiner Tree solver wrapped in Python. On the
Colab Linux build it expects edge indices as `int32`. Passing `int64` causes the C++
layer to read garbage, making the graph appear edgeless, so PCST returns just the
single highest-prize node. The identical bug was documented and fixed in
`PCSTSolver` (Freebase pipeline) — see `src/retrieval/pcst_solver.py` lines 447–453.

Local Windows `pcst_fast` accepts `int64`, so existing tests pass locally but the bug
manifests only on Colab. The fix must be applied regardless.

**Only file to touch:** `src/retrieval/hetero_pcst.py`

---

## Task 1: Add verbose diagnostic logging

**Files:**
- Modify: `src/retrieval/hetero_pcst.py`
- Test: `tests/test_hetero_pcst.py`

### Step 1: Write the failing test

Add this test to `tests/test_hetero_pcst.py`:

```python
def test_hetero_pcst_verbose_logs_to_stdout(capsys):
    """verbose=True prints PCST diagnostics without crashing."""
    config = BenchmarkConfig(pcst_budget=10)
    solver = HeteroPCST(config, verbose=True)
    data = make_chain_graph(10)
    prizes = {0: 0.8, 5: 0.6}
    solver.extract(data, prizes)
    captured = capsys.readouterr()
    assert "nodes" in captured.out
    assert "edges" in captured.out
    assert "prizes" in captured.out
    assert "PCST output" in captured.out
```

### Step 2: Run test to confirm it fails

```bash
python -m pytest tests/test_hetero_pcst.py::test_hetero_pcst_verbose_logs_to_stdout -v
```

Expected: `FAILED` — `HeteroPCST.__init__` does not accept `verbose`.

### Step 3: Add `verbose` to `__init__` and diagnostic block to `extract`

In `src/retrieval/hetero_pcst.py`, replace the `__init__` and `extract` methods:

```python
def __init__(self, config: BenchmarkConfig, verbose: bool = False):
    self.budget = config.pcst_budget
    self.cost = config.pcst_cost
    self.pruning = config.pcst_pruning
    self.temporal_cost_weight = config.pcst_temporal_cost_weight
    self.verbose = verbose

def extract(self, data: HeteroData, prizes: Dict[int, float], root: Optional[int] = None) -> HeteroData:
    """Extract a subgraph from HeteroData using PCST.

    Args:
        data: Full video scene graph as HeteroData.
        prizes: Dict mapping node indices to prize values.
        root: Optional root node for PCST.
    Returns:
        HeteroData subgraph with `selected_nodes` attribute containing original node indices.
    """
    num_nodes = data["object"].num_nodes
    prize_array = np.zeros(num_nodes, dtype=np.float64)
    for node_idx, prize in prizes.items():
        if 0 <= node_idx < num_nodes:
            prize_array[node_idx] = prize

    edges, costs, edge_type_map = self._flatten_edges(data, num_nodes)

    if self.verbose:
        n_prized = int(np.count_nonzero(prize_array))
        prized_vals = prize_array[prize_array > 0]
        edge_counts = {et[1]: data[et].edge_index.shape[1] for et in data.edge_types}
        print(f"  [PCST] num_nodes={num_nodes}, edges={len(edges)} "
              f"({edge_counts}), prizes={n_prized} "
              f"(min={prized_vals.min():.3f}, max={prized_vals.max():.3f}, "
              f"mean={prized_vals.mean():.3f})" if n_prized else
              f"  [PCST] num_nodes={num_nodes}, edges={len(edges)} "
              f"({edge_counts}), prizes=0")

    if len(edges) == 0:
        return self._select_top_prized(data, prizes)

    try:
        selected_nodes = self._run_pcst(num_nodes, edges, costs, prize_array, root)
    except Exception:
        selected_nodes = self._bfs_fallback(edges, prizes, num_nodes)

    if self.verbose:
        print(f"  [PCST] raw output: {len(selected_nodes)} nodes")

    if len(selected_nodes) > self.budget:
        scored = [(n, prize_array[n]) for n in selected_nodes]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [n for n, _ in scored[:self.budget]]

    selected_nodes = sorted(set(selected_nodes))

    if self.verbose:
        print(f"  [PCST] final selected: {len(selected_nodes)} nodes")

    return self._slice_heterodata(data, selected_nodes)
```

### Step 4: Run test to confirm it passes

```bash
python -m pytest tests/test_hetero_pcst.py::test_hetero_pcst_verbose_logs_to_stdout -v
```

Expected: `PASSED`

### Step 5: Confirm existing tests still pass

```bash
python -m pytest tests/test_hetero_pcst.py -v
```

Expected: all 6 tests `PASSED`

### Step 6: Commit

```bash
git add src/retrieval/hetero_pcst.py tests/test_hetero_pcst.py
git commit -m "feat: add verbose diagnostic logging to HeteroPCST"
```

---

## Task 2: Fix int64 → int32 dtype (the primary bug)

**Files:**
- Modify: `src/retrieval/hetero_pcst.py:74,84`
- Test: `tests/test_hetero_pcst.py`

### Step 1: Write the failing test

Add this test to `tests/test_hetero_pcst.py`:

```python
def test_hetero_pcst_edges_are_int32():
    """Edge array passed to pcst_fast must be int32 (Colab pcst_fast build requirement).

    This test patches pcst_fast.pcst_fast to capture the actual dtype of the edges
    array passed in, so we can assert int32 even on Windows where int64 also works.
    """
    import unittest.mock as mock
    config = BenchmarkConfig(pcst_budget=20)
    solver = HeteroPCST(config)
    data = make_chain_graph(20)
    prizes = {0: 0.8, 10: 0.6}

    captured_dtype = {}

    def fake_pcst(edges, prizes, costs, root, n_clusters, pruning, verbosity):
        captured_dtype["edges"] = edges.dtype
        # Return a valid result: first two nodes
        import numpy as np
        return np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64)

    import src.retrieval.hetero_pcst as pcst_module
    with mock.patch.object(pcst_module, "pcst_fast") as mock_lib:
        mock_lib.pcst_fast.side_effect = fake_pcst
        solver.extract(data, prizes)

    assert captured_dtype["edges"] == np.int32, (
        f"Expected int32 edges for pcst_fast compatibility, got {captured_dtype['edges']}"
    )
```

### Step 2: Run test to confirm it fails

```bash
python -m pytest tests/test_hetero_pcst.py::test_hetero_pcst_edges_are_int32 -v
```

Expected: `FAILED` — `AssertionError: Expected int32 edges for pcst_fast compatibility, got int64`

### Step 3: Fix the dtype in `_flatten_edges` and `_run_pcst`

In `src/retrieval/hetero_pcst.py`, make two changes:

**Line 74** — `_flatten_edges` return statement:
```python
# Before
np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64),

# After
np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32),
```

**Line 84** — `_run_pcst` call to `pcst_fast.pcst_fast`:
```python
# Before
edges.astype(np.int64), prizes, costs, root_idx, 1, self.pruning, 0,

# After
edges.astype(np.int32), prizes, costs, root_idx, 1, self.pruning, 0,
```

### Step 4: Run the new test to confirm it passes

```bash
python -m pytest tests/test_hetero_pcst.py::test_hetero_pcst_edges_are_int32 -v
```

Expected: `PASSED`

### Step 5: Run the full test suite to confirm nothing regressed

```bash
python -m pytest tests/test_hetero_pcst.py -v
```

Expected: all 7 tests `PASSED`

### Step 6: Commit

```bash
git add src/retrieval/hetero_pcst.py tests/test_hetero_pcst.py
git commit -m "fix: use int32 edge dtype for pcst_fast compatibility (Colab build)"
```

---

## Task 3: Enable verbose in notebook cell 18

**Files:**
- Modify: `notebooks/arcOS_benchmark.ipynb` (cell 18)

Cell 18 already has the PCST configuration override. Add one line to enable verbose on
the retriever's PCST instance so the diagnostics appear during cell 20 and 21.

### Step 1: Add verbose flag to cell 18

Find this block in cell 18:
```python
config.pcst_budget = 70
config.pcst_cost = 0.015
config.pcst_pruning = 'none'
config.top_k_seeds = 10
config.pcst_temporal_cost_weight = 0.5

config.__post_init__()  # re-validate
```

Append after `config.__post_init__()`:
```python
# Enable PCST diagnostics — remove after confirming fix works on Colab
pcst_verbose = True
```

Then in cell 19, after creating the retriever, patch verbose onto its PCST instance:
```python
embedder = TextEmbedder(config)
retriever = VideoRetriever(config, embedder=embedder)
retriever.pcst.verbose = pcst_verbose   # ← add this line
print(f'✓ VideoRetriever ready (top_k_seeds={config.top_k_seeds}, pcst_budget={config.pcst_budget})')
```

### Step 2: Commit

```bash
git add notebooks/arcOS_benchmark.ipynb
git commit -m "nb: enable PCST verbose diagnostics in cells 18-19 for Colab validation"
```

---

## Colab Validation

After running cells 18–21, the output should show:

```
[PCST] num_nodes=<N>, edges=<E> ({'spatial_rel': X, 'temporal': Y}), prizes=10 (min=0.2xx, max=0.7xx, mean=0.4xx)
[PCST] raw output: <M> nodes      ← should be >> 1 after the fix
[PCST] final selected: <K> nodes
```

And cell 21 should show:
```
Subgraph nodes: <K>   ← was 1 before, should now be 5-50
```

If `edges=0` appears → the graph has no edges (separate connectivity issue).
If `prizes=0` or `min=max=0.0` → embeddings are not being used (separate embedding issue).
If raw output is still 1 after edges>0 and prizes look reasonable → escalate to Approach B (lower temporal cost) or Approach C (local prizes).
