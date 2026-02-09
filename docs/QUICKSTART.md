# arcOS Benchmark - Quick Start Guide

## TL;DR

1. Upload `src/` folder and `notebooks/arcOS_benchmark.ipynb` to Google Colab
2. Set runtime to GPU (T4 or better)
3. Run cells 1-8 sequentially
4. Verify Phase 1 passes in Cell 8

**Time:** ~4 minutes (first run), ~1 minute (cached)

---

## Detailed Setup

### 1. Prepare Files for Upload

From your local `arcOS-benchmark-colab` directory, you need:

```
arcOS-benchmark-colab/
├── src/                          ← Upload this folder
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seeds.py
│   │   └── checkpoints.py
│   └── data/
│       ├── __init__.py
│       ├── dataset_loader.py
│       └── graph_builder.py
└── notebooks/
    └── arcOS_benchmark.ipynb    ← Upload this notebook
```

### 2. Upload to Colab

**Option A: Via Colab Interface**

1. Open https://colab.research.google.com/
2. File → Upload notebook → Select `notebooks/arcOS_benchmark.ipynb`
3. In the Colab file browser (left sidebar), click "Upload" icon
4. Upload the entire `src/` folder to `/content/arcOS-benchmark-colab/src/`

**Option B: Via Google Drive**

1. Upload `src/` folder to Google Drive: `MyDrive/arcOS_benchmark_code/src/`
2. Upload notebook to Drive or open directly in Colab
3. In Cell 2, adjust `repo_root` path to your Drive location

### 3. Configure GPU Runtime

1. Click "Runtime" in top menu
2. Select "Change runtime type"
3. Hardware accelerator: **GPU** (T4 recommended)
4. Click "Save"
5. Runtime will restart with GPU enabled

### 4. Run the Notebook

Execute cells **in order** (Shift+Enter or click play button):

#### Cell 1: Environment Setup (~2 min)
- Installs uv package manager
- Installs PyTorch, datasets, networkx
- Verifies GPU availability

**Expected output:**
```
GPU available: True ✓
GPU name: Tesla T4
GPU memory: 15.00 GB
```

#### Cell 2: Import Modules (instant)
- Adds `src/` to Python path
- Imports all benchmark modules

**Expected output:**
```
✓ Added to Python path: /content/arcOS-benchmark-colab
✓ All imports successful
```

#### Cell 3: Configuration (instant)
- Creates `BenchmarkConfig` with defaults
- Prints hyperparameters

**Expected output:**
```
arcOS Benchmark Configuration
Seed: 42 (deterministic=True)
Dataset: rmanluo/RoG-webqsp
...
```

#### Cell 4: Seed Initialization (instant)
- Sets random seeds for reproducibility

**Expected output:**
```
✓ Random seeds set to 42 (deterministic=True)
```

#### Cell 5: Google Drive Setup (~10 sec)
- Mounts Google Drive (requires authorization)
- Creates checkpoint/results directories

**Expected output:**
```
✓ Google Drive mounted at /content/drive
✓ Checkpoint directory: /content/drive/MyDrive/arcOS_benchmark/checkpoints
✓ Results directory: /content/drive/MyDrive/arcOS_benchmark/results
```

**Note:** First time will prompt for authorization:
1. Click the link
2. Sign in to Google account
3. Copy authorization code
4. Paste and press Enter

#### Cell 6: Dataset Loading (~1-2 min first time, ~5 sec cached)
- Downloads RoG-WebQSP from HuggingFace
- Saves to Drive checkpoint
- Validates schema and split counts

**Expected output:**
```
Loading dataset: rmanluo/RoG-webqsp
✓ Loaded dataset with splits: ['train', 'validation', 'test']
  - train: 2830 examples
  - validation: 246 examples
  - test: 1630 examples

Dataset Split Validation
Train: 2830 ✓
Validation: 246 ✓
Test: 1630 ✓
✓ All splits have expected sizes
```

#### Cell 7: Graph Construction (~1 min first time, ~5 sec cached)
- Builds unified graph from training triples
- Saves to Drive checkpoint
- Computes graph statistics

**Expected output:**
```
Building unified graph from dataset...
✓ Unified graph built from 2830 examples
  Unique nodes: ~15000-25000
  Unique edges: ~40000-80000

Graph Size Validation
Nodes: 15000+ ✓
Edges: 40000+ ✓
✓ Graph meets size requirements
```

#### Cell 8: Phase 1 Validation (instant)
- Automated validation of all success criteria

**Expected output:**
```
Phase 1 Success Criteria Validation
  ✓ GPU Available
  ✓ All Imports Successful
  ✓ Dataset Splits Valid
  ✓ Unified Graph Size Valid
  ✓ Checkpoint Round-Trip

✓ PHASE 1 COMPLETE - All criteria passed!

Ready to proceed to Phase 2: Retrieval Pipeline
```

---

## Verification Checklist

After running all 8 cells, verify:

- [ ] Cell 1: GPU shows "Tesla T4" (or better)
- [ ] Cell 2: All imports successful
- [ ] Cell 5: Google Drive mounted
- [ ] Cell 6: Dataset has 4,706 total examples (2830+246+1630)
- [ ] Cell 7: Unified graph has >10K nodes, >30K edges
- [ ] Cell 8: All 5 validation checks pass (✓)

If all checkmarks pass: **Phase 1 is complete!**

---

## Troubleshooting

### Problem: "GPU available: False"

**Cause:** CPU runtime selected

**Fix:**
1. Runtime → Change runtime type
2. Hardware accelerator: GPU
3. Save
4. Re-run Cell 1

---

### Problem: "ImportError: No module named 'src'"

**Cause:** `src/` folder not uploaded to correct location

**Fix:**
1. Check Colab file browser (left sidebar)
2. Verify path: `/content/arcOS-benchmark-colab/src/config.py`
3. If missing, upload `src/` folder
4. Re-run Cell 2

**Alternative fix:** Adjust path in Cell 2:
```python
repo_root = Path("/content")  # Change this to your upload location
```

---

### Problem: "Drive mount failed"

**Cause:** Authorization not completed

**Fix:**
1. Run Cell 5 again
2. Click authorization link
3. Sign in to Google account
4. Copy code and paste in Colab
5. Press Enter

---

### Problem: "Dataset download timeout"

**Cause:** Slow internet or HuggingFace rate limit

**Fix:**
1. Wait 5 minutes
2. Re-run Cell 6
3. If persists, check HuggingFace status: https://status.huggingface.co/

---

### Problem: "Graph too small (< 10K nodes)"

**Cause:** Dataset incomplete or loading error

**Fix:**
1. Check Cell 6 output for errors
2. Verify dataset has 2,830 training examples
3. Delete checkpoint: `/content/drive/MyDrive/arcOS_benchmark/checkpoints/unified_graph.pkl`
4. Re-run Cell 7

---

### Problem: "Checkpoint not persisting across sessions"

**Cause:** Drive not mounted or insufficient storage

**Fix:**
1. Verify Drive mounted in Cell 5 (should see ✓)
2. Check Drive free space (need ~10GB)
3. Verify files saved to Drive:
   - Go to Google Drive
   - Navigate to `MyDrive/arcOS_benchmark/checkpoints/`
   - Should see: `dataset.pkl`, `unified_graph.pkl`

---

## Expected File Sizes

After successful run, Google Drive should contain:

```
MyDrive/arcOS_benchmark/
├── checkpoints/
│   ├── dataset.pkl            (~200-500 MB)
│   ├── unified_graph.pkl      (~100-300 MB)
│   ├── test_roundtrip.pkl     (~1 KB)
│   └── huggingface_cache/     (~500 MB)
└── results/
    └── (empty for Phase 1)
```

**Total storage:** ~1-1.5 GB

---

## Performance Benchmarks

### Cold Start (First Run)
- Cell 1 (setup): ~2 min
- Cell 2-4: <5 sec total
- Cell 5 (Drive mount): ~10 sec
- Cell 6 (dataset download): ~1-2 min
- Cell 7 (graph build): ~1 min
- Cell 8 (validation): <1 sec

**Total: ~4-5 minutes**

### Warm Start (With Checkpoints)
- Cell 1 (setup): ~2 min
- Cell 2-4: <5 sec total
- Cell 5 (Drive mount): ~10 sec
- Cell 6 (cached load): ~5 sec
- Cell 7 (cached load): ~5 sec
- Cell 8 (validation): <1 sec

**Total: ~2.5 minutes**

### Re-run (Runtime Already Active)
- Cell 1: Skip (already installed)
- Cell 2-4: <5 sec total
- Cell 5 (already mounted): <1 sec
- Cell 6 (cached): ~5 sec
- Cell 7 (cached): ~5 sec
- Cell 8: <1 sec

**Total: ~15 seconds**

---

## Next Steps After Phase 1

Once Cell 8 shows "PHASE 1 COMPLETE":

1. **Phase 2: Retrieval Pipeline**
   - Sentence-Transformers embeddings
   - FAISS similarity search
   - PCST subgraph extraction
   - See: `docs/ROADMAP.md` Phase 2

2. **Phase 3: GNN Encoder**
   - Graph Attention Network (GATv2)
   - Attention pooling
   - Entity/relation embeddings

3. **Phases 4-8:** Verbalization, LLM, E2E, Evaluation, Hardening

---

## Tips for Efficient Development

### Tip 1: Use Checkpoints
- Never delete checkpoints unless debugging
- Cold start: ~4 min → Warm start: ~2.5 min → Re-run: ~15 sec

### Tip 2: Monitor VRAM
```python
# Add to any cell to check GPU memory
import torch
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Tip 3: Incremental Development
- Run Cells 1-5 once per session
- Iterate on Cells 6-7 as needed
- Always verify with Cell 8 before moving to Phase 2

### Tip 4: Save Notebook Often
- File → Save (or Ctrl+S)
- Colab auto-saves, but manual saves ensure no data loss

### Tip 5: Use GPU Wisely
- Free Colab: ~12 hours GPU time per day
- Run overnight or during off-peak hours for long training (Phase 6+)

---

## Support

- **Documentation:** See `PHASE1_COMPLETE.md` for detailed implementation notes
- **Architecture:** See `docs/PRD.md` for system design
- **Roadmap:** See `docs/ROADMAP.md` for all 8 phases
- **Issues:** Check Cell 8 validation output for specific failures

---

**Last Updated:** 2026-02-09

**Phase:** 1 of 8 (Environment & Data Foundation)

**Status:** Ready for Colab execution
