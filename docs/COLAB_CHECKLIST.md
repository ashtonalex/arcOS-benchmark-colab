# Google Colab Upload & Execution Checklist

Use this checklist to ensure Phase 1 is properly deployed and validated on Google Colab.

---

## Pre-Upload Verification ✓

- [x] Run `python test_phase1_imports.py` locally
- [x] Verify all imports pass
- [x] Verify graph construction works
- [x] Read QUICKSTART.md for instructions

---

## Step 1: Prepare Files

- [ ] Locate local directory: `C:\Users\User\arcOS-benchmark-colab\`
- [ ] Verify these files exist:
  - [ ] `src/config.py`
  - [ ] `src/utils/seeds.py`
  - [ ] `src/utils/checkpoints.py`
  - [ ] `src/data/dataset_loader.py`
  - [ ] `src/data/graph_builder.py`
  - [ ] All `__init__.py` files (3 total)
  - [ ] `notebooks/arcOS_benchmark.ipynb`

---

## Step 2: Upload to Colab

### Option A: Via Colab Interface

- [ ] Open https://colab.research.google.com/
- [ ] File → Upload notebook
- [ ] Select `notebooks/arcOS_benchmark.ipynb`
- [ ] Wait for upload to complete
- [ ] In left sidebar, click folder icon (Files)
- [ ] Navigate to `/content/`
- [ ] Right-click → New folder → Name: `arcOS-benchmark-colab`
- [ ] Right-click `arcOS-benchmark-colab` → Upload
- [ ] Upload entire `src/` folder
- [ ] Verify structure:
  ```
  /content/
  └── arcOS-benchmark-colab/
      └── src/
          ├── __init__.py
          ├── config.py
          ├── utils/
          │   ├── __init__.py
          │   ├── seeds.py
          │   └── checkpoints.py
          └── data/
              ├── __init__.py
              ├── dataset_loader.py
              └── graph_builder.py
  ```

### Option B: Via Git Clone (Faster)

- [ ] Create GitHub repo with all files
- [ ] In Colab, add code cell:
  ```python
  !git clone https://github.com/YOUR_USERNAME/arcOS-benchmark-colab.git /content/arcOS-benchmark-colab
  ```
- [ ] Run cell to clone
- [ ] Delete the cell (not needed in notebook)

---

## Step 3: Configure Runtime

- [ ] Click "Runtime" in top menu
- [ ] Select "Change runtime type"
- [ ] Set these options:
  - [ ] Runtime type: Python 3
  - [ ] Hardware accelerator: **GPU**
  - [ ] GPU type: T4 (or better if available)
- [ ] Click "Save"
- [ ] Runtime will restart (wait ~10 seconds)

---

## Step 4: Execute Notebook

Run cells in order by clicking play button or pressing Shift+Enter:

### Cell 1: Environment Setup (~2 min)
- [ ] Run cell
- [ ] Wait for uv installation
- [ ] Wait for PyTorch installation
- [ ] Verify output shows:
  - [ ] `GPU available: True ✓`
  - [ ] GPU name (e.g., "Tesla T4")
  - [ ] GPU memory (e.g., "15.00 GB")

**If GPU not available:**
- Stop → Repeat Step 3 (configure runtime)

---

### Cell 2: Import Modules (~1 sec)
- [ ] Run cell
- [ ] Verify output shows:
  - [ ] `✓ Added to Python path: /content/arcOS-benchmark-colab`
  - [ ] `✓ All imports successful`

**If imports fail:**
- Check Cell 2 `repo_root` path matches upload location
- Verify `src/` folder structure in file browser

---

### Cell 3: Configuration (~1 sec)
- [ ] Run cell
- [ ] Verify configuration summary displays
- [ ] Check these values:
  - [ ] Seed: 42
  - [ ] Dataset: rmanluo/RoG-webqsp
  - [ ] Drive root: /content/drive/MyDrive/arcOS_benchmark

---

### Cell 4: Seed Initialization (~1 sec)
- [ ] Run cell
- [ ] Verify output: `✓ Random seeds set to 42 (deterministic=True)`

---

### Cell 5: Google Drive Setup (~10 sec first time)
- [ ] Run cell
- [ ] If prompted for authorization:
  - [ ] Click authorization link
  - [ ] Sign in to Google account
  - [ ] Copy authorization code
  - [ ] Paste code in Colab input box
  - [ ] Press Enter
- [ ] Verify output shows:
  - [ ] `✓ Google Drive mounted at /content/drive`
  - [ ] `✓ Checkpoint directory: ...`
  - [ ] `✓ Results directory: ...`

**If mount fails:**
- Re-run cell
- Try different browser (Chrome recommended)
- Ensure Google account has Drive enabled

---

### Cell 6: Dataset Loading (~1-2 min first time, ~5 sec after)
- [ ] Run cell
- [ ] Wait for HuggingFace download (first time only)
- [ ] Verify output shows:
  - [ ] `✓ Loaded dataset with splits: ['train', 'validation', 'test']`
  - [ ] Train: 2830 examples
  - [ ] Validation: 246 examples
  - [ ] Test: 1630 examples
  - [ ] Dataset Split Validation: All ✓

**Expected timing:**
- First run: 1-2 minutes (downloads dataset)
- Subsequent runs: ~5 seconds (loads from checkpoint)

**If download fails:**
- Check internet connection
- Wait 5 minutes (rate limit)
- Re-run cell

---

### Cell 7: Graph Construction (~1 min first time, ~5 sec after)
- [ ] Run cell
- [ ] Wait for graph building (first time only)
- [ ] Verify output shows:
  - [ ] `✓ Unified graph built from 2830 examples`
  - [ ] Unique nodes: 10,000+ (typically 15K-25K)
  - [ ] Unique edges: 30,000+ (typically 40K-80K)
  - [ ] Graph Size Validation: Nodes ✓, Edges ✓

**Expected timing:**
- First run: ~1 minute (builds graph)
- Subsequent runs: ~5 seconds (loads from checkpoint)

**If graph too small:**
- Check Cell 6 completed successfully
- Delete checkpoint: `/content/drive/MyDrive/arcOS_benchmark/checkpoints/unified_graph.pkl`
- Re-run Cell 7

---

### Cell 8: Phase 1 Validation (~1 sec)
- [ ] Run cell
- [ ] Verify ALL criteria pass:
  - [ ] ✓ GPU Available
  - [ ] ✓ All Imports Successful
  - [ ] ✓ Dataset Splits Valid
  - [ ] ✓ Unified Graph Size Valid
  - [ ] ✓ Checkpoint Round-Trip
- [ ] Verify final message:
  - [ ] `✓ PHASE 1 COMPLETE - All criteria passed!`
  - [ ] `Ready to proceed to Phase 2: Retrieval Pipeline`

**If any criterion fails:**
- Read error message
- Check corresponding cell (e.g., GPU → Cell 1)
- Fix and re-run from failing cell onward

---

## Step 5: Verify Checkpoints

- [ ] In file browser, navigate to Google Drive
- [ ] Go to: `MyDrive/arcOS_benchmark/checkpoints/`
- [ ] Verify these files exist:
  - [ ] `dataset.pkl` (~200-500 MB)
  - [ ] `unified_graph.pkl` (~100-300 MB)
  - [ ] `test_roundtrip.pkl` (~1 KB)
  - [ ] `huggingface_cache/` (folder, ~500 MB)

**Total Drive usage:** ~1-1.5 GB

---

## Step 6: Test Warm Start

To verify checkpointing works:

- [ ] Runtime → Disconnect and delete runtime
- [ ] Runtime → Run all (or Ctrl+F9)
- [ ] Verify execution is faster (~2.5 min vs. ~5 min)
- [ ] Verify Cell 6 shows "Loading dataset from checkpoint..."
- [ ] Verify Cell 7 shows "Loading unified graph from checkpoint..."
- [ ] Verify Cell 8 still shows all ✓

---

## Step 7: Save Notebook

- [ ] File → Save (or Ctrl+S)
- [ ] Optionally: File → Save a copy in Drive
- [ ] Optionally: File → Download → Download .ipynb

---

## Troubleshooting Reference

| Issue | Cell | Fix |
|-------|------|-----|
| GPU not detected | 1 | Runtime → Change runtime type → GPU |
| Import failed | 2 | Verify `/content/arcOS-benchmark-colab/src/` exists |
| Drive mount failed | 5 | Re-run cell, complete authorization |
| Dataset download timeout | 6 | Wait 5 min, re-run cell |
| Graph too small | 7 | Check Cell 6, delete graph checkpoint, re-run |
| Validation failed | 8 | Check error message, fix corresponding cell |

---

## Success Metrics

After completing all steps, you should have:

- ✓ Notebook runs end-to-end without errors
- ✓ All 8 cells execute successfully
- ✓ Cell 8 shows "PHASE 1 COMPLETE"
- ✓ All 5 validation criteria pass
- ✓ Checkpoints saved to Google Drive
- ✓ Warm start works (faster re-run)

**Total execution time:**
- Cold start: ~5 minutes
- Warm start: ~2.5 minutes
- Re-run: ~15 seconds

---

## Next Steps After Phase 1

Once all criteria pass:

1. **Read Phase 2 plan:** See `docs/ROADMAP.md` Section 2
2. **Begin Phase 2 implementation:**
   - Sentence-Transformers embeddings
   - FAISS index
   - PCST subgraph extraction
3. **Continue with Phases 3-8**

---

## Notes

- **GPU quota:** Free Colab provides ~12 hours GPU per day
- **Session timeout:** Colab disconnects after ~90 min idle
- **Drive persistence:** Checkpoints survive disconnects
- **Best practice:** Save notebook after each successful run

---

**Checklist Version:** 1.0

**Last Updated:** 2026-02-09

**Phase:** 1 of 8 (Environment & Data Foundation)
