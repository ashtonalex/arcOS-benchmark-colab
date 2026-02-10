# Running arcOS Benchmark in Google Colab from IDE

## Method 1: VSCode with Colab Extension

### Initial Setup

1. **Install Colab extension** in VSCode
2. **Open notebook**: `notebooks/arcOS_benchmark.ipynb`
3. **Select kernel**: Click kernel selector → "Google Colab" → Sign in
4. **Change runtime**: Runtime → Change runtime type → T4 GPU

### Upload Project Files

Add this as **Cell 0** (before existing Cell 1) in the notebook:

```python
# Upload project files to Colab runtime
import os
from pathlib import Path

# Create project directory
!mkdir -p /content/arcOS-benchmark-colab

# Upload src folder
# Method 1: Clone from GitHub (recommended)
!git clone https://github.com/YOUR_USERNAME/arcOS-benchmark-colab /content/arcOS-benchmark-colab

# Method 2: Upload manually
# Uncomment if not using git:
# from google.colab import files
# import zipfile
#
# print("Upload your project zip file (create zip of entire folder first)")
# uploaded = files.upload()
#
# # Extract
# for filename in uploaded.keys():
#     !unzip -q {filename} -d /content/
#     print(f"Extracted {filename}")

# Verify upload
print("\nVerifying project structure:")
src_path = Path("/content/arcOS-benchmark-colab/src")
if src_path.exists():
    print(f"✓ Found src/ directory")
    print(f"  Modules: {list(src_path.glob('*/'))}")
else:
    print("✗ src/ directory not found - upload failed")
```

### Run Cells

1. Mount Google Drive (Cell 1)
2. Install dependencies (Cell 2-4)
3. Run Phase 1 (Cells 5-8)
4. Run Phase 2 (Cells 9-11)

---

## Method 2: Direct Colab Web Interface (Simpler)

### Step 1: Push to GitHub

```bash
# Add all files to git
git add .
git commit -m "ready for colab testing"
git push origin main
```

### Step 2: Open in Colab

1. Go to https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Enter your repo URL: `https://github.com/YOUR_USERNAME/arcOS-benchmark-colab`
4. Select `notebooks/arcOS_benchmark.ipynb`

### Step 3: Clone Repo in Colab

First cell:
```python
!git clone https://github.com/YOUR_USERNAME/arcOS-benchmark-colab /content/arcOS-benchmark-colab
%cd /content/arcOS-benchmark-colab
```

Then run cells sequentially.

---

## Method 3: Manual File Upload (Quick Test)

### Step 1: Create Zip

```bash
# Windows
Compress-Archive -Path src -DestinationPath src.zip

# Linux/Mac
zip -r src.zip src/
```

### Step 2: Upload in Colab

Add this cell at the top:

```python
from google.colab import files
import zipfile

# Upload zip
uploaded = files.upload()

# Extract
!unzip -q src.zip -d /content/arcOS-benchmark-colab/
print("✓ Files uploaded")
```

---

## Recommended Workflow

**For development:**
1. Edit code locally in VSCode
2. Push to GitHub
3. In Colab, run `!git pull` to update
4. Run cells to test

**For testing Phase 2:**
1. Ensure latest code is on GitHub
2. Open Colab (web or IDE)
3. Clone/pull latest code
4. Run Cells 1-11 sequentially
5. Check Cell 11 for success criteria

---

## File Structure in Colab

After setup, your Colab runtime should have:

```
/content/
├── drive/                          # Google Drive mount
│   └── MyDrive/
│       └── arcOS_benchmark/
│           ├── checkpoints/       # Phase 1 & 2 checkpoints
│           └── results/
└── arcOS-benchmark-colab/         # Your project
    ├── src/
    │   ├── config.py
    │   ├── utils/
    │   ├── data/
    │   └── retrieval/            # ← Phase 2 modules
    └── notebooks/
        └── arcOS_benchmark.ipynb
```

---

## Troubleshooting

### "Module not found" errors

```python
# Verify sys.path includes project root
import sys
from pathlib import Path

project_root = Path("/content/arcOS-benchmark-colab")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Test import
from src.retrieval import Retriever
print("✓ Import successful")
```

### Files not uploading

```python
# Check current directory
!pwd
!ls -la

# Check if src exists
!ls -la /content/arcOS-benchmark-colab/src/
```

### Git clone fails

```bash
# Delete old clone and retry
!rm -rf /content/arcOS-benchmark-colab
!git clone https://github.com/YOUR_USERNAME/arcOS-benchmark-colab /content/arcOS-benchmark-colab
```

---

## Cell Execution Order

**Phase 1 (Cells 1-8):**
- Cell 1: Mount Drive
- Cell 2: Environment setup
- Cell 3: Import modules
- Cell 4: Configuration
- Cell 5: Set seeds
- Cell 6: Google Drive setup
- Cell 7: Load dataset
- Cell 8: Build graphs & validate

**Phase 2 (Cells 9-11):**
- Cell 9: Build retrieval pipeline (~5-10 min first run)
- Cell 10: Validate on 10 examples (~30 sec)
- Cell 11: Check success criteria

**Expected runtime:**
- Cold start (first run): ~15-20 minutes total
- Warm start (checkpoints exist): ~2-3 minutes total

---

## Tips

1. **Save checkpoints:** Always keep Google Drive mounted to preserve checkpoints
2. **GPU required:** Phase 2 needs GPU for embedding computation (CPU fallback is 10x slower)
3. **Runtime disconnects:** Checkpoints prevent data loss - just re-run cells
4. **Test incrementally:** Run Cells 1-8 first, verify Phase 1 passes, then run 9-11
5. **Monitor progress:** Cell 9 shows progress bars for embedding computation

---

## Quick Start Checklist

- [ ] Code pushed to GitHub (or zipped locally)
- [ ] Colab runtime set to GPU (T4 or better)
- [ ] Google Drive mounted
- [ ] Project files uploaded/cloned to `/content/arcOS-benchmark-colab`
- [ ] Cells 1-8 run successfully (Phase 1 complete)
- [ ] Ready to run Cells 9-11 (Phase 2)
