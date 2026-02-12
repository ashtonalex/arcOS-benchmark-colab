# Colab Plugin Setup Guide

## Using IDE Colab Plugin (VS Code, PyCharm, etc.)

Good news! Using the Colab plugin is **easier** than the web interface because your local files are automatically synced.

---

## Setup Steps

### 1. Open the Notebook in Your IDE

Your files are already in the correct structure:
```
C:\Users\User\arcOS-benchmark-colab\
├── src/
│   └── (all modules)
└── notebooks/
    └── arcOS_benchmark.ipynb
```

- **VS Code:** Open `notebooks/arcOS_benchmark.ipynb` → Select "Colab" kernel
- **PyCharm/JetBrains:** Right-click notebook → "Run on Colab"

### 2. Update Cell 2 (Path Detection)

Replace the current Cell 2 code with this **auto-detecting version**:

```python
import sys
from pathlib import Path

# Auto-detect project root (works with Colab plugin)
# Try multiple possible locations
possible_roots = [
    Path.cwd().parent,  # If running from notebooks/ folder
    Path.cwd(),  # If running from project root
    Path("/content/arcOS-benchmark-colab"),  # Standard Colab path
    Path("/content"),  # Web Colab with manual upload
]

project_root = None
for root in possible_roots:
    src_path = root / "src"
    if src_path.exists() and (src_path / "config.py").exists():
        project_root = root
        break

if project_root is None:
    print("⚠ ERROR: Could not find src/ directory")
    print(f"Searched in: {[str(p) for p in possible_roots]}")
    print(f"Current directory: {Path.cwd()}")
    raise ImportError("src/ not found - check project structure")

# Add to Python path
sys.path.insert(0, str(project_root))
print(f"✓ Found project root: {project_root}")
print(f"✓ Added to Python path: {project_root}")

# Import modules
print("\nImporting modules...")
try:
    from src.config import BenchmarkConfig
    from src.utils.seeds import set_seeds
    from src.utils.checkpoints import (
        ensure_drive_mounted,
        checkpoint_exists,
        save_checkpoint,
        load_checkpoint,
        create_checkpoint_dirs,
    )
    from src.data.dataset_loader import RoGWebQSPLoader
    from src.data.graph_builder import GraphBuilder
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path[:3]}")
    raise
```

This version will **automatically find** the correct path regardless of:
- Which IDE you're using
- Where Colab syncs your files
- Whether you're in web Colab or plugin

---

## 3. Connect to GPU Runtime

In your IDE's Colab plugin:

**VS Code:**
1. Click the kernel selector (top right)
2. Select "Colab" runtime
3. Choose "GPU" accelerator
4. Wait for connection (~10 seconds)

**PyCharm/JetBrains:**
1. Run → Edit Configurations
2. Set runtime to "Google Colab"
3. Enable GPU in runtime settings
4. Click "Run"

---

## 4. Run the Notebook

Execute cells in order (just like normal):
- **Shift+Enter** to run current cell and move to next
- **Ctrl+Enter** to run current cell and stay
- Use IDE's "Run All" button for full execution

**Expected time:**
- First run: ~5 minutes (with downloads)
- Subsequent runs: ~15 seconds (with checkpoints)

---

## Advantages of Colab Plugin

### ✓ **No Manual File Upload**
- Your local `src/` files are automatically synced
- Edit code locally, run on Colab GPU instantly

### ✓ **Better Development Experience**
- Use your IDE's autocomplete, linting, debugging
- Version control (git) works naturally
- Can edit `.py` files and notebook side-by-side

### ✓ **Faster Iteration**
- Make changes locally
- Re-run cells immediately
- No need to re-upload files

### ✓ **Integrated Output**
- Cell outputs appear inline in your IDE
- Can use IDE's variable explorer
- Better error messages and stack traces

---

## Workflow

### Standard Development Loop

1. **Make changes** to `src/config.py` (or any module) in your IDE
2. **Save file** (Ctrl+S)
3. **Restart kernel** (Ctrl+Shift+P → "Restart Kernel")
4. **Re-run cells** that use the changed module
5. Changes take effect immediately!

### Testing Changes

**Example:** Modifying configuration defaults

1. Edit `src/config.py`:
   ```python
   seed: int = 99  # Changed from 42
   ```

2. In notebook, restart kernel

3. Run Cell 2 (imports) + Cell 3 (config):
   ```python
   config = BenchmarkConfig()
   print(config.seed)  # Should show 99
   ```

4. No need to re-run Cell 1 (dependencies already installed)

---

## Troubleshooting

### Issue: "Could not find src/ directory"

**Cause:** Path detection failed

**Fix:** Add debug cell before Cell 2:
```python
from pathlib import Path
print(f"Current directory: {Path.cwd()}")
print(f"Files here: {list(Path.cwd().iterdir())[:10]}")
print(f"Parent directory: {Path.cwd().parent}")
print(f"Files in parent: {list(Path.cwd().parent.iterdir())[:10]}")
```

Then look for where `src/` is located and adjust `possible_roots`.

---

### Issue: "GPU available: False"

**Cause:** Connected to CPU runtime

**Fix:**
- **VS Code:** Click kernel selector → Change runtime → GPU
- **PyCharm:** Settings → Colab → Hardware accelerator → GPU
- Reconnect to runtime

---

### Issue: Changes to `.py` files not taking effect

**Cause:** Python caches imported modules

**Fix:**
1. Restart kernel (Ctrl+Shift+P → "Restart Kernel")
2. Re-run Cell 2 (imports)
3. Continue from there

**Or** use auto-reload magic (add to top of notebook):
```python
%load_ext autoreload
%autoreload 2
```

---

### Issue: "Drive mount failed"

**Cause:** Colab plugin may handle Drive differently

**Fix:** First time will show authorization prompt in IDE. Follow the link, authorize, paste code.

**Alternative:** If Drive mount keeps failing, switch to local checkpointing:
```python
# In Cell 3, change:
config = BenchmarkConfig(
    drive_root="/tmp/arcOS_benchmark",  # Use local temp storage
)
```

**Note:** Local checkpoints won't persist across runtime disconnects.

---

## Performance Tips

### Tip 1: Keep Runtime Connected
- Colab plugin maintains connection while IDE is open
- Free tier: ~12 hours GPU per day
- Close other GPU notebooks to avoid quota issues

### Tip 2: Use Checkpoints
- First run: Downloads dataset (~500 MB) and builds graph (~1 min)
- After that: Loads from checkpoint (~5 sec total)
- Only re-run Cell 1 after runtime disconnect

### Tip 3: Selective Re-runs
After editing code:
- Changed `config.py`? Re-run Cell 2-3
- Changed `graph_builder.py`? Re-run Cell 2, 7
- Changed `dataset_loader.py`? Re-run Cell 2, 6
- No need to re-run Cell 1 (dependencies) unless adding new packages

### Tip 4: Debug Locally First
- For pure Python logic (no GPU needed), test locally:
  ```bash
  python test_phase1_imports.py
  ```
- Only run on Colab for GPU or large dataset operations

---

## Quick Reference

| Action | Command |
|--------|---------|
| Run cell | Shift+Enter |
| Run all cells | Ctrl+Shift+Enter (VS Code) |
| Restart kernel | Ctrl+Shift+P → "Restart" |
| Change runtime | Click kernel selector → GPU |
| Disconnect | Click "Disconnect" in IDE |
| View variables | IDE variable explorer |
| Debug cell | Set breakpoint → Debug button |

---

## Next Steps

Once you have the updated Cell 2:

1. **Connect to GPU runtime** in your IDE
2. **Run Cell 1** to install dependencies (~2 min)
3. **Run Cells 2-8** sequentially
4. **Verify Cell 8** shows all ✓
5. **Proceed to Phase 2** implementation

All your local edits will automatically sync to the Colab runtime!

---

**Last Updated:** 2026-02-09

**Tested With:**
- VS Code + Colab extension
- PyCharm + Colab plugin
- Cursor IDE + Colab integration
