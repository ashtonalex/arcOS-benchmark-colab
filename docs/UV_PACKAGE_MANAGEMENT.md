# UV Package Management for Jupyter Notebooks

## Overview

This document explains the UV package management setup implemented in `arcOS_benchmark.ipynb` Cell 1, which ensures **absolute environment parity** between the notebook kernel and installed packages.

## Key Features

### 1. **Path Verification**
- Captures `sys.executable` to identify the exact Python interpreter running the kernel
- Displays Python version and site-packages location
- Ensures UV targets the correct environment

### 2. **UV Installation with Fallback**
- Checks if UV is already available using `subprocess.run(['uv', '--version'])`
- Installs UV via pip if not found
- Gracefully falls back to standard `%pip install` if UV installation fails

### 3. **Explicit Python Targeting**
- Uses `!uv pip install --python {sys.executable} <packages>` syntax
- Ensures packages are installed to the **exact** Python interpreter running the kernel
- Prevents environment mismatches common in Colab/Jupyter environments

### 4. **Installation Verification**
- After installation, imports each package and checks:
  - Package version
  - Installation location (file path)
  - Whether the location is in `sys.path`
- Prints diagnostic output for each package
- Flags any packages not in the expected location

### 5. **GPU Verification**
- Checks `torch.cuda.is_available()`
- Displays GPU name, memory, and CUDA version
- Provides actionable guidance if GPU is not detected

## Implementation Details

### Package Installation Logic

```python
if uv_available:
    # Install with explicit Python targeting
    !uv pip install --python {current_python} torch torchvision torchaudio --index-url {torch_index}
    !uv pip install --python {current_python} {package}
else:
    # Fallback to standard pip
    %pip install -q torch torchvision torchaudio --index-url {torch_index}
    %pip install -q {package}
```

### Verification Function

```python
def verify_package_location(package_name):
    """Verify package is installed in current kernel's site-packages."""
    try:
        module = __import__(package_name)
        module_path = Path(module.__file__).parent
        
        # Check if module is in one of sys.path locations
        in_sys_path = any(str(module_path).startswith(p) for p in sys.path if p)
        
        # Get version if available
        version = getattr(module, '__version__', 'unknown')
        
        return {
            'installed': True,
            'version': version,
            'location': str(module_path),
            'in_sys_path': in_sys_path
        }
    except ImportError:
        return {'installed': False}
```

## Why This Matters

### Problem: Environment Mismatch
In Jupyter/Colab environments, multiple Python interpreters can coexist:
- System Python
- Virtual environment Python
- Conda environment Python
- Kernel-specific Python

Installing packages without explicit targeting can result in:
- Packages installed to wrong environment
- Import errors despite "successful" installation
- Version conflicts
- Inconsistent behavior between cells

### Solution: Explicit Targeting
Using `--python {sys.executable}` ensures:
- ✓ Packages installed to **exact** kernel interpreter
- ✓ No environment ambiguity
- ✓ Immediate availability in notebook
- ✓ Reproducible across different Colab sessions

## Output Example

```
======================================================================
STEP 1: ENVIRONMENT PATH VERIFICATION
======================================================================
Current kernel executable: /usr/bin/python3
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05)
Site packages: /usr/local/lib/python3.10/dist-packages

✓ UV available: uv 0.1.0

======================================================================
STEP 2: PACKAGE INSTALLATION
======================================================================
Installing packages using UV with --python /usr/bin/python3

Installing PyTorch with CUDA 11.8 support...
Installing additional dependencies...

======================================================================
STEP 3: INSTALLATION VERIFICATION
======================================================================

Verifying installed packages:

✓ torch        v2.1.0+cu118
  Location: /usr/local/lib/python3.10/dist-packages/torch

✓ datasets     v2.14.6
  Location: /usr/local/lib/python3.10/dist-packages/datasets

✓ networkx     v3.2.1
  Location: /usr/local/lib/python3.10/dist-packages/networkx

✓ tqdm         v4.66.1
  Location: /usr/local/lib/python3.10/dist-packages/tqdm

======================================================================
STEP 4: GPU VERIFICATION
======================================================================

GPU available: True ✓
GPU name: Tesla T4
GPU memory: 15.36 GB
CUDA version: 11.8

======================================================================
ENVIRONMENT SETUP SUMMARY
======================================================================
Package manager: UV
Python executable: /usr/bin/python3
All packages verified: ✓ YES
GPU available: ✓ YES
======================================================================

✓ Environment setup complete with full parity!
```

## Error Handling

### UV Not Available
- Automatically installs UV via pip
- Falls back to `%pip install` if UV installation fails
- Continues execution without blocking

### Package Verification Failures
- Displays warning if packages not in `sys.path`
- Shows exact installation location for debugging
- Provides clear summary of verification status

### GPU Not Available
- Displays actionable guidance: "Runtime -> Change runtime type -> Select T4 GPU"
- Does not block execution (allows CPU-only testing)

## Colab-Specific Workarounds

### UV Constraint Files
```python
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""
```

Clears broken constraint files that can interfere with UV in Colab environments.

### PyTorch CUDA Index
```python
torch_index = "https://download.pytorch.org/whl/cu118"
```

Uses CUDA 11.8 wheel index for compatibility with Colab T4 GPUs.

## Best Practices

1. **Always run Cell 1 first** - Establishes environment before imports
2. **Check verification output** - Ensure all packages show ✓ status
3. **Restart kernel if issues** - Runtime -> Restart runtime, then re-run Cell 1
4. **Use UV for all installs** - Maintain consistency throughout notebook

## Troubleshooting

### "Package not in sys.path" Warning
- **Cause**: Package installed to different Python interpreter
- **Solution**: Restart kernel and re-run Cell 1

### "UV installation failed"
- **Cause**: Network issues or pip unavailable
- **Solution**: Cell automatically falls back to pip

### Import errors despite successful installation
- **Cause**: Kernel using different Python than installation target
- **Solution**: Restart kernel, verify `sys.executable` matches installation path

## References

- [UV Documentation](https://github.com/astral-sh/uv)
- [Jupyter Kernel Management](https://jupyter-client.readthedocs.io/en/stable/kernels.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
