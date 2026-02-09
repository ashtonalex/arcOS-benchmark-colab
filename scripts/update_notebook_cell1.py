#!/usr/bin/env python3
"""
Script to update Cell 1 of arcOS_benchmark.ipynb with proper UV package management.
This ensures absolute environment parity between the notebook kernel and installed packages.
"""

import json
from pathlib import Path

# New cell content with UV environment verification
NEW_CELL_CONTENT = """# ============================================================================
# ENVIRONMENT SETUP WITH UV PACKAGE MANAGER
# Ensures absolute environment parity between kernel and installed packages
# ============================================================================

import sys
import os
import subprocess
from pathlib import Path

# Colab UV workaround: Clear broken constraint files
os.environ["UV_CONSTRAINT"] = ""
os.environ["UV_BUILD_CONSTRAINT"] = ""

print("="*70)
print("STEP 1: ENVIRONMENT PATH VERIFICATION")
print("="*70)

# Capture current Python executable
current_python = sys.executable
print(f"Current kernel executable: {current_python}")
print(f"Python version: {sys.version}")
print(f"Site packages: {sys.path[0] if sys.path else 'N/A'}")

# Check if uv is available
def check_uv_available():
    \"\"\"Check if uv is installed and accessible.\"\"\"
    try:
        result = subprocess.run(
            ['uv', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

uv_available = check_uv_available()

if not uv_available:
    print("\\n⚠ UV not found. Installing uv package manager...")
    !pip install -q uv
    uv_available = check_uv_available()

if uv_available:
    # Get uv version
    uv_version = subprocess.run(
        ['uv', '--version'],
        capture_output=True,
        text=True
    ).stdout.strip()
    print(f"✓ UV available: {uv_version}")
else:
    print("✗ UV installation failed. Will fall back to pip.")

print("\\n" + "="*70)
print("STEP 2: PACKAGE INSTALLATION")
print("="*70)

# Define packages to install
packages = [
    "datasets",
    "networkx",
    "tqdm"
]

# PyTorch with CUDA support
torch_packages = "torch torchvision torchaudio"
torch_index = "https://download.pytorch.org/whl/cu118"

if uv_available:
    print(f"Installing packages using UV with --python {current_python}\\n")
    
    # Install PyTorch with CUDA
    print("Installing PyTorch with CUDA 11.8 support...")
    !uv pip install --python {current_python} {torch_packages} --index-url {torch_index}
    
    # Install other packages
    print("\\nInstalling additional dependencies...")
    for package in packages:
        !uv pip install --python {current_python} {package}
else:
    print("Falling back to standard pip installation\\n")
    
    # Install PyTorch with CUDA
    print("Installing PyTorch with CUDA 11.8 support...")
    %pip install -q {torch_packages} --index-url {torch_index}
    
    # Install other packages
    print("\\nInstalling additional dependencies...")
    %pip install -q {' '.join(packages)}

print("\\n" + "="*70)
print("STEP 3: INSTALLATION VERIFICATION")
print("="*70)

# Verify installed packages are in the correct location
def verify_package_location(package_name):
    \"\"\"Verify package is installed in current kernel's site-packages.\"\"\"
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

# Verify key packages
verification_packages = ['torch', 'datasets', 'networkx', 'tqdm']
print("\\nVerifying installed packages:\\n")

all_verified = True
for pkg in verification_packages:
    info = verify_package_location(pkg)
    if info['installed']:
        status = "✓" if info['in_sys_path'] else "⚠"
        print(f"{status} {pkg:12s} v{info['version']:12s}")
        print(f"  Location: {info['location']}")
        if not info['in_sys_path']:
            print(f"  WARNING: Not in sys.path!")
            all_verified = False
    else:
        print(f"✗ {pkg:12s} NOT INSTALLED")
        all_verified = False
    print()

print("="*70)
print("STEP 4: GPU VERIFICATION")
print("="*70)

import torch
gpu_available = torch.cuda.is_available()
print(f"\\nGPU available: {gpu_available} {'✓' if gpu_available else '✗'}")

if gpu_available:
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("⚠ Warning: No GPU detected.")
    print("  Go to: Runtime -> Change runtime type -> Select T4 GPU")

print("\\n" + "="*70)
print("ENVIRONMENT SETUP SUMMARY")
print("="*70)
print(f"Package manager: {'UV' if uv_available else 'pip'}")
print(f"Python executable: {current_python}")
print(f"All packages verified: {'✓ YES' if all_verified else '✗ NO'}")
print(f"GPU available: {'✓ YES' if gpu_available else '✗ NO'}")
print("="*70)

if not all_verified:
    print("\\n⚠ WARNING: Some packages failed verification. Check output above.")
else:
    print("\\n✓ Environment setup complete with full parity!")"""


def update_notebook():
    """Update the notebook's first code cell with new UV setup."""
    notebook_path = Path(__file__).parent.parent / "notebooks" / "arcOS_benchmark.ipynb"
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return False
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find the first code cell (Cell 1 - Environment Setup)
    code_cell_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code_cell_index = i
            break
    
    if code_cell_index is None:
        print("Error: No code cells found in notebook")
        return False
    
    # Split content into lines for notebook format
    new_source = [line + '\n' for line in NEW_CELL_CONTENT.split('\n')]
    # Remove trailing newline from last line
    if new_source:
        new_source[-1] = new_source[-1].rstrip('\n')
    
    # Update the cell
    notebook['cells'][code_cell_index]['source'] = new_source
    
    # Save updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✓ Successfully updated {notebook_path}")
    print(f"  Updated cell index: {code_cell_index}")
    print(f"  New cell has {len(new_source)} lines")
    return True


if __name__ == "__main__":
    success = update_notebook()
    exit(0 if success else 1)
