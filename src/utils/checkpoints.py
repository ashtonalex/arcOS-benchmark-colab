"""
Google Drive checkpoint management for persistent caching.

Enables idempotent save/load of expensive operations (dataset downloads, graph builds, etc.).
"""

import os
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Literal


def ensure_drive_mounted(drive_path: str = "/content/drive") -> bool:
    """
    Ensure Google Drive is mounted in Colab.

    Args:
        drive_path: Path where Drive should be mounted (default: /content/drive)

    Returns:
        True if mounted successfully, False otherwise
    """
    if os.path.exists(drive_path):
        print(f"✓ Google Drive already mounted at {drive_path}")
        return True

    try:
        from google.colab import drive
        drive.mount(drive_path)
        print(f"✓ Google Drive mounted at {drive_path}")
        return True
    except ImportError:
        print("Warning: Not running in Colab, Drive mount skipped")
        return False
    except Exception as e:
        print(f"Error mounting Drive: {e}")
        return False


def checkpoint_exists(filepath: Path | str) -> bool:
    """
    Check if a checkpoint file exists.

    Args:
        filepath: Path to checkpoint file

    Returns:
        True if file exists, False otherwise
    """
    return Path(filepath).exists()


def save_checkpoint(
    obj: Any,
    filepath: Path | str,
    format: Literal["pickle", "json", "graphml"] = "pickle"
):
    """
    Save an object to a checkpoint file.

    Args:
        obj: Object to save
        filepath: Destination file path
        format: Serialization format (pickle, json, graphml)

    Raises:
        ValueError: If format is unsupported
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == "json":
        with open(filepath, "w") as f:
            json.dump(obj, f, indent=2)
    elif format == "graphml":
        try:
            import networkx as nx
            nx.write_graphml(obj, filepath)
        except ImportError:
            raise ValueError("NetworkX required for graphml format")
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Checkpoint saved: {filepath} ({format})")


def load_checkpoint(
    filepath: Path | str,
    format: Literal["pickle", "json", "graphml"] = "pickle"
) -> Optional[Any]:
    """
    Load an object from a checkpoint file.

    Args:
        filepath: Source file path
        format: Serialization format (pickle, json, graphml)

    Returns:
        Loaded object, or None if file doesn't exist

    Raises:
        ValueError: If format is unsupported
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    if format == "pickle":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
    elif format == "json":
        with open(filepath, "r") as f:
            obj = json.load(f)
    elif format == "graphml":
        try:
            import networkx as nx
            obj = nx.read_graphml(filepath)
        except ImportError:
            raise ValueError("NetworkX required for graphml format")
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Checkpoint loaded: {filepath} ({format})")
    return obj


def create_checkpoint_dirs(checkpoint_dir: Path, results_dir: Path):
    """
    Create checkpoint and results directories if they don't exist.

    Args:
        checkpoint_dir: Path to checkpoint directory
        results_dir: Path to results directory
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint directory: {checkpoint_dir}")
    print(f"✓ Results directory: {results_dir}")
