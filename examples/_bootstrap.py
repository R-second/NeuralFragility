"""Shared setup for examples run directly from this repository."""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = REPO_ROOT / ".cache"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))


def output_path(filename: str) -> Path:
    """Return a path for an output file under `outputs/`, creating the directory if needed.

    Args:
        filename: Filename (with extension) for the output file, e.g. "convergence_plot.png".

    Returns:
        Path to the output file under `outputs/`.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename
