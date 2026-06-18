"""Matrix examples used by numerical experiments.

Add paper-specific matrices here and select them from experiment scripts by
name. Large matrices should live in `examples/matrix_data/*.npz` and be loaded with
`allow_pickle=False`.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

DATA_DIR = Path(__file__).resolve().parent / "matrix_data"
FloatArray: TypeAlias = NDArray[np.floating]
PathLikeStr: TypeAlias = str | PathLike[str]

A0 = np.array(
    [
        [0.5, 0.2],
        [0.1, 0.4],
    ]
)


IN_MEMORY_MATRICES = {
    "A0": A0,
}

NPZ_MATRICES = {
    "A_hard": DATA_DIR / "A_hard.npz",
}


def load_npz_matrix(path: PathLikeStr) -> FloatArray:
    """Load a matrix from a `.npz` file.

    Args:
        path: Path to the `.npz` file containing the matrix under the key "A".

    Returns:
        The loaded matrix.
    """
    with np.load(path, allow_pickle=False) as data:
        return data["A"]


def available_matrices() -> list[str]:
    """List the names of matrices available for use in experiment scripts.

    Args:
        None.

    Returns:
        A sorted list of registered matrix names.
    """
    return sorted({*IN_MEMORY_MATRICES, *NPZ_MATRICES})


def get_matrix(name: str) -> FloatArray:
    """Get a matrix example by name.

    Args:
        name: The name of the matrix to retrieve, which must be in `available_matrices`.

    Returns:
        The requested matrix.
    """
    if name in IN_MEMORY_MATRICES:
        return IN_MEMORY_MATRICES[name]

    if name in NPZ_MATRICES:
        return load_npz_matrix(NPZ_MATRICES[name])

    available = ", ".join(available_matrices())
    raise ValueError(f"Unknown matrix {name!r}. Available: {available}")
