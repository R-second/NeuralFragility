"""Matrix examples used by numerical experiments.

Add paper-specific matrices here and select them from experiment scripts by
name. Large matrices should live in `examples/data/*.npz` and be loaded with
`allow_pickle=False`.
"""
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

A0 = np.array([
    [0.5, 0.2],
    [0.1, 0.4],
])


IN_MEMORY_MATRICES = {
    "A0": A0,
}

NPZ_MATRICES = {
    "A_hard": DATA_DIR / "A_hard.npz",
}


def load_npz_matrix(path):
    with np.load(path, allow_pickle=False) as data:
        return data["A"]


def available_matrices():
    return sorted({*IN_MEMORY_MATRICES, *NPZ_MATRICES})


def get_matrix(name):
    if name in IN_MEMORY_MATRICES:
        return IN_MEMORY_MATRICES[name]

    if name in NPZ_MATRICES:
        return load_npz_matrix(NPZ_MATRICES[name])

    available = ", ".join(available_matrices())
    raise ValueError(f"Unknown matrix {name!r}. Available: {available}")
