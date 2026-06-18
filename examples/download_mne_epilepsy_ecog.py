"""Download MNE's small epilepsy ECoG sample into git-ignored `data/`."""

try:
    from ._bootstrap import REPO_ROOT
except ImportError:
    from _bootstrap import REPO_ROOT

import os


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    import mne

    data_path = mne.datasets.epilepsy_ecog.data_path(
        path=str(REPO_ROOT / "data"),
        update_path=False,
        download=True,
        verbose=True,
    )
    print(f"Downloaded/found MNE epilepsy ECoG data at: {data_path}")
    print("Next try:")
    print(
        "python examples/openneuro_real_data_pipeline.py --data-dir data --max-windows 2"
    )


if __name__ == "__main__":
    main()
