"""Run the OpenNeuro BrainVision pipeline without committing raw data.

Expected local files under `data/` by default:

- sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg.vhdr
- sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_events.tsv
- sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_channels.tsv
"""

from __future__ import annotations

try:
    from ._bootstrap import output_path
except ImportError:
    from _bootstrap import output_path

import argparse
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from eeg_fragility import (
    compute_stability_radius_from_matrices,
    create_sliding_windows,
    estimate_transition_matrices,
    calculate_neural_fragility,
    save_fragility_npz,
)

DEFAULT_PREFIX = "auto"
FloatArray: TypeAlias = NDArray[np.floating]
PathLikeStr: TypeAlias = str | PathLike[str]


def require_optional_dependencies() -> tuple[Any, Any]:
    """Import optional dependencies for OpenNeuro pipeline, exiting with an error message if not found.

    Args:
        None.

    Returns:
        Imported `mne` and `pandas` modules.
    """
    try:
        import mne
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Install OpenNeuro extras first: pip install -r requirements-openneuro.txt"
        ) from exc
    return mne, pd


def detect_seizure_onset(events_file: PathLikeStr) -> float:
    """Detect seizure onset time from events.tsv.

    Args:
        events_file: BIDS format events.tsv path.

    Returns:
        The onset time of the first row containing `onset` in the `trial_type` column.
    """
    _, pd = require_optional_dependencies()
    events = pd.read_csv(events_file, sep="\t")
    onset_rows = events[
        events["trial_type"].str.contains("onset", case=False, na=False)
    ]
    if onset_rows.empty:
        raise ValueError(f"Onset event not found in {events_file}")
    return float(onset_rows.iloc[0]["onset"])


def load_bad_channels(channels_file: PathLikeStr) -> list[str]:
    """Load bad channel names from channels.tsv.

    Args:
        channels_file: BIDS format channels.tsv path.

    Returns:
        List of channel names where `status` is `bad`.
    """
    _, pd = require_optional_dependencies()
    channels = pd.read_csv(channels_file, sep="\t")
    if "status" not in channels.columns:
        return []
    bad_rows = channels[channels["status"].astype(str).str.lower() == "bad"]
    return bad_rows["name"].tolist()


def resolve_openneuro_files(
    data_dir: PathLikeStr,
    subject_prefix: str,
) -> tuple[Path, Path, Path, str]:
    """Resolve BrainVision-related files under the data directory.

    Args:
        data_dir: Directory containing OpenNeuro/BIDS data.
        subject_prefix: Target file prefix, or `"auto"` for automatic detection.

    Returns:
        Path to `.vhdr` file, events.tsv file, channels.tsv file, and resolved subject prefix.
    """
    data_dir = Path(data_dir)
    if subject_prefix == "auto":
        vhdr_files = sorted(data_dir.rglob("*_ieeg.vhdr"))
        if not vhdr_files:
            raise FileNotFoundError(f"No *_ieeg.vhdr files found under {data_dir}")
        if len(vhdr_files) > 1:
            choices = "\n".join(f"- {path}" for path in vhdr_files[:20])
            raise ValueError(
                "Multiple *_ieeg.vhdr files found. Pass --subject-prefix explicitly.\n"
                f"Candidates:\n{choices}"
            )
        vhdr_file = vhdr_files[0]
        subject_prefix = vhdr_file.name.removesuffix("_ieeg.vhdr")
    else:
        matches = sorted(data_dir.rglob(f"{subject_prefix}_ieeg.vhdr"))
        if not matches:
            raise FileNotFoundError(
                f"No {subject_prefix}_ieeg.vhdr found under {data_dir}"
            )
        vhdr_file = matches[0]

    file_dir = vhdr_file.parent
    events_file = file_dir / f"{subject_prefix}_events.tsv"
    channels_file = file_dir / f"{subject_prefix}_channels.tsv"

    missing = [
        path for path in [vhdr_file, events_file, channels_file] if not path.exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing OpenNeuro files:\n{missing_list}")

    return vhdr_file, events_file, channels_file, subject_prefix


def load_and_preprocess_raw(args: argparse.Namespace) -> tuple[Any, float, float]:
    """Load BrainVision recordings and preprocess them according to CLI settings.

    Args:
        args: `argparse` で得た設定値。

    Returns:
        Preprocessed MNE Raw object, detected seizure onset time, and crop start time.
    """
    mne, _ = require_optional_dependencies()

    vhdr_file, events_file, channels_file, subject_prefix = resolve_openneuro_files(
        Path(args.data_dir),
        args.subject_prefix,
    )
    seizure_onset_time = detect_seizure_onset(events_file)
    bad_channels = load_bad_channels(channels_file)

    tmin = seizure_onset_time - args.seconds_before
    tmax = seizure_onset_time + args.seconds_after

    print(f"Using recording: {subject_prefix}")
    print(f"Header file: {vhdr_file}")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=False, verbose=False)
    crop_start_time = max(0.0, tmin)
    crop_end_time = min(float(raw.times[-1]), tmax)
    raw_cropped = raw.crop(tmin=crop_start_time, tmax=crop_end_time)
    raw_cropped.load_data()

    channels_to_drop = [
        channel for channel in bad_channels if channel in raw_cropped.ch_names
    ]
    if channels_to_drop:
        raw_cropped.drop_channels(channels_to_drop)

    if args.notch_freq > 0:
        raw_cropped.notch_filter(freqs=args.notch_freq, notch_widths=args.notch_width)

    if args.highpass_freq > 0:
        iir_params = {"order": 4, "ftype": "butter"}
        raw_cropped.filter(
            l_freq=args.highpass_freq, h_freq=None, method="iir", iir_params=iir_params
        )

    if not args.no_car:
        raw_cropped.set_eeg_reference(ref_channels="average", projection=False)

    print(f"Detected onset: {seizure_onset_time:.3f} s")
    print(f"Cropped time range: {crop_start_time:.3f} s to {crop_end_time:.3f} s")
    print(f"Loaded data shape: {raw_cropped.get_data().shape} (channels, samples)")
    print(f"Dropped bad channels: {len(channels_to_drop)}")
    return raw_cropped, seizure_onset_time, crop_start_time


def plot_heatmap(
    neural_fragility: FloatArray,
    times: FloatArray,
    onset_time: float,
    channel_names: list[str],
    output_file: PathLikeStr,
) -> None:
    """Save a neural fragility heatmap with the seizure onset position overlaid.

    Args:
        neural_fragility: Neural fragility array of shape `(n_channels, n_windows)`.
        times: Absolute center times of each window.
        onset_time: Seizure onset time.
        channel_names: List of channel names.
        output_file: Path to the output image file.

    Returns:
        None.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times)
    if times.size == 0:
        raise ValueError("times must contain at least one window center.")
    if times.size != neural_fragility.shape[1]:
        raise ValueError("times must have one value per heatmap window.")

    plt.figure(figsize=(15, 10))
    plt.imshow(neural_fragility, aspect="auto", cmap="turbo", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Neural Fragility")

    n_windows = neural_fragility.shape[1]
    n_ticks = min(20, n_windows)
    tick_indices = np.linspace(0, n_windows - 1, n_ticks, dtype=int)
    tick_labels = [f"{times[i] - onset_time:.1f}" for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=45)

    if len(channel_names) <= 120:
        plt.yticks(np.arange(len(channel_names)), channel_names, fontsize=7)

    if len(times) > 1:
        median_step = float(np.median(np.diff(times)))
        if median_step == 0:
            raise ValueError("times must contain distinct window centers.")
        half_step = abs(median_step) / 2
        display_start = float(times[0]) - half_step
        display_end = float(times[-1]) + half_step
        onset_x = (onset_time - float(times[0])) / median_step
    else:
        display_start = float(times[0])
        display_end = float(times[0])
        onset_x = 0.0

    if display_start <= onset_time <= display_end:
        plt.axvline(
            x=onset_x,
            color="white",
            linestyle="--",
            linewidth=2,
            label="Seizure Onset",
        )
        plt.legend(loc="upper right")
    plt.xlabel("Time relative to Seizure Onset (s)")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Saved heatmap to {output_file}")


def main() -> None:
    """Parse CLI arguments and run the local OpenNeuro BrainVision pipeline.

    Args:
        None.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Run OpenNeuro BrainVision fragility analysis from local data."
    )
    parser.add_argument(
        "--data-dir", default="data", help="Local data directory ignored by git."
    )
    parser.add_argument(
        "--subject-prefix",
        default=DEFAULT_PREFIX,
        help="OpenNeuro file prefix, or 'auto'.",
    )
    parser.add_argument(
        "--seconds-before",
        type=float,
        default=60.0,
        help="Seconds before seizure onset.",
    )
    parser.add_argument(
        "--seconds-after", type=float, default=60.0, help="Seconds after seizure onset."
    )
    parser.add_argument(
        "--window-ms", type=float, default=250.0, help="Window size in milliseconds."
    )
    parser.add_argument(
        "--step-ms", type=float, default=125.0, help="Step size in milliseconds."
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="Ridge regularization for transition estimation.",
    )
    parser.add_argument(
        "--method",
        choices=["branch filtering method", "grid search"],
        default="branch filtering method",
        help="Fragility solver.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=100,
        help='Grid points when method="grid search".',
    )
    parser.add_argument("--gamma", type=float, default=0.01, help="Algorithm gamma.")
    parser.add_argument(
        "--max-iter", type=int, default=20, help="Maximum algorithm iterations."
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-6, help="Algorithm convergence tolerance."
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=5,
        help="Limit windows for experiments; 0 means all.",
    )
    parser.add_argument(
        "--notch-freq",
        type=float,
        default=60.0,
        help="Notch filter frequency; 0 disables.",
    )
    parser.add_argument(
        "--notch-width", type=float, default=2.0, help="Notch filter width."
    )
    parser.add_argument(
        "--highpass-freq",
        type=float,
        default=0.5,
        help="High-pass frequency; 0 disables.",
    )
    parser.add_argument(
        "--no-car", action="store_true", help="Disable common average reference."
    )
    parser.add_argument(
        "--output",
        default=str(output_path("openneuro_real_heatmap.png")),
        help="Output heatmap path.",
    )
    parser.add_argument(
        "--data-output",
        default=str(output_path("openneuro_real_fragility.npz")),
        help="Output arrays path.",
    )
    args = parser.parse_args()

    raw, onset_time, crop_start_time = load_and_preprocess_raw(args)
    windows, window_times = create_sliding_windows(
        raw, window_size_ms=args.window_ms, step_size_ms=args.step_ms
    )
    absolute_window_times = window_times + crop_start_time

    if args.max_windows > 0:
        windows = windows[: args.max_windows]
        absolute_window_times = absolute_window_times[: args.max_windows]

    print(f"Windows shape: {windows.shape}")
    transition_matrices = estimate_transition_matrices(windows, l2_lambda=args.ridge)

    raw_fragility = compute_stability_radius_from_matrices(
        transition_matrices,
        gamma=args.gamma,
        method=args.method,
        grid_points=args.grid_points,
        max_iter=args.max_iter,
        epsilon=args.epsilon,
        progress=True,
    )
    normalized_fragility = calculate_neural_fragility(raw_fragility)

    save_fragility_npz(
        args.data_output,
        raw_fragility,
        normalized_fragility,
        times=absolute_window_times,
        channel_names=raw.ch_names,
    )
    print(f"Saved fragility arrays to {args.data_output}")
    plot_heatmap(
        normalized_fragility,
        absolute_window_times,
        onset_time,
        raw.ch_names,
        args.output,
    )


if __name__ == "__main__":
    main()
