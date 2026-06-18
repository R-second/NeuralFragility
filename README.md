NeuralFragility
===============

This repository implements the neural fragility algorithm.

## Quick start

From this repository directory, install dependencies:

```
pip install -r requirements.txt
```

Run the bundled example from this directory:

```
python example_usage.py
```

The example writes `outputs/synthetic_fragility_heatmap.png` and prints the
computed heatmap shape. To compute without generating an image, run:

```
python example_usage.py --no-plot
```

Additional examples live in `examples/`. Paper-oriented numerical experiments
share matrices through `examples/matrix_examples.py`, so new experiments can
reuse the same matrix registry. Larger matrices are stored under
`examples/data/` as compressed NumPy `.npz` files and loaded with
`allow_pickle=False`.

Run the level-set behavior check used for paper figures:

```
python examples/level_set_behavior.py
```

The script writes `outputs/behaviour_check.eps`. You can select another matrix
registered in `examples/matrix_examples.py`:

```
python examples/level_set_behavior.py --matrix A0 --output outputs/behaviour_check_A0.eps
```

Run convergence analysis on random stable matrices:

```
python examples/convergence_analysis.py
```

The script writes `outputs/convergence_analysis.png`. For a quick smoke test,
reduce the workload:

```
python examples/convergence_analysis.py --trials 2 --size 8 --max-iter 3
```

Compare grid search against the proposed level-set method:

```
python examples/method_comparison.py
```

The script stores numeric results as compressed `.npz` files under
`outputs/comparison_results/` and writes PNG figures under `outputs/`. To
reproduce the combined comparison for several matrix sizes:

```
python examples/method_comparison.py --sizes 10 50 100
```

For a quick smoke test:

```
python examples/method_comparison.py --sizes 6 8 --trials 1 --grid-resolutions 20 40 --epsilons 1e-1 1e-3 --max-iter 3
```

EEG pipeline utilities live in `eeg_fragility.py`. They cover sliding windows,
ridge model fitting, fragility computation from precomputed transition matrices,
normalization, and `.npz` persistence. A small NumPy-only smoke example is
available:

```
python examples/openneuro_numpy_pipeline.py
```

For local OpenNeuro BrainVision files, keep raw data under the git-ignored
`data/` directory and run:

```
pip install -r requirements-openneuro.txt
python examples/openneuro_real_data_pipeline.py --max-windows 5
```

Set `--max-windows 0` to process all windows after a small trial succeeds.
If you want a small OpenNeuro-derived sample through MNE first:

```
python examples/download_mne_epilepsy_ecog.py
python examples/openneuro_real_data_pipeline.py --data-dir data --max-windows 2
```

If you want package-style imports from another directory, install this repo:

```
pip install -e .
```

API
---
- `NeuralFragility.fragility_algorithm.compute_neural_fragility(transition_matrix, channel_index, gamma=0.01)` - compute fragility for one transition matrix and channel.
- `NeuralFragility.eeg_fragility.compute_fragility_heatmap(eeg, fs, window_sec, step_sec)` - compute a heatmap from an `eeg` array with shape `(n_channels, n_times)`.
- `NeuralFragility.eeg_fragility.compute_fragility_from_matrices(transition_matrices)` - compute fragility from pre-estimated transition matrices with shape `(n_windows, n_channels, n_channels)`.

If you want integration with `openNeuroAnalysis.ipynb`, pass preprocessed
EEG arrays (channels x times) to `compute_fragility_heatmap`.
