NeuralFragility
===============

This small package exposes the `sreedhar_alg` implementation and
provides utilities to compute Neural Fragility heatmaps from EEG-like
time series.

Quick start
-----------

From this repository directory, install dependencies (recommended in a virtualenv):

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
reuse the same matrix registry.

Run the level-set behavior check used for paper figures:

```
python examples/paper_level_set_behavior.py
```

The script writes `outputs/behaviour_check.eps`. You can select another matrix
registered in `examples/matrix_examples.py`:

```
python examples/paper_level_set_behavior.py --matrix A0 --output outputs/behaviour_check_A0.eps
```

If you want package-style imports from another directory, install this repo:

```
pip install -e .
```

API
---
- `NeuralFragility.sreedhar_alg.neural_fragility_inf(A, k, gamma=0.01)` - compute fragility for matrix `A` and channel `k`.
- `NeuralFragility.fragility_from_eeg.compute_fragility_heatmap(eeg, fs, window_sec, step_sec)` - compute heatmap from `eeg` array (n_channels, n_times).

If you want integration with `openNeuroAnalysis.ipynb`, pass preprocessed
EEG arrays (channels x times) to `compute_fragility_heatmap`.
