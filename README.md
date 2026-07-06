NeuralFragility
===============

This repository implements the neural fragility algorithm.

## Requirements

### Minimal requirements

For synthetic examples and numerical experiments, install:

```bash
pip install -r requirements.txt
```

The minimal dependencies are:

```text
numpy
scipy
matplotlib
```

Python `>=3.10` is recommended.

### Additional requirements for real EEG examples

For real data examples, also install:

```bash
pip install -r requirements-openneuro.txt
```

The additional dependencies are:

```text
mne
pandas
tqdm
```

## Quick start

Clone the repository and install the minimal dependencies:

```bash
git clone https://github.com/R-second/NeuralFragility.git
cd NeuralFragility

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run the simplest bundled example:

```bash
python example_usage.py
```

This generates a synthetic EEG-like signal, computes the neural fragility heatmap, and saves:

```text
outputs/synthetic_fragility_heatmap.png
```

If you only want to check whether the computation works without saving a figure, run:

```bash
python example_usage.py --no-plot
```


## Basic usage from Python

After installing the package, you can directly call the core functions from Python.

This section shows minimal usage examples. More complete examples and executable scripts are introduced in the following sections.

Also, This basic usage examples are collected in one executable script:

```bash
python examples/basic_usage.py
```

### Compute stability radius for one matrix and one channel

Use `compute_stability_radius` when you already have a transition matrix and want to compute the column pertubation stability radius value of a specific channel.

```python
import numpy as np

from NeuralFragility.fragility_algorithm import compute_stability_radius

# Example transition matrix
A = np.array([
    [0.8, 0.1, 0.0],
    [0.0, 0.7, 0.2],
    [0.1, 0.0, 0.6],
])

# Compute fragility for channel 0
value, theta, _ = compute_stability_radius(
    transition_matrix=A,
    channel_index=0,
    gamma=0.01,
    method="branch filtering method",
    # method="grid search"
)

print(value)
```

Here, `channel_index` specifies which channel is perturbed and `method` specifies which algorithm to use. The default is the branch filtering method, which is more accurate than the grid search method. The grid search method is also available for comparison.

### Compute a fragility heatmap from EEG-like data

Use `compute_neural_fragility_heatmap` when you have multichannel time series data and want to compute a time-by-channel fragility heatmap.

```python
import numpy as np

from NeuralFragility.eeg_fragility import compute_neural_fragility_heatmap

# Example EEG-like data
# Shape: n_channels x n_times
eeg = np.random.randn(4, 2000)

# Sampling frequency in Hz
fs = 200

heatmap, times = compute_neural_fragility_heatmap(
    eeg=eeg,
    fs=fs,
    window_sec=0.25,
    step_sec=0.125,
)

print(heatmap.shape)
print(times.shape)
```

### Compute fragility from pre-estimated transition matrices

Use `compute_fragility_from_matrices` when you have already estimated transition matrices for each time window.

```python
import numpy as np

from NeuralFragility.eeg_fragility import compute_fragility_from_matrices

# Example transition matrices
# Shape: n_windows x n_channels x n_channels
transition_matrices = np.random.randn(10, 4, 4) * 0.1

fragility = compute_fragility_from_matrices(
    transition_matrices=transition_matrices,
    gamma=0.01,
)

print(fragility.shape)
```

### When to use each function

| Function | Use case |
|---|---|
| `compute_stability_radius` | Compute stability radius for one transition matrix and one channel |
| `compute_neural_fragility_heatmap` | Start from EEG-like time series data and compute a full heatmap |
| `compute_stability_radius_from_matrices` | Start from pre-estimated transition matrices and compute stability radius values |

For complete runnable examples, see the examples introduced below.


## Examples overview

The examples are divided into four groups:

1. Basic usage examples
2. Synthetic data examples
3. Numerical experiment examples
4. OpenNeuro / real EEG examples

| Category | Script | Purpose | 
|---|---|---|
| Basic usage | `examples/basic_usage.py` | README basic usage examples |
| Synthetic data | `example_usage.py` | Minimal working example | 
| Synthetic data | `examples/synthetic_heatmap.py` | Direct synthetic heatmap example | 
| EEG pipeline | `examples/openneuro_numpy_pipeline.py` | NumPy-only OpenNeuro-style pipeline | 
| Numerical experiment | `examples/level_set_behavior.py` | Level-set behavior visualization | 
| Numerical experiment | `examples/convergence_analysis.py` | Convergence analysis on random stable matrices | 
| Numerical experiment | `examples/method_comparison.py` | Grid search vs branch filtering method comparison | 
| Real EEG data | `examples/download_mne_epilepsy_ecog.py` | Download a small MNE epilepsy ECoG sample | 
| Real EEG data | `examples/openneuro_real_data_pipeline.py` | Run the BrainVision / OpenNeuro-style pipeline | 


## Example details

### `examples/synthetic_heatmap.py`

This script directly generates a synthetic multichannel time series, computes a fragility heatmap, and saves a figure.

Run:

```bash
python examples/synthetic_heatmap.py
```

You can also change basic parameters:

```bash
python examples/synthetic_heatmap.py \
    --output path/to/heatmap.png
```

### `examples/openneuro_numpy_pipeline.py`

This is a NumPy-only OpenNeuro-style pipeline.

It performs:

1. Synthetic EEG generation
2. Sliding-window segmentation
3. Transition matrix estimation by ridge regression
4. Neural fragility computation
5. Normalization
6. `.npz` saving
7. Heatmap plotting

Run:

```bash
python examples/openneuro_numpy_pipeline.py
```

You can also change basic parameters:

```bash
python examples/openneuro_numpy_pipeline.py \
  --channels 8 \
  --samples 1000 \
  --fs 200 \
  --window-ms 250 \
  --step-ms 125
```

This example is recommended before using real EEG data.


### `examples/level_set_behavior.py`

This script visualizes the behavior of the level-set method.

Run:

```bash
python examples/level_set_behavior.py
```

The default output is:

```text
outputs/behaviour_check.png
```

You can also choose a matrix registered in `examples/matrix_examples.py`:

```bash
python examples/level_set_behavior.py \
  --matrix A0 \
  --output outputs/behaviour_check_A0.png
```


### `examples/convergence_analysis.py`

This script studies convergence behavior on random stable matrices.

Default execution:

```bash
python examples/convergence_analysis.py
```

For a quick smoke test:

```bash
python examples/convergence_analysis.py --trials 2 --size 8 --max-iter 3
```

The default output is:

```text
outputs/convergence_analysis.png
```

The full default setting may take longer, so it is better to start with the smoke test.


### `examples/method_comparison.py`

This script compares grid search with the branch filtering method.

Default execution:

```bash
python examples/method_comparison.py
```

The script saves numerical results under:

```text
outputs/comparison_results/
```

and writes figures under:

```text
outputs/
```

For a small smoke test:

```bash
python examples/method_comparison.py \
  --sizes 6 8 \
  --trials 1 \
  --grid-resolutions 20 40 \
  --epsilons 1e-1 1e-3 \
  --max-iter 3
```

For a larger comparison over several matrix sizes:

```bash
python examples/method_comparison.py --sizes 10 50 100
```


## OpenNeuro / real EEG pipeline

### Expected data format

The real-data pipeline expects local BrainVision-style files under `data/` by default.

The required files are:

```text
*_ieeg.vhdr
*_events.tsv
*_channels.tsv
```

For example:

```text
data/
  sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg.vhdr
  sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_events.tsv
  sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_channels.tsv
```

The `data/` directory is intended for local raw data and should not be committed to git.


You can use MNE to download a small epilepsy ECoG sample:

```bash
python examples/download_mne_epilepsy_ecog.py
```

### Run a small real-data test

```bash
python examples/openneuro_real_data_pipeline.py --max-windows 5
```

This limits the analysis to the first few windows, which is useful for checking whether preprocessing and fragility computation work.

### Process all windows

```bash
python examples/openneuro_real_data_pipeline.py --max-windows 0
```

### Specify a data directory

```bash
python examples/openneuro_real_data_pipeline.py \
  --data-dir data \
  --max-windows 5
```

### Specify a subject/file prefix

If multiple `*_ieeg.vhdr` files are found, pass the prefix explicitly:

```bash
python examples/openneuro_real_data_pipeline.py \
  --data-dir data \
  --subject-prefix sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01 \
  --max-windows 5
```
