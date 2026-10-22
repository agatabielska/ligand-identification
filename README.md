# Ligand identification

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then
install the project dependencies:

```bash
uv sync
```

The default configuration expects this dataset layout:

```text
data/unthresholded_transform_probabilistic/
├── train/
├── val/
└── test/
```

## Prepare data

Download and preprocess the Cryo-EM blobs:

```bash
./src/scripts/download_cryoem_blobs.sh
uv run -m src.scripts.preprocess \
  --raw-data-dir cryoem_blobs \
  --output-folder data/thresholded \
  --transform-stack probabilistic
```

For unthresholded blobs:

```bash
uv run -m src.scripts.trim_unthresholded
uv run -m src.scripts.preprocess \
  --raw-data-dir NoThresholdBlobs \
  --output-folder data/unthresholded \
  --transform-stack probabilistic
```

## Train one model

The default command trains the strict 2,000-point spectral model:

```bash
uv run -m src.scripts.train
```

Override configuration values with Hydra:

```bash
uv run -m src.scripts.train \
  model.n_mix=64 \
  model.pow_mode=diag \
  train.epochs=30
```

Train the raw-NPZ scalar variant:

```bash
uv run -m src.scripts.train_raw_npz_scalars
```

Train the Clifford model:

```bash
uv run -m src.scripts.train model=clifford
```

Check available configuration values:

```bash
uv run -m src.scripts.train --help
```

## Run spectral experiments

Run selected strict 2,000-point presets:

```bash
uv run -m src.experiments.spectral_experiments \
  --only r1_gall_bn,r2_bn_m32d,r3_bn_m32t \
  --seeds 1 \
  --force
```

Run the complete preset set:

```bash
uv run -m src.experiments.spectral_experiments --force
```

Results are written to `results/best_e2e/`. The runner prints individual
metrics and validation-selected `POOL all` and `POOL top3` ensemble metrics.

## Evaluate a checkpoint ensemble

The checkpoint ensemble evaluator expects run directories below `checkpoints/`,
each containing `model-*.ckpt` files:

```bash
uv run -m src.models.e3nn.ensemble \
  model=points-2000-r3 \
  +arch=points-2000-r3 \
  +select=acc \
  '+runs=[run_directory_1,run_directory_2]'
```

Use only checkpoints produced by the same configurable spectral model family.

To reproduce the saved strict-protocol ensemble without retraining:

```bash
uv run python -c \
'from src.experiments.spectral_experiments import report, CONFIGS, OUT_DIR; report(list(CONFIGS), OUT_DIR, "acc")'
```

## Visualize blobs

```bash
uv run streamlit run src/visualizations/blob_visualization.py
```
