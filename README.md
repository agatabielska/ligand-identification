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

Change model and training values in the YAML files under `configs/` before
starting a run. The command remains:

```bash
uv run -m src.scripts.train
```

Train the complete-input scalar variant:

```bash
uv run -m src.scripts.train --config-name config_global_statistics_e3nn
```

Train the Clifford model:

```bash
uv run -m src.scripts.train model=clifford
```

Check available configuration values:

```bash
uv run -m src.scripts.train --help
```

## Point-count experiment

Compares 2k / 4k / 10k / 25k points per blob, with global features
(`model=spectral`) and without (`model=spectral_no_globals`, which is
`spectral.yaml` with `use_globals: none`).

```bash
# 1. preprocess all point counts
for N in 2000 4000 10000 25000; do
  uv run -m src.scripts.preprocess --raw-data-dir NoThresholdBlobs \
    --output-folder data/unthresholded --transform-stack probabilistic --max-points $N
done

# 2. train (10k+: data kept in CPU RAM; 25k: smaller batch)
for N in 2000 4000 10000 25000; do
  for M in spectral spectral_no_globals; do
    BS=256; GPU=true
    [ $N -ge 10000 ] && GPU=false
    [ $N -eq 25000 ] && BS=64
    uv run -m src.scripts.train model=$M data.max_points=$N \
      paths.data_dir=data/unthresholded_transform_probabilistic_${N}pts \
      machine.batch_size=$BS machine.use_gpu_loader=$GPU \
      wandb.run_name=${M}_${N}pts
  done
done

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

For larger GPU-oriented models while keeping exactly 2,000 points:

```bash
uv run -m src.experiments.gpu_scaling_experiments \
  --preset baseline,head-wide,head-deep,mix48,mix64 \
  --batch-size 256 \
  --epochs 20 \
  --force
```

The sweep starts from the measured `r3_bn_m32t` configuration and expands one
capacity axis at a time. Available presets include wider/deeper heads,
additional channel mixing, more radial shells, and a third input channel.
Results are written to `results/gpu_scaling_2000/`.

## Evaluate a checkpoint ensemble

The checkpoint ensemble evaluator expects run directories below `checkpoints/`,
each containing `model-*.ckpt` files:

```bash
uv run -m src.models.e3nn.ensemble \
  model=spectral \
  +arch=spectral \
  +select=acc \
  '+runs=[run_directory_1,run_directory_2]'
```

Use only checkpoints produced by the same configurable spectral model family.

To reuse saved strict-protocol experiment outputs without retraining:

```bash
uv run -m src.experiments.spectral_experiments --report-only
```

The complete-input scalar protocol can also be ensembled. Run it with
different seeds by changing `random_seed` in
`configs/config_global_statistics_e3nn.yaml`; each run is saved under
`results/raw_npz_scalars/`. Then report the saved ensemble with:

```bash
uv run -m src.experiments.spectral_experiments \
  --report-only \
  --output-dir results/raw_npz_scalars
```

## Visualize blobs

```bash
uv run streamlit run src/visualizations/blob_visualization.py
```