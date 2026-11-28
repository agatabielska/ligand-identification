<h1 align="center">Ligand identification</h1>
<p align="center">An equivariant e3nn spectral model and a 1D radial-shell CNN, trained with probabilistic sampling and compared with MinkLoc.</p>

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and
sync the project dependencies:

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

### Thresholded data

Download and preprocess the thresholded blobs:

```bash
./src/scripts/download_cryoem_blobs.sh
uv run -m src.scripts.preprocess \
  --raw-data-dir cryoem_blobs \
  --output-folder data/thresholded \
  --transform-stack probabilistic
```

### Unthresholded data

Trim the unthresholded blobs, then preprocess them with probabilistic sampling:

```bash
uv run -m src.scripts.trim_unthresholded
uv run -m src.scripts.preprocess \
  --raw-data-dir NoThresholdBlobs \
  --output-folder data/unthresholded \
  --transform-stack probabilistic
```

## Run the default model

The default configuration (`configs/config.yaml`) trains the spectral model
with 2,000 points. It reads `data/unthresholded_transform_probabilistic/`,
which must contain `train/`, `val/`, and `test/` splits. Start training with:

```bash
uv run -m src.scripts.train
```

## Configure other runs

Use Hydra overrides on the training command, or edit the YAML presets in
`configs/`. The main settings are:

| Setting | Examples | Purpose |
|---|---|---|
| `model` | `spectral`, `radialCNN` | Select the model family |
| `paths.data_dir` | `data/unthresholded_transform_probabilistic` | Select dataset |
| `data.max_points` | `2000`, `4000`, `10000`, `25000` | Set the input point budget |
| `model.use_globals` | `all`, `none` | Enable or disable global features for the spectral model |
| `train.epochs`, `random_seed` | YAML values or overrides | Control training |
| `machine.batch_size`, `machine.use_gpu_loader` | YAML values or overrides | Control loading and memory use |

For example, run the coordinate-only RadialShellCNN on the default dataset:

```bash
uv run -m src.scripts.train model=radialCNN
```

### Model presets in `configs/model/`

Use the YAML filename (without `.yaml`) as the `model` override. `UP`, `TP`,
and `TU` stand for unthresholded probabilistic, thresholded probabilistic, and
thresholded uniform data.

| Model family | Validation-accuracy presets | Validation-loss presets |
|---|---|---|
| Spectral | `best_spectral_UP`, `best_spectral_TP`, `best_spectral_TU` | `best_spectral_UP_opt_val_loss` |
| RadialShellCNN | `best_radialCNN_UP`, `best_radialCNN_TP`, `best_radialCNN_TU` | `best_radialCNN_UP_opt_val_loss`, `best_radialCNN_TP_opt_val_loss`, `best_radialCNN_TU_opt_val_loss` |

`configs/hpo/` defines the search spaces. Unsuffixed presets maximize
`val_acc`. Presets with `_opt_val_loss` minimize validation cross-entropy.

## Ensembles

Ensemble compatible spectral checkpoints from run directories under
`checkpoints/`. The ensemble averages model probabilities and selects its
prior-correction parameter using validation data:

```bash
uv run -m src.models.e3nn.ensemble \
  model=spectral \
  +arch=spectral \
  +select=acc \
  '+runs=[run_directory_1,run_directory_2]'
```

Use `+select=recall` to select by macro recall subject to the validation
accuracy constraint.

## Export predictions

For finalized HPO checkpoint directories containing a `final-epoch=*.ckpt`,
export per-sample test predictions:

```bash
uv run -m src.scripts.evaluate_checkpoints
```

The script writes one CSV per checkpoint to `checkpoint_predictions/`. Each
CSV contains the sample identifier, true and predicted classes,
predicted-class probability, and 1-based rank of the true class. Override the
checkpoint, data, or output folders with `--checkpoints-dir`, `--data-root`,
and `--output-dir`.

## Prediction analysis

Analyze all prediction CSVs and write a separate report folder for each:

```bash
uv run python prediction_analysis/analyze_checkpoint_predictions.py
```

Reports are saved under `prediction_analysis/results/` and include accuracy
and mean true-class rank alongside class performance, confusions, and
confidence. These metrics can favor different runs; lower mean rank is better.
Use repeated `--csv <filename>` options to select inputs.

## Visualization

```bash
uv run streamlit run src/visualizations/blob_visualization.py
```
