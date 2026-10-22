# e3nn spectral model

This directory contains the retained spectral model used for ligand
identification. The model classifies volumetric ligand blobs represented as
weighted 3-D points. It is designed for the current dataset protocol:

1. load one sparse `.npz` blob;
2. keep its 2,000 highest-density points;
3. optionally normalize coordinates to a unit sphere;
4. convert the resulting point cloud into rotation-invariant spectral
   descriptors;
5. classify the descriptors into one of the ligand classes.

The implementation is intentionally small. The directory contains only three
model files:

| File | Responsibility |
| --- | --- |
| [`features.py`](./features.py) | Point-cloud featurization into spherical-harmonic coefficients. |
| [`spectral_net.py`](./spectral_net.py) | E3NN channel mixing and invariant power/bispectrum feature construction. |
| [`configurable_spectral.py`](./configurable_spectral.py) | PyTorch Lightning wrapper, global scalar branch, augmentation, metrics, and optimizer. |

The training entry points are outside this directory:

- [`src/scripts/train.py`](../../scripts/train.py) — the default strict
  2,000-point protocol, with Clifford still available as an alternative.
- [`src/scripts/train_raw_npz_scalars.py`](../../scripts/train_raw_npz_scalars.py)
  — the historical protocol that computes additional scalars from the
  complete raw NPZ blob.
- [`src/experiments/spectral_experiments.py`](../../experiments/spectral_experiments.py) — the exact BigNet experiment
  runner and ensemble report.
- [`ensemble.py`](./ensemble.py) — checkpoint ensemble evaluation with
  validation-selected prior adjustment.

## Why this model exists

The raw data is a sparse density field. A sample contains voxel coordinates
(`indices`) and density values (`values`), but the class should not depend on
the arbitrary ordering of those entries. It should also remain stable under
rotations of the ligand.

The spectral model addresses this by:

- treating density as a weighted point cloud;
- centering each cloud at its density-weighted center of mass;
- expanding radial neighborhoods in Gaussian shells;
- expanding angular structure in spherical harmonics;
- contracting those coefficients into rotational invariants;
- classifying the resulting invariant vector with an MLP.

The spherical-harmonic features retain angular shape information without
requiring a k-nearest-neighbor graph. The final classifier therefore receives
invariant features rather than coordinates in a fixed coordinate frame.

## Data representation

### Point tensor

The dataset loader returns a tensor with shape:

```text
(batch, 2000, 4)
```

The last dimension is:

```text
[x, y, z, density]
```

If a blob contains fewer than 2,000 selected points, the remaining rows are
zero padded. Padding is identified by `density == 0`. If it contains more,
the 2,000 highest-density points are retained by
[`make_top_density_transform`](../../scripts/train.py).

The e3nn spectral path is permutation invariant, so the selected points do
not need a canonical order. Clifford uses a separate ordered transform and
keeps its existing model-specific behavior.

### Coordinate normalization

The default configuration uses `train.normalize_data: true`. Each blob is
centered and divided by its maximum point radius before the top-density tensor
is passed to the spectral model. This makes the spectral representation focus
on normalized shape.

That normalization removes some absolute-size information. This was the
reason for testing optional global scalar features. The two retained
protocols differ exactly in how those scalars are obtained.

## Model pipeline

### 1. Spherical-harmonic coefficients

[`sh_coefficients`](./features.py) performs the following operations for each
batch:

1. mask padded points;
2. construct the configured channels;
3. normalize each channel over the point dimension;
4. compute the density-weighted center of mass;
5. express points relative to that center;
6. evaluate real spherical harmonics for degrees `0..lmax`;
7. weight them by Gaussian radial shells;
8. contract over points.

The output shape is:

```text
(batch, n_shells * n_channels, (lmax + 1)^2)
```

The default uses:

```text
n_shells   = 4
lmax       = 6
n_channels = 2
```

The two channels are density and occupancy. The implementation also supports
a third density-squared channel for experiments.

The featurizer uses FP32 for spherical harmonics and tensor contractions.
This is deliberate: these operations are numerically more sensitive than the
final MLP and should not be silently evaluated in reduced precision.

### 2. Equivariant channel mixing

[`SpectralNet`](./spectral_net.py) first groups coefficients by angular
degree and applies an E3NN linear map:

```text
K * C input channels  ->  n_mix mixed channels
```

The default is:

```text
n_mix = 32
```

Optional E3NN input and mixing batch normalization layers are available.
The retained `r3_bn_m32t` configuration uses:

```text
in_bn  = false
mix_bn = true
norm   = batch
```

### 3. Invariant feature construction

The mixed coefficient tensors are converted into invariant descriptors.

#### Power spectrum

For each angular degree, channel pairs are contracted with themselves. The
default `pow_mode: triu` keeps the upper-triangular channel-pair products,
including cross-channel terms.

The alternative `pow_mode: diag` keeps only same-channel terms and was tested
as `r2_bn_m32d`. It is useful when reducing feature count or memory.

#### Bispectrum

For allowed triples `(l1, l2, l3)`, the model contracts three angular
coefficient tensors with Wigner-3j coefficients. The default is:

```text
bispec_lmax = 6
triples     = nonzero
bispec_mode = batched
```

`triples: nonzero` removes contractions that are identically zero because of
Wigner-3j antisymmetry. The available implementations are:

- `loop` — simple one-contraction-per-triple reference implementation;
- `batched` — default padded batched contraction;
- `sparse` — gathers nonzero Wigner-3j entries and uses indexed accumulation.

#### Gram features

The retained configurations use:

```text
gram: none
```

The core still supports density-only, all-channel, and full Gram variants for
controlled experiments. They are not part of the default two-protocol
reproduction.

#### Signed logarithm

Invariant values are transformed using:

```text
sign(x) * log1p(abs(x))
```

This reduces the dynamic range while preserving the sign of parity-sensitive
invariants.

### 4. Classifier and global branch

The default spectral invariant vector is passed through:

```text
normalization -> Linear(512) -> GELU -> Dropout(0.2)
             -> Linear(256) -> GELU
             -> Linear(number_of_classes)
```

The optional global branch is:

```text
global scalars -> BatchNorm1d -> Linear(12, 64) -> GELU
```

Its output is concatenated with the spectral invariant vector before the
classifier head.

## The two retained protocols

Both protocols use the same spectral architecture and the same 2,000-point
tensor for spherical-harmonic features. They are **not information-equivalent**
because their optional global scalars have different sources.

### Protocol A: strict 2,000-point model

Configuration:

[`configs/model/points-2000-r3.yaml`](../../../configs/model/points-2000-r3.yaml)

Model name:

```text
points-2000-r3
```

The 12 global scalars are computed from the normalized and padded 2,000-point
tensor. No discarded raw points are read for feature extraction.

This is the default training protocol:

```bash
uv run -m src.scripts.train
```

The equivalent explicit command is:

```bash
uv run -m src.scripts.train model=points-2000-r3
```

The default model settings are:

```text
n_shells          = 4
lmax              = 6
r_max             = 1.0
n_channels        = 2
n_mix             = 32
pow_mode          = triu
gram              = none
triples           = nonzero
bispec_lmax       = 6
bispec_mode       = batched
norm              = batch
in_bn             = false
use_globals       = all
glob_dim          = 64
glob_noise        = 0.0
widths            = [512, 256]
head_dropout      = 0.2
feat_dropout      = 0.0
point_dropout     = 0.1
density_noise     = 0.0
clean_probability = 0.125
label_smoothing   = 0.05
learning_rate     = 3e-3
weight_decay      = 5e-2
epochs            = 15
```

This is the correct protocol when the requirement is that every feature must
come from the 2,000-point representation.

### Protocol B: historical raw-NPZ scalar model

Configuration:

[`configs/model/raw-npz-scalars.yaml`](../../../configs/model/raw-npz-scalars.yaml)

Model name:

```text
raw-npz-scalars
```

Run it with:

```bash
uv run -m src.scripts.train_raw_npz_scalars
```

or:

```bash
uv run -m src.scripts.train model=raw-npz-scalars
```

The spectral features still use the 2,000-point normalized tensor. However,
the additional 12 scalar features are computed from the complete sparse NPZ
blob, before point selection.

The 12 raw-NPZ scalars are:

| Index | Feature |
| ---: | --- |
| 0 | `log_n_all`: logarithm of the complete raw point count |
| 1 | `log_sum_rho`: logarithm of total density |
| 2 | `log_max_rho`: logarithm of maximum density |
| 3 | `r_max`: maximum distance from the density-weighted center |
| 4 | `r_g`: density-weighted root-mean-square radius |
| 5 | `r_mean`: density-weighted mean radius |
| 6–8 | square roots of covariance eigenvalues |
| 9–11 | sorted bounding-box side lengths |

This protocol preserves absolute size and information about points discarded
by the 2,000-point representation. It is therefore expected to differ from
the strict protocol and must not be described as a pure 2,000-point-only
experiment.

The raw scalar calculation is implemented in
[`src/experiments/spectral_experiments.py`](../../experiments/spectral_experiments.py)
as `raw_npz_globals`. The loader
checks that the raw scalar sample order and labels match the point-tensor
dataset before training.

## Training procedure

The retained BigNet procedure is used for both protocols.

### Augmentation

During training:

- each point density is independently dropped with probability `0.1`;
- optional multiplicative log-normal density noise is available;
- with probability `0.125`, a sample remains clean instead of receiving the
  sampled point augmentation.

Global scalars are computed from the clean input tensor. This prevents
point-dropout augmentation from changing the size signal in the same step.
When configured, `glob_noise` is added to standardized global features during
training only.

### Optimization

The optimizer is AdamW:

```text
learning rate = 3e-3
weight decay  = 5e-2
```

The learning rate uses:

1. a short linear warmup covering approximately 3% of optimization steps;
2. cosine decay to 1% of the initial learning rate.

The loss is cross-entropy with label smoothing `0.05`.

### Validation and checkpointing

The best checkpoint is the epoch with the lowest validation loss. Test data is
evaluated only after restoring that checkpoint. Validation is never merged
with training data.

This matters because the experiment selects models and any ensemble threshold
on validation data. Test metrics are reporting metrics, not selection inputs.

## Results and experiment history

The project went through several controlled comparisons.

### Initial configurable implementation

The first configurable Lightning implementation did not reproduce the
historical result. Investigation found protocol differences, including:

- different global-feature source;
- different standardization path;
- mismatched augmentation behavior;
- different batch size;
- different global-branch handling;
- different checkpoint/training-loop details.

The implementation was aligned with the BigNet procedure rather than
silently presenting the lower result as equivalent.

### Historical raw-NPZ result

The previously reported winner was:

```text
r3_bn_m32t
seed 200
validation accuracy  0.7900
validation loss      0.8308
test accuracy        0.8021
test macro recall    0.5149
parameters           approximately 2.67M
```

That result used the raw-NPZ 12 scalar protocol. It was not a strict
2,000-point-only result.

### Fresh strict 2,000-point comparison

A fresh no-cache comparison was then run after removing full-NPZ scalar
extraction. The strict `r3_bn_m32t` result was approximately:

```text
test accuracy      0.7808
test macro recall  0.4796
```

The full eight-model strict ensemble reached approximately:

```text
test accuracy      0.7950
test macro recall  0.4906
```

The strict results are lower than the raw-NPZ result because the strict
protocol deliberately removes information carried by discarded points and
absolute raw blob size.

These numbers are historical reference results, not guarantees. Re-running
with a different seed, dependency version, GPU, or data revision can change
the metrics.

## Ensemble evaluation

For Lightning checkpoints from the retained spectral protocols, use:

```bash
uv run -m src.models.e3nn.ensemble \
  model=points-2000-r3 \
  +arch=points-2000-r3 \
  +select=acc \
  '+runs=[run_directory_1,run_directory_2]'
```

Replace the run directory names with directories under `checkpoints/` that
contain `model-*.ckpt` files.

The exact runner supports heterogeneous model pools and validation-selected
prior adjustment. Run selected strict configurations with:

```bash
uv run -m src.experiments.spectral_experiments \
  --only r1_gall_bn,r2_bn_m32d,r3_bn_m32t \
  --seeds 1 \
  --force
```

The runner reports:

- per-model validation and test metrics;
- POOL-all;
- POOL-top3;
- thresholds `tau = 0.00, 0.25, 0.50, 0.75`;
- the threshold selected from validation metrics.

For a prediction probability vector `p` and smoothed class prior `log_prior`,
the adjusted prediction is:

```text
argmax(log(p) - tau * log_prior)
```

The test split is not used to choose `tau`.

Generated caches and checkpoints are not required by the current strict
runner. The runner computes spectral and strict global features directly from
the loaded 2,000-point tensors. Existing output files can be reused by the
runner, while `--force` retrains selected runs.

## Configuration and overrides

The default Hydra configuration is:

[`configs/config.yaml`](../../../configs/config.yaml)

It selects:

```yaml
defaults:
  - model: points-2000-r3
```

The raw-NPZ entry point uses:

[`configs/config_raw_npz_scalars.yaml`](../../../configs/config_raw_npz_scalars.yaml)

Examples:

```bash
# Use a different seed for a reproducibility run.
uv run -m src.scripts.train random_seed=200

# Change spectral channel mixing.
uv run -m src.scripts.train model.n_mix=64 model.pow_mode=diag

# Use geometry-only global features.
uv run -m src.scripts.train model.use_globals=geom

# Disable optional globals entirely.
uv run -m src.scripts.train model.use_globals=none

# Change data location.
uv run -m src.scripts.train \
  paths.data_dir=data/unthresholded_transform_probabilistic

# Run the raw-NPZ protocol with a named W&B run.
uv run -m src.scripts.train_raw_npz_scalars \
  wandb.run_name=raw_scalars_s200
```

The data directory must contain:

```text
data_dir/
  train/
  val/
  test/
```

Each split contains one directory per class, with sparse `.npz` files inside.
The class-to-index mapping is built from the union of class directories and is
shared across all three splits.

## Numerical and implementation safeguards

The current code includes several safeguards added during debugging:

- a hard `MAX_POINTS = 2000` invariant;
- validation that `train/`, `val/`, and `test/` exist before logger setup;
- safe normalization for zero-radius point clouds;
- rejection of non-finite normalized coordinates;
- non-finite global-feature checks;
- covariance regularization before eigendecomposition;
- FP32 feature extraction under mixed-precision training;
- chunked feature and global-stat computation;
- explicit raw-scalar label-order validation;
- train-split-only standardization of global features;
- GPU-loader fallback to pinned CPU memory on CUDA out-of-memory.

If a CUDA eigendecomposition error reports NaNs, inspect the input data and
normalization first. Do not work around it by silently replacing non-finite
values with arbitrary features.

## Removed and superseded components

The e3nn directory previously contained several overlapping implementations:

- an older equivariant point-cloud model hierarchy;
- separate spectral and baseline wrappers;
- an experimental graph model;
- a simple wrapper used by earlier experiments.

They were removed because they were not part of the retained best-run
protocols and made it unclear which implementation produced a result. The
active model path now consists of the three files listed at the top of this
document.

Likewise, cache-generation, sweep, probe, and seed-launcher scripts that were
not needed for the two retained protocols were removed. The historical logic
needed to reproduce the raw-scalar and strict BigNet comparisons remains in
[`src/experiments/spectral_experiments.py`](../../experiments/spectral_experiments.py).

## Reproducibility checklist

Before comparing a new run with a historical result, record:

1. the selected protocol (`points-2000-r3` or `raw-npz-scalars`);
2. the exact dataset directory and data revision;
3. the random seed;
4. the model YAML and any Hydra overrides;
5. PyTorch, e3nn, Lightning, and CUDA versions;
6. whether the run used `--force` or reused an existing output;
7. whether metrics are for one model or an ensemble;
8. the ensemble `tau` selection rule.

Most importantly, never compare the historical `0.8021` raw-NPZ result
directly with a strict 2,000-point result without labeling the protocol.
