# Spectral model

This directory contains the spectral model used to classify ligand blobs.
It converts each sparse 3-D density blob into rotation-invariant features and
then predicts one of the ligand classes.

## Data flow

```text
sparse .npz blob
      |
      v
coordinates + density values
      |
      v
center and scale coordinates
      |
      v
fixed tensor: (2000 points, 4 values)
             [x, y, z, density]
      |
      v
Gaussian radial shells + spherical harmonics
      |
      v
E3NN channel mixing
      |
      v
power spectrum + bispectrum invariants
      |
      +----------------------+
      |                      |
      v                      v
spectral features       optional scalar features
      |                      |
      +----------+-----------+
                 v
          classifier MLP
                 |
                 v
          ligand class
```

The point representation always has exactly 2,000 rows. Blobs with fewer
points are padded with zero-density rows. The model ignores the order of the
points and uses density-zero rows as padding.

## Files

| File | Purpose |
| --- | --- |
| [`features.py`](./features.py) | Converts points into spherical-harmonic coefficients. |
| [`spectral_net.py`](./spectral_net.py) | Builds invariant power-spectrum and bispectrum features. |
| [`configurable_spectral.py`](./configurable_spectral.py) | Training model, scalar branch, metrics, and optimization. |
| [`ensemble.py`](./ensemble.py) | Evaluates compatible checkpoint ensembles. |

Data transforms and loaders are in
[`src/pipeline/data_loader.py`](../../pipeline/data_loader.py).
The unified entry point is [`src/scripts/train.py`](../../scripts/train.py).
It contains the config-driven training orchestration for both Clifford and
E3NN. The experiment scripts are separate sweep and reporting tools.

## Spectral features

For each point cloud, the featurizer:

1. masks padded points;
2. creates density and occupancy channels;
3. computes a density-weighted center;
4. evaluates spherical harmonics in radial Gaussian shells;
5. contracts over points.

The default representation is:

```text
n_shells   = 4
lmax       = 6
n_channels = 2
```

The resulting coefficients are mixed by an E3NN linear layer with
`n_mix = 32`. They are converted into:

- power-spectrum products between angular channels;
- bispectrum contractions using Wigner-3j coefficients.

The default invariant settings are:

```text
pow_mode     = triu
triples      = nonzero
bispec_lmax  = 6
bispec_mode  = batched
gram         = none
```

The classifier uses batch normalization followed by hidden layers
`[512, 256]` and a final class-prediction layer.

## Optional scalar features

The model can append 12 rotation-invariant scalar features to the spectral
features. They describe counts, density totals, radii, covariance eigenvalues,
and bounding-box dimensions.

Scalar values are calculated independently for every sample in a tensor batch.
The batch is only a vectorized execution unit; one sample does not affect
another sample's scalar values.

During training, scalar standardization is fitted once from the complete
training split. Validation and test samples use the training mean and
standard deviation.

The available selections are configured with `model.use_globals`:

```text
none   no scalar branch
count  count and density-total features
geom   geometry features
all    all 12 features
```

## Training protocols

### Strict 2,000-point protocol

This is the default. Both spectral features and optional scalar features use
the normalized, padded 2,000-point tensor.

Configuration:

[`configs/model/spectral.yaml`](../../../configs/model/spectral.yaml)

Run:

```bash
uv run -m src.scripts.train
```

### Complete-input scalar protocol

The spectral features still use the 2,000-point tensor, but the optional
12 scalars are calculated from the complete sparse input blob before point
selection. This protocol uses information that is not present in the strict
2,000-point tensor.

Configuration:

[`configs/config_global_statistics_e3nn.yaml`](../../../configs/config_global_statistics_e3nn.yaml)

Run:

```bash
uv run -m src.scripts.train --config-name config_global_statistics_e3nn
```

The two protocols must be reported separately.

## Configuration

Keep training settings in the YAML files under [`configs/`](../../../configs/).
The normal commands do not require model overrides on the command line.

The default data layout is:

```text
data/unthresholded_transform_probabilistic/
├── train/
├── val/
└── test/
```

Each split contains one directory per class with sparse `.npz` files.

Important default settings are in:

- [`configs/config.yaml`](../../../configs/config.yaml)
- [`configs/model/spectral.yaml`](../../../configs/model/spectral.yaml)
- [`configs/config_global_statistics_e3nn.yaml`](../../../configs/config_global_statistics_e3nn.yaml)

## Ensembles

For Lightning checkpoints produced by the same configurable spectral model,
use directories below `checkpoints/` that contain `model-*.ckpt` files:

```bash
uv run -m src.models.e3nn.ensemble \
  model=spectral \
  +arch=spectral \
  +select=acc \
  '+runs=[run_directory_1,run_directory_2]'
```

The evaluator selects the class-prior adjustment parameter on validation data
and reports the resulting test metrics.

For the saved strict spectral experiment outputs:

```bash
uv run -m src.experiments.spectral_experiments --report-only
```

## Related experiment runner

[`src/experiments/spectral_experiments.py`](../../experiments/spectral_experiments.py)
contains the reproducible strict 2,000-point experiment presets and ensemble
reporting. Results are written to `results/best_e2e/`.
