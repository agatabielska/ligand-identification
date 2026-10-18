# e3nn models

Rotation-invariant classifiers for ligand density blobs (97 classes).
Input: the 2000 highest-density points of a blob, `[x, y, z, density]`, zero-padded.
Data: `data/unthresholded_transform_probabilistic` (train 41,105 / val 12,280 / test 13,172).

## Models

| `model=` | Class | File | Status |
|---|---|---|---|
| `spectral` | `SpectralModel` | `spectral_model.py` | **Best results** |
| `spectral_baseline` | `BaselineSpectralModel` | `baseline_spectral_model.py` | Original v1; reference |
| `e3nn` | `E3NNPointCloudModel` | `model.py` | Equivariant network; also the base class (featurizer) of the spectral models |
| `e3nn_graph` | `E3NNGraphModel` | `graph_model.py` | Experimental, kNN-graph network; no results recorded |

### How the spectral models work

1. **Featurizer** (parameter-free, in `model.py`): the point cloud is projected on radial shells (`n_shells=16`) and spherical harmonics up to `lmax=6`, for 3 input channels (density, occupancy, density²).
2. **Learned channel mix** (`n_mix` channels per degree `l`): an equivariant linear map that acts within each `l`.
3. **Rotation-invariant features:** a power spectrum (Gram matrix per degree) and a bispectrum (Wigner-3j couplings of `l1 <= l2 <= l3`, both parities, so chirality is preserved), plus a Gram matrix across shells. All are passed through a signed log, `sign(f) * log1p(|f|)`.
4. **Head:** LayerNorm, then an MLP `d_in -> 1024 -> 512 -> classes` with GELU and dropout 0.2.

`SpectralModel` has opt-in flags on top of the baseline (all off by default, which gives the baseline):

| Flag | Effect |
|---|---|
| `use_globals` | adds 9 global invariants (point count, total/max density, radius of gyration, inertia eigenvalues) |
| `gram_all` | shell Gram for all 3 input channels instead of density only |
| `feat_dropout` | dropout on the feature vector before the head |
| `density_noise` | log-normal multiplicative noise on densities during training |

Training uses plain cross-entropy. Validation and test also report accuracy and macro recall after **post-hoc logit adjustment**, `argmax(log p - tau * log prior)`, for `tau` in 0.25, 0.5, 0.75, 1.0. `tau = 0` is no correction. The correction costs no retraining.

## Best commands

Common settings (all runs below, train-only, no validation data in training):

```bash
B="paths.data_dir=data/unthresholded_transform_probabilistic machine.batch_size=256 machine.num_workers=8 \
   train.learning_rate=1e-3 train.weight_decay=5e-2 train.epochs=15 train.early_stopping.patience=15"
```

**Best single model** (`n_mix=32`):
```bash
uv run -m src.scripts.train $B model=spectral model.n_mix=32 wandb.run_name=spec_v5_M
```
Add `random_seed=<n>` and a new `wandb.run_name` for more seeds. A run takes about 2.5 minutes on one L4.

**Best ensemble** (8 models: 3 seeds of `M`, 2 of `R`, 3 of `S1`; softmax probabilities are averaged):
```bash
uv run -m src.scripts.ensemble_multi paths.data_dir=data/unthresholded_transform_probabilistic \
  model=spectral machine.batch_size=256 machine.num_workers=8 \
  '+runs=[spec_v5_M,spec_v5_M_s0,spec_v5_M_s1,spec_v5_R_s1,spec_v5_R_s2,spec_v5_S1,spec_v5_S1_s1,spec_v5_S1_s2]'
```
For each run the script loads the checkpoint with the lowest `val_loss` from `checkpoints/<run_name>/`.
The `R` and `S1` variants differ from `M` in settings that are stored in each run's wandb config.

## Results

All numbers are train-only. Checkpoints are chosen by validation loss, tau by validation, and test is reported once. Tau 0 means no logit adjustment.

| Model | Val acc | Test acc | Test macro recall |
|---|---|---|---|
| single `M` (`n_mix=32`) | 0.7603 | 0.7536 | 0.4117 |
| single, other seeds and variants (`M`, `R`, `S1`, `n_mix=64`; 11 runs) | 0.755 – 0.762 | 0.744 – 0.754 | 0.390 – 0.420 |
| ensemble of 6 (`M` seeds + `n_mix=64`) | 0.7700 | 0.7613 | 0.4204 |
| **ensemble of 8 (`M`, `R`, `S1`)** | **0.7701** | **0.7632** | 0.4132 |

Effect of `tau` on the 8-model ensemble:

| tau | Val acc | Test acc | Test macro recall |
|---|---|---|---|
| 0.00 | 0.7701 | 0.7632 | 0.4132 |
| 0.25 | 0.7663 | 0.7606 | 0.4700 |
| 0.50 | 0.7429 | 0.7420 | 0.5240 |
| 0.75 | 0.3572 | 0.3819 | 0.4951 |

Takeaways:
- **For accuracy, use `tau = 0`.** Increasing tau trades about 2 accuracy points for about 11 points of macro recall at `tau = 0.5`, and collapses at 0.75 (the largest class, about a third of the test set, is almost never predicted).
- **For balanced performance across classes, use `tau` near 0.25–0.5.**
- **Seeds and variants give small gains:** a single model scores about 0.75, and ensembling reaches about 0.763. `M`, `R` and `S1` are indistinguishable within seed noise (about 0.5 points).
- **`n_mix=64` is no better than 32.**

## Notes

- Tau is selected on the validation set only. In `ensemble_multi.py` the current rule is the highest val macro recall with at most a 2-point val accuracy drop. For an accuracy-oriented selection, use `tau = 0` or select by val accuracy.
- `train.train_on_val=true` trains on train+val (a final refit). Test stays untouched, but val metrics and val-based choices (checkpoint, tau) are then meaningless. A refit ensemble reached 0.782 test accuracy; it is not comparable to the table above and should be reported separately and labelled as such.

