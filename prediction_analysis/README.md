# Checkpoint prediction analysis

Run from the repository root to analyze every CSV in `checkpoint_predictions/`:

```bash
uv run python prediction_analysis/analyze_checkpoint_predictions.py
```

By default, each CSV gets its own directory under
`prediction_analysis/results/<csv-name>/`. To analyze selected files or use
different locations:

```bash
uv run python prediction_analysis/analyze_checkpoint_predictions.py \
  --csv final-hpo_2000-spectral-UP_4k.csv \
  --csv final-hpo_2000-spectral-UP_4k-min_val_loss.csv \
  --input-dir checkpoint_predictions \
  --output-dir prediction_analysis/results
```

Each result directory contains a Markdown report, overall metrics in JSON,
class-level metrics, confusion pairs, highest-confidence errors, confidence
calibration bins, and plots. Mean and median true-class rank measure the model's
ranking of the actual class; the source CSV does not contain full class
probabilities, so calibration is evaluated for the predicted class only.
