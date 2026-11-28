# Prediction analysis: final-hpo-radialCNN-TU-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 57.076%; macro precision: 25.491%; macro recall: 16.942%; macro F1: 16.561%.
- Mean predicted-class confidence: 0.583; correct: 0.690; incorrect: 0.440.
- 10-bin expected calibration error: 0.0224. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 3.35; median: 1.0; rank ≤1: 57.1%, rank ≤3: 77.9%, rank ≤5: 86.9%, rank ≤10: 93.6%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| BMA | 186 | 0.000 | 0.000 | 4.81 | 0.747 |
| BCL | 150 | 0.000 | 0.000 | 4.62 | 0.468 |
| ZN | 140 | 0.000 | 0.000 | 3.44 | 0.361 |
| FME | 73 | 0.000 | 0.000 | 6.63 | 0.739 |
| MN | 56 | 0.000 | 0.000 | 17.93 | 0.338 |
| FE | 43 | 0.000 | 0.000 | 7.09 | 0.336 |
| ANP | 39 | 0.000 | 0.000 | 10.64 | 0.459 |
| 6OU | 38 | 0.000 | 0.000 | 12.21 | 0.323 |
| LHG | 36 | 0.000 | 0.000 | 17.22 | 0.444 |
| NA | 35 | 0.000 | 0.000 | 8.20 | 0.408 |
| PEV | 35 | 0.000 | 0.000 | 47.11 | 0.441 |
| PCW | 34 | 0.000 | 0.000 | 19.71 | 0.262 |
| PGV | 29 | 0.000 | 0.000 | 18.69 | 0.310 |
| PIO | 26 | 0.000 | 0.000 | 16.50 | 0.380 |
| 9TY | 24 | 0.000 | 0.000 | 9.67 | 0.613 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| NAG | 4480 | 0.968 | 0.719 | 1.12 | 0.840 |
| 2MR | 21 | 0.857 | 0.581 | 4.05 | 0.633 |
| SF4 | 329 | 0.848 | 0.730 | 1.77 | 0.750 |
| EHZ | 16 | 0.812 | 0.650 | 1.62 | 0.430 |
| CLA | 1106 | 0.750 | 0.631 | 1.73 | 0.576 |
| MG | 435 | 0.731 | 0.381 | 1.58 | 0.345 |
| ATP | 369 | 0.645 | 0.354 | 2.80 | 0.548 |
| GTP | 589 | 0.611 | 0.855 | 2.90 | 0.476 |
| FMN | 44 | 0.591 | 0.448 | 5.00 | 0.579 |
| NDP | 34 | 0.559 | 0.500 | 4.15 | 0.614 |
| BCR | 67 | 0.552 | 0.463 | 2.57 | 0.453 |
| Y01 | 94 | 0.532 | 0.379 | 2.60 | 0.399 |
| CYC | 115 | 0.504 | 0.667 | 2.44 | 0.497 |
| LUT | 14 | 0.500 | 0.259 | 5.29 | 0.589 |
| SAH | 12 | 0.500 | 0.750 | 2.25 | 0.250 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 41 | 438 | 479 | 8.5% |
| MAN ↔ NAG | 238 | 0 | 238 | 4.2% |
| CLA ↔ NAG | 172 | 17 | 189 | 3.3% |
| BMA ↔ NAG | 169 | 0 | 169 | 3.0% |
| CLA ↔ rare | 82 | 76 | 158 | 2.8% |
| CA ↔ MG | 140 | 7 | 147 | 2.6% |
| ADP ↔ ATP | 99 | 25 | 124 | 2.2% |
| ATP ↔ rare | 53 | 65 | 118 | 2.1% |
| BCL ↔ CLA | 115 | 0 | 115 | 2.0% |
| MG ↔ ZN | 0 | 105 | 105 | 1.9% |
| LMT ↔ rare | 85 | 5 | 90 | 1.6% |
| MG ↔ NAG | 69 | 16 | 85 | 1.5% |
| NAG ↔ UNL | 0 | 84 | 84 | 1.5% |
| 3PE ↔ rare | 65 | 18 | 83 | 1.5% |
| ADP ↔ GDP | 0 | 78 | 78 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6X3Z_G_7_MAN | MAN | NAG | 0.981 | 4 |
| 6PLT_D_504_UNL | UNL | NAG | 0.979 | 8 |
| 7E51_F_501_PEP | rare | NAG | 0.978 | 2 |
| 6PLT_A_511_UNL | UNL | NAG | 0.977 | 4 |
| 7L06_J_3_BMA | BMA | NAG | 0.977 | 2 |
| 6PLO_C_504_ABU | rare | NAG | 0.976 | 3 |
| 7E51_D_501_PEP | rare | NAG | 0.973 | 2 |
| 7E51_H_501_PEP | rare | NAG | 0.971 | 2 |
| 6PLT_C_504_UNL | UNL | NAG | 0.971 | 3 |
| 6PLT_C_511_UNL | UNL | NAG | 0.971 | 7 |
| 7E51_E_501_PEP | rare | NAG | 0.971 | 2 |
| 7E51_G_501_PEP | rare | NAG | 0.969 | 2 |
| 8OM1_L_1_FME | FME | NAG | 0.968 | 5 |
| 6PLT_B_504_UNL | UNL | NAG | 0.968 | 6 |
| 7Y7W_A_701_ABU | rare | NAG | 0.966 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
