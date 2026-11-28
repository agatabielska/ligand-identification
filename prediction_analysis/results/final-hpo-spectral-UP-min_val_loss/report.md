# Prediction analysis: final-hpo-spectral-UP-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 79.836%; macro precision: 59.654%; macro recall: 52.122%; macro F1: 52.775%.
- Mean predicted-class confidence: 0.803; correct: 0.873; incorrect: 0.526.
- 10-bin expected calibration error: 0.0149. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 1.81; median: 1.0; rank ≤1: 79.8%, rank ≤3: 92.7%, rank ≤5: 95.6%, rank ≤10: 98.2%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 15.73 | 0.264 |
| PEV | 35 | 0.000 | 0.000 | 29.31 | 0.510 |
| CU | 17 | 0.000 | 0.000 | 7.94 | 0.642 |
| PEF | 13 | 0.000 | 0.000 | 8.62 | 0.405 |
| FE2 | 10 | 0.000 | 0.000 | 9.10 | 0.286 |
| SEP | 9 | 0.000 | 0.000 | 5.22 | 0.503 |
| DTP | 8 | 0.000 | 0.000 | 25.50 | 0.640 |
| FUC | 8 | 0.000 | 0.000 | 3.50 | 0.467 |
| NI | 6 | 0.000 | 0.000 | 10.83 | 0.472 |
| GLY | 5 | 0.000 | 0.000 | 3.60 | 0.336 |
| PLC | 2 | 0.000 | 0.000 | 9.50 | 0.712 |
| ANP | 39 | 0.026 | 0.091 | 4.92 | 0.583 |
| FE | 43 | 0.047 | 0.062 | 5.84 | 0.384 |
| PLM | 20 | 0.050 | 0.250 | 4.70 | 0.494 |
| NA | 35 | 0.057 | 0.118 | 5.63 | 0.382 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| FMN | 44 | 1.000 | 1.000 | 1.00 | 0.963 |
| NAI | 42 | 1.000 | 0.977 | 1.00 | 0.808 |
| IMP | 21 | 1.000 | 0.750 | 1.00 | 0.971 |
| 6EU | 13 | 1.000 | 0.929 | 1.00 | 0.993 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.997 |
| 7ID | 4 | 1.000 | 1.000 | 1.00 | 0.688 |
| UTP | 1 | 1.000 | 1.000 | 1.00 | 0.674 |
| CYC | 115 | 0.991 | 0.864 | 1.01 | 0.971 |
| SF4 | 329 | 0.991 | 0.967 | 1.07 | 0.971 |
| NAG | 4480 | 0.991 | 0.939 | 1.01 | 0.964 |
| CLA | 1106 | 0.982 | 0.858 | 1.04 | 0.895 |
| NDP | 34 | 0.971 | 0.825 | 1.06 | 0.916 |
| CDL | 169 | 0.964 | 0.886 | 1.09 | 0.930 |
| GTP | 589 | 0.930 | 0.961 | 1.15 | 0.891 |
| PQN | 13 | 0.923 | 0.750 | 1.08 | 0.926 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 19 | 121 | 140 | 5.3% |
| BMA ↔ MAN | 65 | 52 | 117 | 4.4% |
| CA ↔ MG | 85 | 24 | 109 | 4.1% |
| MG ↔ ZN | 4 | 105 | 109 | 4.1% |
| CHL ↔ CLA | 94 | 4 | 98 | 3.7% |
| AGS ↔ ATP | 25 | 37 | 62 | 2.3% |
| 3PE ↔ rare | 30 | 29 | 59 | 2.2% |
| UNL ↔ rare | 51 | 5 | 56 | 2.1% |
| BCL ↔ CLA | 47 | 6 | 53 | 2.0% |
| MAN ↔ NAG | 39 | 14 | 53 | 2.0% |
| ATP ↔ rare | 15 | 36 | 51 | 1.9% |
| NAG ↔ UNL | 0 | 46 | 46 | 1.7% |
| BCR ↔ rare | 19 | 26 | 45 | 1.7% |
| ADP ↔ ATP | 28 | 13 | 41 | 1.5% |
| MAN ↔ rare | 9 | 28 | 37 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7Z5J_G_304_IWO | rare | NAG | 0.988 | 2 |
| 6KGX_W1_201_PEB | PEB | CYC | 0.988 | 2 |
| 7F1O_F_501_LDP | rare | NAG | 0.988 | 2 |
| 8DQV_D_404_F3S | F3S | rare | 0.987 | 2 |
| 7YMI_b_613_CL7 | rare | CLA | 0.986 | 3 |
| 5JCO_J_501_G2P | rare | GTP | 0.986 | 3 |
| 6TTF_A_1001_LZ2 | rare | NAG | 0.985 | 2 |
| 7Z6Q_a_815_BCL | BCL | CLA | 0.984 | 3 |
| 7Z5J_C_301_IV9 | rare | NAG | 0.984 | 2 |
| 7YMI_B_612_CL7 | rare | CLA | 0.983 | 3 |
| 7Z5J_E_301_IV9 | rare | NAG | 0.983 | 2 |
| 8AA3_K_5_FRU | rare | NAG | 0.983 | 2 |
| 7XXF_I_101_I7D | rare | SPO | 0.982 | 2 |
| 7Z5J_A_304_IWO | rare | NAG | 0.982 | 2 |
| 7BVG_C_1_BXY | rare | NAG | 0.981 | 3 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
