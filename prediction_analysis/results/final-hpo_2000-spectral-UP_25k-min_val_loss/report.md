# Prediction analysis: final-hpo_2000-spectral-UP_25k-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 81.339%; macro precision: 61.531%; macro recall: 55.712%; macro F1: 56.022%.
- Mean predicted-class confidence: 0.825; correct: 0.890; incorrect: 0.539.
- 10-bin expected calibration error: 0.0208. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 1.79; median: 1.0; rank ≤1: 81.3%, rank ≤3: 93.5%, rank ≤5: 95.9%, rank ≤10: 98.0%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 13.91 | 0.478 |
| NA | 35 | 0.000 | 0.000 | 5.49 | 0.474 |
| PEV | 35 | 0.000 | 0.000 | 29.69 | 0.530 |
| CU | 17 | 0.000 | 0.000 | 7.29 | 0.711 |
| PEF | 13 | 0.000 | 0.000 | 5.85 | 0.437 |
| FE2 | 10 | 0.000 | 0.000 | 6.50 | 0.535 |
| DTP | 8 | 0.000 | 0.000 | 27.50 | 0.470 |
| NI | 6 | 0.000 | 0.000 | 7.17 | 0.443 |
| GLY | 5 | 0.000 | 0.000 | 2.00 | 0.554 |
| PLC | 2 | 0.000 | 0.000 | 8.50 | 0.429 |
| FE | 43 | 0.023 | 0.125 | 5.84 | 0.574 |
| PGV | 29 | 0.034 | 0.500 | 6.59 | 0.418 |
| PGW | 17 | 0.118 | 0.222 | 3.82 | 0.462 |
| FUC | 8 | 0.125 | 0.143 | 3.12 | 0.537 |
| ZN | 140 | 0.150 | 0.216 | 2.41 | 0.469 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.927 | 1.00 | 0.971 |
| IMP | 21 | 1.000 | 0.840 | 1.00 | 0.966 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.975 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.994 |
| 7ID | 4 | 1.000 | 1.000 | 1.00 | 0.928 |
| CPL | 4 | 1.000 | 0.571 | 1.00 | 0.514 |
| UTP | 1 | 1.000 | 0.333 | 1.00 | 0.711 |
| NAG | 4480 | 0.993 | 0.950 | 1.01 | 0.971 |
| CDL | 169 | 0.988 | 0.865 | 1.04 | 0.943 |
| CLA | 1106 | 0.981 | 0.872 | 1.02 | 0.911 |
| FMN | 44 | 0.977 | 1.000 | 1.02 | 0.981 |
| NAI | 42 | 0.976 | 0.976 | 1.02 | 0.827 |
| SF4 | 329 | 0.976 | 0.979 | 1.08 | 0.964 |
| NDP | 34 | 0.971 | 0.943 | 1.03 | 0.952 |
| 2MR | 21 | 0.952 | 1.000 | 1.05 | 0.972 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| BMA ↔ MAN | 58 | 68 | 126 | 5.1% |
| NAG ↔ rare | 14 | 110 | 124 | 5.0% |
| MG ↔ ZN | 22 | 98 | 120 | 4.9% |
| CA ↔ MG | 93 | 21 | 114 | 4.6% |
| CHL ↔ CLA | 90 | 8 | 98 | 4.0% |
| AGS ↔ ATP | 16 | 37 | 53 | 2.2% |
| NAG ↔ UNL | 0 | 52 | 52 | 2.1% |
| 3PE ↔ rare | 24 | 24 | 48 | 2.0% |
| CA ↔ MN | 0 | 48 | 48 | 2.0% |
| UNL ↔ rare | 43 | 4 | 47 | 1.9% |
| ATP ↔ rare | 13 | 29 | 42 | 1.7% |
| BCL ↔ CLA | 36 | 6 | 42 | 1.7% |
| BCR ↔ rare | 13 | 24 | 37 | 1.5% |
| MAN ↔ rare | 10 | 26 | 36 | 1.5% |
| CA ↔ ZN | 20 | 11 | 31 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7YMI_B_612_CL7 | rare | CLA | 0.992 | 3 |
| 7YMI_b_613_CL7 | rare | CLA | 0.991 | 3 |
| 6PLS_E_614_UNL | UNL | NAG | 0.989 | 6 |
| 5JCO_D_501_G2P | rare | GTP | 0.988 | 3 |
| 6KGX_W4_201_PEB | PEB | CYC | 0.988 | 2 |
| 7B2L_B_301_PIO | PIO | rare | 0.987 | 27 |
| 7W3G_F_501_ATP | ATP | ADP | 0.985 | 2 |
| 7F1O_F_501_LDP | rare | NAG | 0.983 | 2 |
| 7Z6Q_a_811_BCL | BCL | CDL | 0.983 | 13 |
| 5JCO_K_501_G2P | rare | GTP | 0.982 | 3 |
| 7TJ5_B_601_D21 | rare | LMT | 0.982 | 2 |
| 5JCO_J_501_G2P | rare | GTP | 0.982 | 4 |
| 7XXF_K_101_I7D | rare | SPO | 0.981 | 2 |
| 7Y8A_A_853_8CT | rare | BCR | 0.981 | 2 |
| 7WDB_D_905_657 | rare | NAG | 0.981 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
