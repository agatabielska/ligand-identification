# Prediction analysis: final-hpo_2000-spectral-UP_10k-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 81.096%; macro precision: 61.102%; macro recall: 55.677%; macro F1: 55.230%.
- Mean predicted-class confidence: 0.821; correct: 0.887; incorrect: 0.537.
- 10-bin expected calibration error: 0.0160. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 1.82; median: 1.0; rank ≤1: 81.1%, rank ≤3: 93.2%, rank ≤5: 95.9%, rank ≤10: 98.1%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 14.88 | 0.374 |
| PEV | 35 | 0.000 | 0.000 | 35.31 | 0.540 |
| CU | 17 | 0.000 | 0.000 | 7.12 | 0.697 |
| PEF | 13 | 0.000 | 0.000 | 7.92 | 0.452 |
| FE2 | 10 | 0.000 | 0.000 | 8.30 | 0.406 |
| DTP | 8 | 0.000 | 0.000 | 33.75 | 0.620 |
| FUC | 8 | 0.000 | 0.000 | 3.38 | 0.506 |
| NI | 6 | 0.000 | 0.000 | 9.17 | 0.442 |
| GLY | 5 | 0.000 | 0.000 | 2.20 | 0.468 |
| PLC | 2 | 0.000 | 0.000 | 11.50 | 0.610 |
| NA | 35 | 0.029 | 0.200 | 5.34 | 0.452 |
| FE | 43 | 0.047 | 0.222 | 4.70 | 0.462 |
| PGV | 29 | 0.069 | 1.000 | 6.03 | 0.418 |
| PLM | 20 | 0.100 | 0.167 | 4.50 | 0.512 |
| SEP | 9 | 0.111 | 0.333 | 4.00 | 0.529 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| FMN | 44 | 1.000 | 1.000 | 1.00 | 0.964 |
| NAI | 42 | 1.000 | 0.977 | 1.00 | 0.827 |
| IMP | 21 | 1.000 | 0.955 | 1.00 | 0.990 |
| 6EU | 13 | 1.000 | 0.812 | 1.00 | 0.993 |
| SAH | 12 | 1.000 | 0.923 | 1.00 | 0.998 |
| HEA | 8 | 1.000 | 0.800 | 1.00 | 0.716 |
| 7ID | 4 | 1.000 | 0.800 | 1.00 | 0.858 |
| CPL | 4 | 1.000 | 0.667 | 1.00 | 0.594 |
| UTP | 1 | 1.000 | 0.167 | 1.00 | 0.314 |
| CYC | 115 | 0.991 | 0.912 | 1.01 | 0.983 |
| SF4 | 329 | 0.991 | 0.976 | 1.04 | 0.975 |
| NAG | 4480 | 0.991 | 0.953 | 1.01 | 0.966 |
| CLA | 1106 | 0.980 | 0.862 | 1.05 | 0.902 |
| CDL | 169 | 0.976 | 0.864 | 1.07 | 0.946 |
| NDP | 34 | 0.971 | 0.892 | 1.03 | 0.940 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 23 | 101 | 124 | 5.0% |
| BMA ↔ MAN | 54 | 63 | 117 | 4.7% |
| MG ↔ ZN | 14 | 89 | 103 | 4.1% |
| CHL ↔ CLA | 96 | 3 | 99 | 4.0% |
| CA ↔ MG | 73 | 25 | 98 | 3.9% |
| AGS ↔ ATP | 21 | 39 | 60 | 2.4% |
| 3PE ↔ rare | 18 | 34 | 52 | 2.1% |
| ATP ↔ rare | 17 | 33 | 50 | 2.0% |
| UNL ↔ rare | 43 | 5 | 48 | 1.9% |
| BCL ↔ CLA | 41 | 6 | 47 | 1.9% |
| NAG ↔ UNL | 0 | 43 | 43 | 1.7% |
| MAN ↔ NAG | 27 | 12 | 39 | 1.6% |
| CA ↔ ZN | 22 | 12 | 34 | 1.4% |
| CLA ↔ rare | 9 | 25 | 34 | 1.4% |
| BCR ↔ rare | 9 | 24 | 33 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7YMI_B_612_CL7 | rare | CLA | 0.989 | 3 |
| 7M6J_C_501_GDP | GDP | rare | 0.988 | 59 |
| 7XXF_A_103_I7D | rare | SPO | 0.987 | 2 |
| 8AGC_L_1_NDG | rare | NAG | 0.985 | 2 |
| 7XXF_K_101_I7D | rare | SPO | 0.984 | 2 |
| 7YMI_b_613_CL7 | rare | CLA | 0.984 | 3 |
| 7K18_A_2210_6OU | 6OU | rare | 0.983 | 3 |
| 7W3G_F_501_ATP | ATP | ADP | 0.983 | 2 |
| 8AA3_K_5_FRU | rare | NAG | 0.982 | 2 |
| 6IGZ_B_852_LHG | LHG | NAG | 0.980 | 5 |
| 7Z6Q_G_406_BCL | BCL | CDL | 0.980 | 3 |
| 7SSS_G_303_PEV | PEV | rare | 0.979 | 16 |
| 7F1O_F_501_LDP | rare | NAG | 0.979 | 2 |
| 7XXF_3_101_I7D | rare | SPO | 0.979 | 2 |
| 7B2L_Q_301_PIO | PIO | rare | 0.978 | 40 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
