# Prediction analysis: final-hpo-spectral-TP.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 74.696%; macro precision: 54.267%; macro recall: 46.196%; macro F1: 46.353%.
- Mean predicted-class confidence: 0.667; correct: 0.741; incorrect: 0.451.
- 10-bin expected calibration error: 0.0798. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 3.45; median: 1.0; rank ≤1: 74.7%, rank ≤3: 88.4%, rank ≤5: 92.2%, rank ≤10: 95.2%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 46.89 | 0.410 |
| NA | 35 | 0.000 | 0.000 | 6.06 | 0.456 |
| PEV | 35 | 0.000 | 0.000 | 36.51 | 0.270 |
| 9TY | 24 | 0.000 | 0.000 | 6.92 | 0.494 |
| CU | 17 | 0.000 | 0.000 | 12.29 | 0.526 |
| D12 | 12 | 0.000 | 0.000 | 32.42 | 0.366 |
| FE2 | 10 | 0.000 | 0.000 | 11.70 | 0.299 |
| DTP | 8 | 0.000 | 0.000 | 58.25 | 0.469 |
| FUC | 8 | 0.000 | 0.000 | 4.38 | 0.662 |
| NI | 6 | 0.000 | 0.000 | 9.33 | 0.320 |
| GLY | 5 | 0.000 | 0.000 | 6.00 | 0.343 |
| PLC | 2 | 0.000 | 0.000 | 39.50 | 0.389 |
| FE | 43 | 0.023 | 0.031 | 11.05 | 0.285 |
| 6OU | 38 | 0.053 | 0.111 | 23.11 | 0.327 |
| PGW | 17 | 0.059 | 0.167 | 20.24 | 0.226 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.818 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.851 |
| CPL | 4 | 1.000 | 0.364 | 1.00 | 0.536 |
| UTP | 1 | 1.000 | 0.250 | 1.00 | 0.345 |
| NAG | 4480 | 0.980 | 0.841 | 1.05 | 0.787 |
| FMN | 44 | 0.977 | 0.977 | 1.02 | 0.862 |
| CYC | 115 | 0.974 | 0.862 | 1.03 | 0.775 |
| NDP | 34 | 0.971 | 0.971 | 1.12 | 0.768 |
| SF4 | 329 | 0.970 | 0.952 | 1.13 | 0.839 |
| IMP | 21 | 0.952 | 0.714 | 1.05 | 0.790 |
| CLA | 1106 | 0.949 | 0.833 | 1.26 | 0.801 |
| F3S | 12 | 0.917 | 0.786 | 1.17 | 0.615 |
| CDL | 169 | 0.905 | 0.750 | 2.05 | 0.781 |
| GDP | 297 | 0.889 | 0.930 | 1.83 | 0.735 |
| MG | 435 | 0.878 | 0.630 | 1.18 | 0.590 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 32 | 268 | 300 | 9.0% |
| MAN ↔ NAG | 171 | 33 | 204 | 6.1% |
| BMA ↔ NAG | 130 | 11 | 141 | 4.2% |
| CHL ↔ CLA | 98 | 17 | 115 | 3.5% |
| MG ↔ ZN | 15 | 73 | 88 | 2.6% |
| CA ↔ MG | 60 | 17 | 77 | 2.3% |
| NAG ↔ UNL | 0 | 77 | 77 | 2.3% |
| BCL ↔ CLA | 71 | 5 | 76 | 2.3% |
| CA ↔ ZN | 28 | 23 | 51 | 1.5% |
| ATP ↔ rare | 17 | 32 | 49 | 1.5% |
| LMT ↔ rare | 37 | 11 | 48 | 1.4% |
| CLR ↔ Y01 | 17 | 27 | 44 | 1.3% |
| 3PE ↔ rare | 20 | 21 | 41 | 1.2% |
| AGS ↔ ATP | 19 | 22 | 41 | 1.2% |
| BCR ↔ rare | 13 | 28 | 41 | 1.2% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6KAD_C_524_DGD | DGD | rare | 0.978 | 44 |
| 8ET3_B_500_AGS | AGS | ATP | 0.963 | 3 |
| 8C29_N_619_XAT | rare | BCR | 0.962 | 4 |
| 6KAD_g_609_CHL | CHL | CLA | 0.956 | 2 |
| 7LGW_B_831_D10 | rare | POV | 0.955 | 4 |
| 6OA9_B_901_ATP | ATP | ADP | 0.948 | 2 |
| 7K4A_C_703_Y01 | Y01 | CLR | 0.948 | 2 |
| 8E4P_A_1203_Y01 | Y01 | CLR | 0.946 | 2 |
| 7Y8A_A_848_8CT | rare | BCR | 0.942 | 2 |
| 8B4O_A_603_HEB | rare | HEM | 0.940 | 2 |
| 7SGR_F_802_6OU | 6OU | rare | 0.938 | 87 |
| 8D1G_D_505_MC3 | MC3 | rare | 0.934 | 10 |
| 7XXF_1_101_I7D | rare | SPO | 0.932 | 2 |
| 5W3H_A_500_GTP | GTP | GDP | 0.930 | 2 |
| 6KAD_n_608_CHL | CHL | CLA | 0.927 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
