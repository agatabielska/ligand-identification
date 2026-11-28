# Prediction analysis: final-hpo-spectral-TU.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 75.061%; macro precision: 55.410%; macro recall: 42.693%; macro F1: 44.818%.
- Mean predicted-class confidence: 0.724; correct: 0.801; incorrect: 0.492.
- 10-bin expected calibration error: 0.0365. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.81; median: 1.0; rank ≤1: 75.1%, rank ≤3: 88.8%, rank ≤5: 92.7%, rank ≤10: 96.0%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 18.05 | 0.307 |
| NA | 35 | 0.000 | 0.000 | 6.00 | 0.417 |
| PEV | 35 | 0.000 | 0.000 | 40.89 | 0.343 |
| 9TY | 24 | 0.000 | 0.000 | 11.33 | 0.491 |
| PO4 | 24 | 0.000 | 0.000 | 5.96 | 0.376 |
| CU | 17 | 0.000 | 0.000 | 13.59 | 0.571 |
| D12 | 12 | 0.000 | 0.000 | 11.25 | 0.417 |
| FE2 | 10 | 0.000 | 0.000 | 11.80 | 0.372 |
| SEP | 9 | 0.000 | 0.000 | 9.56 | 0.566 |
| DTP | 8 | 0.000 | 0.000 | 11.50 | 0.845 |
| FUC | 8 | 0.000 | 0.000 | 4.62 | 0.753 |
| NI | 6 | 0.000 | 0.000 | 10.67 | 0.312 |
| GLY | 5 | 0.000 | 0.000 | 9.20 | 0.439 |
| PLC | 2 | 0.000 | 0.000 | 58.00 | 0.329 |
| UTP | 1 | 0.000 | 0.000 | 5.00 | 0.244 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.849 |
| F3S | 12 | 1.000 | 0.857 | 1.00 | 0.643 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.957 |
| CPL | 4 | 1.000 | 0.364 | 1.00 | 0.689 |
| NAG | 4480 | 0.977 | 0.853 | 1.05 | 0.836 |
| SF4 | 329 | 0.976 | 0.967 | 1.10 | 0.888 |
| CYC | 115 | 0.974 | 0.889 | 1.03 | 0.809 |
| CLA | 1106 | 0.956 | 0.834 | 1.14 | 0.846 |
| FMN | 44 | 0.955 | 1.000 | 1.05 | 0.926 |
| NAI | 42 | 0.952 | 0.976 | 1.10 | 0.562 |
| NDP | 34 | 0.912 | 0.969 | 1.41 | 0.836 |
| GTP | 589 | 0.898 | 0.964 | 1.42 | 0.863 |
| CDL | 169 | 0.888 | 0.765 | 1.37 | 0.825 |
| HEM | 90 | 0.878 | 0.878 | 1.79 | 0.774 |
| GDP | 297 | 0.875 | 0.935 | 1.48 | 0.882 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 49 | 246 | 295 | 9.0% |
| MAN ↔ NAG | 152 | 26 | 178 | 5.4% |
| BMA ↔ NAG | 110 | 16 | 126 | 3.8% |
| CHL ↔ CLA | 101 | 5 | 106 | 3.2% |
| MG ↔ ZN | 19 | 61 | 80 | 2.4% |
| CA ↔ MG | 58 | 20 | 78 | 2.4% |
| NAG ↔ UNL | 0 | 76 | 76 | 2.3% |
| BCL ↔ CLA | 71 | 0 | 71 | 2.2% |
| CA ↔ ZN | 27 | 32 | 59 | 1.8% |
| BMA ↔ MAN | 29 | 26 | 55 | 1.7% |
| 3PE ↔ rare | 25 | 25 | 50 | 1.5% |
| ATP ↔ rare | 23 | 26 | 49 | 1.5% |
| CLR ↔ Y01 | 11 | 36 | 47 | 1.4% |
| LMT ↔ rare | 40 | 6 | 46 | 1.4% |
| BCR ↔ rare | 18 | 23 | 41 | 1.2% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6VZ1_B_503_3VV | rare | BCR | 0.979 | 2 |
| 8ET3_A_500_AGS | AGS | ATP | 0.978 | 2 |
| 8ET3_D_500_AGS | AGS | ATP | 0.976 | 10 |
| 6W22_E_802_ATP | ATP | AGS | 0.973 | 2 |
| 7D3U_D_608_LMT | LMT | rare | 0.971 | 2 |
| 7VY2_q_102_PGV | PGV | rare | 0.967 | 9 |
| 8C29_R_615_XAT | rare | LUT | 0.966 | 2 |
| 8ET3_C_500_AGS | AGS | ATP | 0.964 | 3 |
| 8OM1_N_1_FME | FME | rare | 0.963 | 3 |
| 6KAD_D_411_LMG | LMG | rare | 0.962 | 25 |
| 6PP5_C_600_AGS | AGS | ATP | 0.962 | 2 |
| 6KAD_C_524_DGD | DGD | rare | 0.961 | 78 |
| 7EW6_B_845_BCR | BCR | rare | 0.961 | 2 |
| 7Z18_C_1000_ZN | ZN | MG | 0.960 | 3 |
| 7PIN_S1_602_CLA | CLA | rare | 0.960 | 10 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
