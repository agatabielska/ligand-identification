# Prediction analysis: final-hpo-radialCNN-TP.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 64.690%; macro precision: 35.465%; macro recall: 28.114%; macro F1: 28.084%.
- Mean predicted-class confidence: 0.677; correct: 0.762; incorrect: 0.521.
- 10-bin expected calibration error: 0.0298. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.73; median: 1.0; rank ≤1: 64.7%, rank ≤3: 84.0%, rank ≤5: 91.0%, rank ≤10: 95.6%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| BMA | 186 | 0.000 | 0.000 | 3.43 | 0.744 |
| ZN | 140 | 0.000 | 0.000 | 3.21 | 0.385 |
| MN | 56 | 0.000 | 0.000 | 12.77 | 0.391 |
| FE | 43 | 0.000 | 0.000 | 5.07 | 0.376 |
| ANP | 39 | 0.000 | 0.000 | 10.46 | 0.528 |
| LHG | 36 | 0.000 | 0.000 | 9.11 | 0.463 |
| NA | 35 | 0.000 | 0.000 | 7.37 | 0.401 |
| PEV | 35 | 0.000 | 0.000 | 46.29 | 0.481 |
| PGV | 29 | 0.000 | 0.000 | 15.45 | 0.391 |
| 9TY | 24 | 0.000 | 0.000 | 5.38 | 0.724 |
| PO4 | 24 | 0.000 | 0.000 | 8.00 | 0.581 |
| PLM | 20 | 0.000 | 0.000 | 9.95 | 0.326 |
| CU | 17 | 0.000 | 0.000 | 9.00 | 0.466 |
| PGW | 17 | 0.000 | 0.000 | 10.06 | 0.267 |
| CMP | 16 | 0.000 | 0.000 | 7.00 | 0.501 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| EHZ | 16 | 1.000 | 0.941 | 1.00 | 0.930 |
| NAG | 4480 | 0.976 | 0.787 | 1.04 | 0.859 |
| 2MR | 21 | 0.952 | 0.909 | 1.62 | 0.925 |
| CLA | 1106 | 0.917 | 0.708 | 1.28 | 0.788 |
| SAH | 12 | 0.917 | 0.917 | 1.08 | 0.602 |
| MG | 435 | 0.878 | 0.392 | 1.21 | 0.416 |
| SF4 | 329 | 0.872 | 0.905 | 1.47 | 0.886 |
| 6EU | 13 | 0.846 | 0.846 | 1.15 | 0.965 |
| CYC | 115 | 0.730 | 0.700 | 1.73 | 0.715 |
| ADP | 370 | 0.730 | 0.539 | 1.69 | 0.673 |
| ATP | 369 | 0.721 | 0.492 | 2.18 | 0.673 |
| CDL | 169 | 0.686 | 0.655 | 2.29 | 0.719 |
| FMN | 44 | 0.682 | 0.811 | 4.77 | 0.839 |
| IMP | 21 | 0.667 | 0.341 | 1.86 | 0.855 |
| GTP | 589 | 0.660 | 0.882 | 2.12 | 0.660 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 42 | 372 | 414 | 8.9% |
| MAN ↔ NAG | 234 | 1 | 235 | 5.1% |
| BMA ↔ NAG | 173 | 0 | 173 | 3.7% |
| CA ↔ MG | 159 | 14 | 173 | 3.7% |
| BCL ↔ CLA | 134 | 2 | 136 | 2.9% |
| CLA ↔ rare | 38 | 77 | 115 | 2.5% |
| MG ↔ ZN | 0 | 114 | 114 | 2.5% |
| CHL ↔ CLA | 99 | 0 | 99 | 2.1% |
| ATP ↔ GTP | 3 | 91 | 94 | 2.0% |
| ADP ↔ GDP | 3 | 73 | 76 | 1.6% |
| ATP ↔ rare | 30 | 44 | 74 | 1.6% |
| MG ↔ rare | 0 | 73 | 73 | 1.6% |
| ADP ↔ ATP | 44 | 28 | 72 | 1.5% |
| FME ↔ NAG | 64 | 0 | 64 | 1.4% |
| 3PE ↔ rare | 45 | 17 | 62 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 8AGC_L_8_GLC | rare | NAG | 0.986 | 2 |
| 6PLT_D_504_UNL | UNL | NAG | 0.978 | 10 |
| 6PLT_A_511_UNL | UNL | NAG | 0.975 | 8 |
| 6PLT_C_504_UNL | UNL | NAG | 0.973 | 3 |
| 6PLT_E_504_UNL | UNL | NAG | 0.972 | 4 |
| 7DHI_R_401_68H | rare | 2MR | 0.971 | 3 |
| 7VCS_K_903_AGS | AGS | IMP | 0.970 | 2 |
| 6KGX_NI_72_MEN | rare | NAG | 0.969 | 2 |
| 7PIN_G1_612_CLA | CLA | NAG | 0.967 | 9 |
| 7OIX_D_7_MAN | MAN | NAG | 0.967 | 4 |
| 7PIN_G1_605_CHL | CHL | NAG | 0.967 | 13 |
| 7L06_J_3_BMA | BMA | NAG | 0.966 | 2 |
| 7VY2_i_1_FME | FME | NAG | 0.966 | 5 |
| 6PLS_E_605_UNL | UNL | NAG | 0.966 | 5 |
| 6PLT_A_509_UNL | UNL | NAG | 0.965 | 4 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
