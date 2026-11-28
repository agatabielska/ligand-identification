# Prediction analysis: final-hpo-spectral-TP-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 72.290%; macro precision: 46.067%; macro recall: 38.389%; macro F1: 39.365%.
- Mean predicted-class confidence: 0.730; correct: 0.817; incorrect: 0.500.
- 10-bin expected calibration error: 0.0100. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.23; median: 1.0; rank ≤1: 72.3%, rank ≤3: 88.0%, rank ≤5: 93.1%, rank ≤10: 96.8%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| ZN | 140 | 0.000 | 0.000 | 2.98 | 0.351 |
| CHL | 121 | 0.000 | 0.000 | 2.72 | 0.773 |
| MN | 56 | 0.000 | 0.000 | 12.75 | 0.266 |
| NA | 35 | 0.000 | 0.000 | 6.89 | 0.357 |
| PEV | 35 | 0.000 | 0.000 | 39.46 | 0.349 |
| PGV | 29 | 0.000 | 0.000 | 10.55 | 0.350 |
| 9TY | 24 | 0.000 | 0.000 | 5.71 | 0.596 |
| PO4 | 24 | 0.000 | 0.000 | 7.67 | 0.380 |
| LBN | 23 | 0.000 | 0.000 | 21.96 | 0.356 |
| CU | 17 | 0.000 | 0.000 | 10.71 | 0.571 |
| PGW | 17 | 0.000 | 0.000 | 4.47 | 0.258 |
| PEF | 13 | 0.000 | 0.000 | 14.85 | 0.383 |
| D12 | 12 | 0.000 | 0.000 | 13.33 | 0.356 |
| FE2 | 10 | 0.000 | 0.000 | 8.10 | 0.325 |
| SEP | 9 | 0.000 | 0.000 | 7.33 | 0.685 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.913 | 1.00 | 0.945 |
| FMN | 44 | 1.000 | 0.898 | 1.00 | 0.946 |
| IMP | 21 | 1.000 | 0.750 | 1.00 | 0.928 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.985 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.980 |
| NAG | 4480 | 0.982 | 0.828 | 1.04 | 0.889 |
| NDP | 34 | 0.971 | 0.868 | 1.32 | 0.949 |
| CLA | 1106 | 0.958 | 0.805 | 1.15 | 0.867 |
| SF4 | 329 | 0.951 | 0.948 | 1.19 | 0.953 |
| F3S | 12 | 0.917 | 0.846 | 1.25 | 0.661 |
| CDL | 169 | 0.905 | 0.769 | 1.44 | 0.869 |
| 2MR | 21 | 0.905 | 0.905 | 1.38 | 0.877 |
| NAI | 42 | 0.881 | 0.974 | 1.21 | 0.835 |
| GDP | 297 | 0.879 | 0.847 | 1.51 | 0.829 |
| EHZ | 16 | 0.875 | 0.933 | 2.06 | 0.858 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 33 | 279 | 312 | 8.5% |
| MAN ↔ NAG | 206 | 18 | 224 | 6.1% |
| BMA ↔ NAG | 159 | 3 | 162 | 4.4% |
| CA ↔ MG | 91 | 49 | 140 | 3.8% |
| CHL ↔ CLA | 111 | 0 | 111 | 3.0% |
| MG ↔ ZN | 6 | 96 | 102 | 2.8% |
| BCL ↔ CLA | 88 | 4 | 92 | 2.5% |
| NAG ↔ UNL | 1 | 75 | 76 | 2.1% |
| 3PE ↔ rare | 39 | 19 | 58 | 1.6% |
| LMT ↔ rare | 48 | 10 | 58 | 1.6% |
| AGS ↔ ATP | 25 | 31 | 56 | 1.5% |
| BCR ↔ rare | 24 | 27 | 51 | 1.4% |
| CLR ↔ Y01 | 15 | 35 | 50 | 1.4% |
| ADP ↔ ATP | 24 | 25 | 49 | 1.3% |
| ATP ↔ GTP | 6 | 42 | 48 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6KGX_a2_202_PEB | PEB | CYC | 0.994 | 2 |
| 6KGX_aI_202_PEB | PEB | CYC | 0.992 | 2 |
| 6PLT_A_511_UNL | UNL | NAG | 0.988 | 6 |
| 8AA1_C_3_FRU | rare | NAG | 0.988 | 2 |
| 8EOG_D_1108_LEU | rare | NAG | 0.987 | 2 |
| 6PLT_B_504_UNL | UNL | NAG | 0.985 | 5 |
| 6U1N_R_490_TPO | rare | NAG | 0.984 | 2 |
| 7PBZ_C_601_HSM | rare | NAG | 0.984 | 2 |
| 7R4D_i_1_SAC | rare | NAG | 0.980 | 2 |
| 8DGT_A_729_SEP | SEP | NAG | 0.979 | 17 |
| 8AGC_L_8_GLC | rare | NAG | 0.976 | 2 |
| 6KAD_y_609_CHL | CHL | CLA | 0.975 | 2 |
| 6PLT_D_511_UNL | UNL | NAG | 0.974 | 7 |
| 6VI0_P_6_MAN | MAN | NAG | 0.973 | 4 |
| 6PLT_D_502_UNL | UNL | NAG | 0.973 | 4 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
