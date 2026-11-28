# Prediction analysis: final-hpo-spectral-UP.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 81.673%; macro precision: 66.730%; macro recall: 54.571%; macro F1: 57.226%.
- Mean predicted-class confidence: 0.725; correct: 0.785; incorrect: 0.458.
- 10-bin expected calibration error: 0.0919. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.73; median: 1.0; rank ≤1: 81.7%, rank ≤3: 92.5%, rank ≤5: 95.0%, rank ≤10: 97.2%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 73.34 | 0.334 |
| PEV | 35 | 0.000 | 0.000 | 37.91 | 0.417 |
| CU | 17 | 0.000 | 0.000 | 14.82 | 0.589 |
| PEF | 13 | 0.000 | 0.000 | 16.46 | 0.320 |
| FE2 | 10 | 0.000 | 0.000 | 8.30 | 0.279 |
| DTP | 8 | 0.000 | 0.000 | 42.38 | 0.522 |
| NI | 6 | 0.000 | 0.000 | 18.83 | 0.369 |
| GLY | 5 | 0.000 | 0.000 | 4.00 | 0.420 |
| PLC | 2 | 0.000 | 0.000 | 54.00 | 0.671 |
| FE | 43 | 0.047 | 0.051 | 5.33 | 0.426 |
| PGW | 17 | 0.059 | 0.200 | 3.41 | 0.315 |
| SEP | 9 | 0.111 | 0.333 | 3.67 | 0.537 |
| UNL | 161 | 0.155 | 0.893 | 4.35 | 0.518 |
| ZN | 140 | 0.157 | 0.289 | 2.86 | 0.370 |
| PO4 | 24 | 0.167 | 0.667 | 7.54 | 0.489 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.891 | 1.00 | 0.836 |
| FMN | 44 | 1.000 | 1.000 | 1.00 | 0.842 |
| IMP | 21 | 1.000 | 0.955 | 1.00 | 0.876 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.806 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.934 |
| HEA | 8 | 1.000 | 1.000 | 1.00 | 0.615 |
| 7ID | 4 | 1.000 | 1.000 | 1.00 | 0.536 |
| CPL | 4 | 1.000 | 0.500 | 1.00 | 0.719 |
| UTP | 1 | 1.000 | 1.000 | 1.00 | 0.698 |
| SF4 | 329 | 0.988 | 0.988 | 1.04 | 0.846 |
| NAG | 4480 | 0.988 | 0.954 | 1.02 | 0.844 |
| CLA | 1106 | 0.975 | 0.893 | 1.04 | 0.819 |
| 2MR | 21 | 0.952 | 1.000 | 1.19 | 0.903 |
| GTP | 589 | 0.937 | 0.975 | 1.14 | 0.810 |
| GDP | 297 | 0.936 | 0.952 | 1.32 | 0.841 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 30 | 91 | 121 | 5.0% |
| BMA ↔ MAN | 47 | 64 | 111 | 4.6% |
| CA ↔ MG | 79 | 28 | 107 | 4.4% |
| MG ↔ ZN | 11 | 78 | 89 | 3.7% |
| UNL ↔ rare | 80 | 1 | 81 | 3.4% |
| CHL ↔ CLA | 61 | 17 | 78 | 3.2% |
| 3PE ↔ rare | 34 | 24 | 58 | 2.4% |
| CA ↔ ZN | 23 | 26 | 49 | 2.0% |
| ATP ↔ rare | 14 | 30 | 44 | 1.8% |
| MAN ↔ NAG | 26 | 17 | 43 | 1.8% |
| AGS ↔ ATP | 24 | 17 | 41 | 1.7% |
| BCL ↔ CLA | 40 | 0 | 40 | 1.7% |
| NAG ↔ UNL | 1 | 37 | 38 | 1.6% |
| BCR ↔ rare | 11 | 22 | 33 | 1.4% |
| PC1 ↔ rare | 18 | 12 | 30 | 1.2% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6KGX_NI_72_MEN | rare | HIC | 0.972 | 2 |
| 7XVE_A_2016_LPE | LPE | rare | 0.965 | 4 |
| 8DQV_C_602_MG | MG | NA | 0.958 | 2 |
| 7B2L_H_302_PIO | PIO | rare | 0.951 | 35 |
| 6V1X_C_601_QNJ | rare | CLR | 0.944 | 84 |
| 7B2L_C_302_PIO | PIO | rare | 0.942 | 68 |
| 7LGJ_C_904_ACP | rare | ATP | 0.941 | 9 |
| 7KNS_B_902_PO4 | PO4 | rare | 0.940 | 2 |
| 6IGZ_7_315_CLA | CLA | CHL | 0.938 | 2 |
| 7SGR_J_802_6OU | 6OU | rare | 0.938 | 74 |
| 8DQV_A_602_MG | MG | NA | 0.937 | 2 |
| 8ASP_b_854_ZEX | rare | BCR | 0.935 | 2 |
| 7PIN_Y1_620_LUT | LUT | rare | 0.933 | 2 |
| 7LGJ_B_904_ACP | rare | ATP | 0.929 | 62 |
| 6PLU_C_615_UNL | UNL | NAG | 0.927 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
