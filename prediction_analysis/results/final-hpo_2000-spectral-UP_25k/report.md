# Prediction analysis: final-hpo_2000-spectral-UP_25k.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 82.858%; macro precision: 70.347%; macro recall: 57.022%; macro F1: 59.437%.
- Mean predicted-class confidence: 0.743; correct: 0.797; incorrect: 0.477.
- 10-bin expected calibration error: 0.0860. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.63; median: 1.0; rank ≤1: 82.9%, rank ≤3: 93.2%, rank ≤5: 95.4%, rank ≤10: 97.1%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 64.48 | 0.589 |
| PEV | 35 | 0.000 | 0.000 | 44.49 | 0.451 |
| CU | 17 | 0.000 | 0.000 | 6.24 | 0.648 |
| PGW | 17 | 0.000 | 0.000 | 7.59 | 0.361 |
| FE2 | 10 | 0.000 | 0.000 | 21.80 | 0.482 |
| DTP | 8 | 0.000 | 0.000 | 30.62 | 0.403 |
| NI | 6 | 0.000 | 0.000 | 6.67 | 0.531 |
| GLY | 5 | 0.000 | 0.000 | 2.80 | 0.695 |
| PLC | 2 | 0.000 | 0.000 | 15.00 | 0.605 |
| FE | 43 | 0.047 | 0.250 | 12.72 | 0.652 |
| NA | 35 | 0.057 | 0.667 | 4.83 | 0.518 |
| PEF | 13 | 0.077 | 1.000 | 13.54 | 0.294 |
| LBN | 23 | 0.130 | 1.000 | 34.09 | 0.320 |
| 6OU | 38 | 0.132 | 0.263 | 12.16 | 0.337 |
| PGV | 29 | 0.138 | 0.667 | 12.97 | 0.305 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.958 | 1.00 | 0.781 |
| IMP | 21 | 1.000 | 1.000 | 1.00 | 0.855 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.872 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.922 |
| 7ID | 4 | 1.000 | 1.000 | 1.00 | 0.594 |
| CPL | 4 | 1.000 | 0.444 | 1.00 | 0.745 |
| UTP | 1 | 1.000 | 1.000 | 1.00 | 0.528 |
| NAG | 4480 | 0.992 | 0.963 | 1.03 | 0.858 |
| CLA | 1106 | 0.982 | 0.889 | 1.04 | 0.825 |
| FMN | 44 | 0.977 | 1.000 | 1.02 | 0.835 |
| SF4 | 329 | 0.976 | 0.982 | 1.04 | 0.845 |
| CDL | 169 | 0.970 | 0.863 | 1.06 | 0.786 |
| GTP | 589 | 0.961 | 0.983 | 1.07 | 0.795 |
| HIC | 22 | 0.955 | 0.913 | 1.05 | 0.881 |
| GDP | 297 | 0.953 | 0.973 | 1.32 | 0.861 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| BMA ↔ MAN | 48 | 67 | 115 | 5.1% |
| NAG ↔ rare | 18 | 83 | 101 | 4.5% |
| MG ↔ ZN | 22 | 65 | 87 | 3.9% |
| CHL ↔ CLA | 68 | 12 | 80 | 3.5% |
| CA ↔ MG | 54 | 24 | 78 | 3.5% |
| CA ↔ ZN | 32 | 35 | 67 | 3.0% |
| UNL ↔ rare | 64 | 2 | 66 | 2.9% |
| 3PE ↔ rare | 31 | 31 | 62 | 2.7% |
| ATP ↔ rare | 25 | 29 | 54 | 2.4% |
| NAG ↔ UNL | 1 | 42 | 43 | 1.9% |
| ADP ↔ ATP | 19 | 17 | 36 | 1.6% |
| BCL ↔ CLA | 34 | 0 | 34 | 1.5% |
| AGS ↔ ATP | 19 | 14 | 33 | 1.5% |
| PC1 ↔ rare | 22 | 11 | 33 | 1.5% |
| BCR ↔ rare | 13 | 18 | 31 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7Y8A_A_848_8CT | rare | BCR | 0.971 | 2 |
| 7EW6_J_103_DGD | DGD | PC1 | 0.963 | 77 |
| 7EW6_1_517_CHL | CHL | CLA | 0.945 | 18 |
| 7VY2_w_101_SPO | SPO | rare | 0.942 | 5 |
| 7Y8A_B_845_8CT | rare | BCR | 0.937 | 5 |
| 7N77_A_1201_ALF | rare | FES | 0.929 | 7 |
| 7MXE_X_6_MAN | MAN | BMA | 0.928 | 2 |
| 7B2L_B_301_PIO | PIO | rare | 0.926 | 17 |
| 8G27_C_1002_MG | MG | rare | 0.921 | 2 |
| 6NYY_D_801_ANP | ANP | ATP | 0.920 | 7 |
| 8ASP_b_854_ZEX | rare | BCR | 0.919 | 2 |
| 7ZPO_I_502_K | K | CA | 0.918 | 5 |
| 7Y8A_R_201_8CT | rare | BCR | 0.917 | 2 |
| 7BVG_B_1201_PO4 | PO4 | SF4 | 0.917 | 2 |
| 7EW6_2_513_CHL | CHL | CLA | 0.916 | 3 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
