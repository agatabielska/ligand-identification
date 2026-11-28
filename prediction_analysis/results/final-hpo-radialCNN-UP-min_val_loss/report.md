# Prediction analysis: final-hpo-radialCNN-UP-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 69.215%; macro precision: 35.069%; macro recall: 29.292%; macro F1: 28.855%.
- Mean predicted-class confidence: 0.685; correct: 0.780; incorrect: 0.470.
- 10-bin expected calibration error: 0.0106. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.33; median: 1.0; rank ≤1: 69.2%, rank ≤3: 88.0%, rank ≤5: 93.0%, rank ≤10: 96.8%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| CHL | 121 | 0.000 | 0.000 | 3.79 | 0.726 |
| MN | 56 | 0.000 | 0.000 | 11.91 | 0.281 |
| ANP | 39 | 0.000 | 0.000 | 6.87 | 0.473 |
| 6OU | 38 | 0.000 | 0.000 | 10.87 | 0.340 |
| LHG | 36 | 0.000 | 0.000 | 6.39 | 0.409 |
| PEV | 35 | 0.000 | 0.000 | 46.40 | 0.384 |
| PCW | 34 | 0.000 | 0.000 | 16.03 | 0.320 |
| PGV | 29 | 0.000 | 0.000 | 13.86 | 0.328 |
| 9TY | 24 | 0.000 | 0.000 | 3.54 | 0.441 |
| LMG | 19 | 0.000 | 0.000 | 12.32 | 0.435 |
| CU | 17 | 0.000 | 0.000 | 5.94 | 0.563 |
| PGW | 17 | 0.000 | 0.000 | 8.76 | 0.257 |
| CMP | 16 | 0.000 | 0.000 | 3.44 | 0.526 |
| TPP | 16 | 0.000 | 0.000 | 11.19 | 0.410 |
| 9Z9 | 14 | 0.000 | 0.000 | 4.57 | 0.617 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| NAG | 4480 | 0.983 | 0.906 | 1.03 | 0.930 |
| SF4 | 329 | 0.936 | 0.893 | 1.33 | 0.899 |
| CLA | 1106 | 0.929 | 0.742 | 1.20 | 0.799 |
| GTP | 589 | 0.907 | 0.830 | 1.33 | 0.759 |
| BCR | 67 | 0.896 | 0.513 | 2.03 | 0.668 |
| EHZ | 16 | 0.875 | 0.824 | 1.12 | 0.735 |
| 2MR | 21 | 0.857 | 0.692 | 2.00 | 0.813 |
| IMP | 21 | 0.857 | 0.720 | 1.29 | 0.670 |
| CDL | 169 | 0.846 | 0.614 | 1.68 | 0.727 |
| CYC | 115 | 0.783 | 0.616 | 1.68 | 0.699 |
| MG | 435 | 0.770 | 0.515 | 1.32 | 0.510 |
| 6EU | 13 | 0.769 | 0.714 | 1.31 | 0.617 |
| NDP | 34 | 0.765 | 0.481 | 1.82 | 0.792 |
| CLR | 101 | 0.762 | 0.472 | 2.42 | 0.639 |
| FES | 92 | 0.750 | 0.535 | 1.47 | 0.744 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 38 | 186 | 224 | 5.5% |
| CA ↔ MG | 102 | 44 | 146 | 3.6% |
| BMA ↔ MAN | 61 | 72 | 133 | 3.3% |
| BCL ↔ CLA | 108 | 3 | 111 | 2.7% |
| CHL ↔ CLA | 111 | 0 | 111 | 2.7% |
| MG ↔ ZN | 7 | 87 | 94 | 2.3% |
| ADP ↔ ATP | 38 | 48 | 86 | 2.1% |
| MAN ↔ NAG | 70 | 16 | 86 | 2.1% |
| CLA ↔ rare | 26 | 59 | 85 | 2.1% |
| GDP ↔ GTP | 74 | 1 | 75 | 1.8% |
| NAG ↔ UNL | 1 | 74 | 75 | 1.8% |
| LMT ↔ rare | 61 | 13 | 74 | 1.8% |
| BMA ↔ NAG | 56 | 16 | 72 | 1.8% |
| ADP ↔ rare | 23 | 43 | 66 | 1.6% |
| 3PE ↔ rare | 42 | 22 | 64 | 1.6% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7XE4_F_1916_HP6 | rare | NAG | 0.994 | 2 |
| 6PLT_A_509_UNL | UNL | NAG | 0.994 | 3 |
| 7XE4_F_1918_HP6 | rare | NAG | 0.992 | 2 |
| 7SUD_A_2609_TPO | rare | NAG | 0.990 | 2 |
| 6UDK_a_5_MAN | MAN | NAG | 0.988 | 4 |
| 8FCA_D_1002_XS9 | rare | NAG | 0.987 | 2 |
| 6PLT_B_504_UNL | UNL | NAG | 0.986 | 7 |
| 6PLT_E_502_UNL | UNL | NAG | 0.986 | 3 |
| 6PLU_D_602_UNL | UNL | NAG | 0.985 | 8 |
| 6PLU_B_608_UNL | UNL | NAG | 0.982 | 14 |
| 6PLS_E_603_UNL | UNL | NAG | 0.982 | 3 |
| 7Z5J_K_301_IV9 | rare | NAG | 0.981 | 2 |
| 6X3Z_G_7_MAN | MAN | NAG | 0.980 | 4 |
| 6PLS_A_516_UNL | UNL | NAG | 0.979 | 3 |
| 6PLS_B_603_UNL | UNL | NAG | 0.978 | 3 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
