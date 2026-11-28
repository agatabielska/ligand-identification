# Prediction analysis: final-hpo-radialCNN-TP-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 61.502%; macro precision: 32.383%; macro recall: 25.205%; macro F1: 25.240%.
- Mean predicted-class confidence: 0.640; correct: 0.738; incorrect: 0.484.
- 10-bin expected calibration error: 0.0252. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.90; median: 1.0; rank ≤1: 61.5%, rank ≤3: 81.6%, rank ≤5: 89.4%, rank ≤10: 95.3%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| BMA | 186 | 0.000 | 0.000 | 3.48 | 0.702 |
| ZN | 140 | 0.000 | 0.000 | 3.26 | 0.366 |
| CHL | 121 | 0.000 | 0.000 | 5.60 | 0.664 |
| MN | 56 | 0.000 | 0.000 | 13.20 | 0.336 |
| 6OU | 38 | 0.000 | 0.000 | 8.82 | 0.341 |
| LHG | 36 | 0.000 | 0.000 | 10.44 | 0.445 |
| NA | 35 | 0.000 | 0.000 | 6.71 | 0.383 |
| PEV | 35 | 0.000 | 0.000 | 45.09 | 0.405 |
| PGV | 29 | 0.000 | 0.000 | 15.97 | 0.336 |
| 9TY | 24 | 0.000 | 0.000 | 5.54 | 0.692 |
| PO4 | 24 | 0.000 | 0.000 | 11.79 | 0.554 |
| LMG | 19 | 0.000 | 0.000 | 14.37 | 0.353 |
| CU | 17 | 0.000 | 0.000 | 10.35 | 0.350 |
| PGW | 17 | 0.000 | 0.000 | 8.06 | 0.273 |
| CMP | 16 | 0.000 | 0.000 | 5.19 | 0.528 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| NAG | 4480 | 0.971 | 0.781 | 1.06 | 0.837 |
| 2MR | 21 | 0.952 | 0.714 | 1.10 | 0.906 |
| SF4 | 329 | 0.891 | 0.864 | 1.58 | 0.865 |
| CLA | 1106 | 0.883 | 0.715 | 1.32 | 0.766 |
| EHZ | 16 | 0.875 | 1.000 | 1.44 | 0.771 |
| 6EU | 13 | 0.846 | 0.846 | 1.69 | 0.899 |
| SAH | 12 | 0.833 | 0.667 | 1.33 | 0.564 |
| NDP | 34 | 0.765 | 0.722 | 2.21 | 0.746 |
| FMN | 44 | 0.727 | 0.696 | 3.93 | 0.796 |
| CDL | 169 | 0.716 | 0.651 | 2.25 | 0.712 |
| IMP | 21 | 0.714 | 0.441 | 2.33 | 0.622 |
| MG | 435 | 0.708 | 0.408 | 1.43 | 0.362 |
| HEM | 90 | 0.678 | 0.545 | 1.83 | 0.685 |
| ATP | 369 | 0.667 | 0.471 | 2.60 | 0.624 |
| ADP | 370 | 0.570 | 0.402 | 1.94 | 0.589 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 63 | 375 | 438 | 8.6% |
| MAN ↔ NAG | 236 | 3 | 239 | 4.7% |
| CA ↔ MG | 114 | 85 | 199 | 3.9% |
| BMA ↔ NAG | 175 | 0 | 175 | 3.5% |
| ADP ↔ GDP | 0 | 153 | 153 | 3.0% |
| BCL ↔ CLA | 128 | 1 | 129 | 2.5% |
| CLA ↔ rare | 40 | 71 | 111 | 2.2% |
| ATP ↔ GTP | 0 | 104 | 104 | 2.1% |
| ATP ↔ rare | 68 | 34 | 102 | 2.0% |
| ADP ↔ rare | 83 | 14 | 97 | 1.9% |
| CHL ↔ CLA | 93 | 0 | 93 | 1.8% |
| MG ↔ ZN | 6 | 78 | 84 | 1.7% |
| ADP ↔ GTP | 1 | 82 | 83 | 1.6% |
| ADP ↔ ATP | 52 | 22 | 74 | 1.5% |
| GTP ↔ rare | 67 | 7 | 74 | 1.5% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6KGX_JC_201_PEB | PEB | CYC | 0.987 | 3 |
| 6PLT_D_504_UNL | UNL | NAG | 0.985 | 9 |
| 7PIN_G1_612_CLA | CLA | NAG | 0.979 | 12 |
| 6PLT_B_511_UNL | UNL | NAG | 0.977 | 5 |
| 6PLT_E_502_UNL | UNL | NAG | 0.976 | 4 |
| 8G27_E_1_BGC | rare | NAG | 0.976 | 2 |
| 7OIX_D_7_MAN | MAN | NAG | 0.974 | 3 |
| 6PLT_E_504_UNL | UNL | NAG | 0.973 | 5 |
| 6KAD_Y_601_CHL | CHL | CLA | 0.973 | 3 |
| 6PLT_D_512_UNL | UNL | NAG | 0.971 | 4 |
| 6PLT_C_504_UNL | UNL | NAG | 0.971 | 5 |
| 6PLS_E_605_UNL | UNL | NAG | 0.969 | 7 |
| 6X3Z_G_7_MAN | MAN | NAG | 0.966 | 3 |
| 6UDK_V_5_MAN | MAN | NAG | 0.965 | 4 |
| 6PLT_D_502_UNL | UNL | NAG | 0.964 | 4 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
