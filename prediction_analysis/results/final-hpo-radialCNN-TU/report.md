# Prediction analysis: final-hpo-radialCNN-TU.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 57.425%; macro precision: 26.758%; macro recall: 19.755%; macro F1: 19.699%.
- Mean predicted-class confidence: 0.602; correct: 0.703; incorrect: 0.467.
- 10-bin expected calibration error: 0.0309. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 3.32; median: 1.0; rank ≤1: 57.4%, rank ≤3: 79.1%, rank ≤5: 87.1%, rank ≤10: 93.7%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| BMA | 186 | 0.000 | 0.000 | 4.89 | 0.741 |
| FME | 73 | 0.000 | 0.000 | 5.58 | 0.703 |
| MN | 56 | 0.000 | 0.000 | 17.70 | 0.401 |
| FE | 43 | 0.000 | 0.000 | 6.77 | 0.317 |
| ANP | 39 | 0.000 | 0.000 | 11.69 | 0.454 |
| 6OU | 38 | 0.000 | 0.000 | 13.11 | 0.344 |
| LHG | 36 | 0.000 | 0.000 | 14.78 | 0.449 |
| NA | 35 | 0.000 | 0.000 | 8.23 | 0.397 |
| PEV | 35 | 0.000 | 0.000 | 46.83 | 0.448 |
| PGV | 29 | 0.000 | 0.000 | 19.24 | 0.318 |
| NAD | 26 | 0.000 | 0.000 | 14.62 | 0.316 |
| PIO | 26 | 0.000 | 0.000 | 19.38 | 0.426 |
| 9TY | 24 | 0.000 | 0.000 | 11.54 | 0.560 |
| FAD | 24 | 0.000 | 0.000 | 10.75 | 0.337 |
| PO4 | 24 | 0.000 | 0.000 | 12.79 | 0.602 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| EHZ | 16 | 1.000 | 1.000 | 1.00 | 0.727 |
| NAG | 4480 | 0.961 | 0.717 | 1.14 | 0.840 |
| 2MR | 21 | 0.905 | 0.704 | 1.81 | 0.811 |
| 6EU | 13 | 0.846 | 0.458 | 1.46 | 0.811 |
| SF4 | 329 | 0.802 | 0.632 | 2.02 | 0.736 |
| CLA | 1106 | 0.718 | 0.652 | 1.74 | 0.589 |
| LUT | 14 | 0.714 | 0.370 | 1.57 | 0.523 |
| ATP | 369 | 0.686 | 0.376 | 2.18 | 0.597 |
| MG | 435 | 0.674 | 0.392 | 1.66 | 0.335 |
| SAH | 12 | 0.667 | 0.800 | 1.58 | 0.325 |
| NDP | 34 | 0.647 | 0.564 | 3.76 | 0.879 |
| CDL | 169 | 0.609 | 0.510 | 2.17 | 0.531 |
| GTP | 589 | 0.538 | 0.798 | 2.81 | 0.593 |
| BCR | 67 | 0.537 | 0.371 | 2.75 | 0.446 |
| FMN | 44 | 0.523 | 0.548 | 7.39 | 0.824 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 57 | 434 | 491 | 8.8% |
| MAN ↔ NAG | 236 | 0 | 236 | 4.2% |
| CLA ↔ NAG | 190 | 24 | 214 | 3.8% |
| BMA ↔ NAG | 169 | 0 | 169 | 3.0% |
| CLA ↔ rare | 72 | 67 | 139 | 2.5% |
| CA ↔ MG | 118 | 18 | 136 | 2.4% |
| ATP ↔ GTP | 1 | 134 | 135 | 2.4% |
| ATP ↔ rare | 53 | 62 | 115 | 2.1% |
| BCL ↔ CLA | 111 | 1 | 112 | 2.0% |
| ADP ↔ GDP | 2 | 102 | 104 | 1.9% |
| ADP ↔ ATP | 85 | 15 | 100 | 1.8% |
| MG ↔ SF4 | 46 | 37 | 83 | 1.5% |
| MG ↔ ZN | 3 | 79 | 82 | 1.5% |
| NAG ↔ UNL | 0 | 81 | 81 | 1.4% |
| MG ↔ NAG | 67 | 13 | 80 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6X3Z_G_7_MAN | MAN | NAG | 0.978 | 4 |
| 7E51_E_501_PEP | rare | NAG | 0.978 | 2 |
| 6PLT_A_511_UNL | UNL | NAG | 0.976 | 3 |
| 7RZ4_B_1101_ZK1 | rare | FMN | 0.971 | 3 |
| 6PLT_D_504_UNL | UNL | NAG | 0.969 | 3 |
| 6UDK_V_5_MAN | MAN | NAG | 0.969 | 4 |
| 7PIN_G1_612_CLA | CLA | NAG | 0.966 | 6 |
| 8F6Z_F_3_BMA | BMA | NAG | 0.963 | 3 |
| 6PLT_C_504_UNL | UNL | NAG | 0.963 | 3 |
| 8DMH_P_3_BMA | BMA | NAG | 0.963 | 4 |
| 7E51_B_501_PEP | rare | NAG | 0.963 | 2 |
| 7E51_F_501_PEP | rare | NAG | 0.962 | 2 |
| 7KNS_E_902_PO4 | PO4 | NAG | 0.962 | 15 |
| 6PLO_C_504_ABU | rare | NAG | 0.962 | 4 |
| 6UDK_a_5_MAN | MAN | NAG | 0.961 | 3 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
