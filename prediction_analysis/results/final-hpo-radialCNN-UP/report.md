# Prediction analysis: final-hpo-radialCNN-UP.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 69.359%; macro precision: 40.464%; macro recall: 32.018%; macro F1: 32.321%.
- Mean predicted-class confidence: 0.687; correct: 0.782; incorrect: 0.471.
- 10-bin expected calibration error: 0.0177. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.37; median: 1.0; rank ≤1: 69.4%, rank ≤3: 87.5%, rank ≤5: 92.8%, rank ≤10: 96.6%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 11.02 | 0.277 |
| 6OU | 38 | 0.000 | 0.000 | 7.79 | 0.390 |
| PEV | 35 | 0.000 | 0.000 | 49.17 | 0.450 |
| PGV | 29 | 0.000 | 0.000 | 16.41 | 0.346 |
| PIO | 26 | 0.000 | 0.000 | 23.08 | 0.598 |
| 9TY | 24 | 0.000 | 0.000 | 4.54 | 0.460 |
| PO4 | 24 | 0.000 | 0.000 | 6.46 | 0.485 |
| CU | 17 | 0.000 | 0.000 | 7.47 | 0.521 |
| PGW | 17 | 0.000 | 0.000 | 9.59 | 0.336 |
| CMP | 16 | 0.000 | 0.000 | 5.12 | 0.561 |
| 9Z9 | 14 | 0.000 | 0.000 | 4.93 | 0.530 |
| PEF | 13 | 0.000 | 0.000 | 15.54 | 0.367 |
| PQN | 13 | 0.000 | 0.000 | 9.69 | 0.392 |
| F3S | 12 | 0.000 | 0.000 | 8.92 | 0.463 |
| FE2 | 10 | 0.000 | 0.000 | 9.50 | 0.335 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| NAG | 4480 | 0.983 | 0.907 | 1.04 | 0.932 |
| SF4 | 329 | 0.945 | 0.869 | 1.29 | 0.931 |
| EHZ | 16 | 0.938 | 1.000 | 1.06 | 0.892 |
| 6EU | 13 | 0.923 | 0.706 | 1.08 | 0.764 |
| CLA | 1106 | 0.914 | 0.741 | 1.20 | 0.771 |
| 2MR | 21 | 0.905 | 0.864 | 2.24 | 0.941 |
| IMP | 21 | 0.857 | 0.783 | 1.24 | 0.723 |
| NDP | 34 | 0.853 | 0.763 | 1.71 | 0.804 |
| GTP | 589 | 0.834 | 0.863 | 1.47 | 0.658 |
| SAH | 12 | 0.833 | 0.833 | 1.17 | 0.462 |
| MG | 435 | 0.825 | 0.515 | 1.26 | 0.479 |
| Y01 | 94 | 0.809 | 0.603 | 1.35 | 0.693 |
| HIC | 22 | 0.773 | 0.773 | 1.23 | 0.634 |
| CDL | 169 | 0.763 | 0.733 | 2.01 | 0.771 |
| FES | 92 | 0.761 | 0.631 | 1.63 | 0.737 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 45 | 192 | 237 | 5.9% |
| CA ↔ MG | 124 | 21 | 145 | 3.6% |
| CLA ↔ rare | 45 | 64 | 109 | 2.7% |
| MG ↔ ZN | 19 | 90 | 109 | 2.7% |
| BCL ↔ CLA | 105 | 3 | 108 | 2.7% |
| BMA ↔ MAN | 64 | 42 | 106 | 2.6% |
| CHL ↔ CLA | 105 | 1 | 106 | 2.6% |
| ADP ↔ ATP | 61 | 36 | 97 | 2.4% |
| ADP ↔ GDP | 1 | 83 | 84 | 2.1% |
| ATP ↔ rare | 32 | 48 | 80 | 2.0% |
| NAG ↔ UNL | 0 | 80 | 80 | 2.0% |
| MAN ↔ NAG | 55 | 19 | 74 | 1.8% |
| ATP ↔ GTP | 6 | 65 | 71 | 1.8% |
| ADP ↔ rare | 45 | 20 | 65 | 1.6% |
| LMT ↔ rare | 50 | 10 | 60 | 1.5% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6PLU_D_602_UNL | UNL | NAG | 0.996 | 9 |
| 8FCA_D_1002_XS9 | rare | NAG | 0.992 | 2 |
| 7Z5J_E_301_IV9 | rare | NAG | 0.992 | 2 |
| 6PLS_A_516_UNL | UNL | NAG | 0.992 | 2 |
| 7PGQ_F_2902_ZN | ZN | SF4 | 0.991 | 14 |
| 6UDK_a_5_MAN | MAN | NAG | 0.991 | 4 |
| 7XE4_F_1918_HP6 | rare | NAG | 0.989 | 2 |
| 7SUD_A_2609_TPO | rare | NAG | 0.988 | 2 |
| 6TTF_A_1001_LZ2 | rare | NAG | 0.988 | 3 |
| 6PLS_E_603_UNL | UNL | NAG | 0.988 | 3 |
| 7XE4_F_1916_HP6 | rare | NAG | 0.986 | 2 |
| 8GZ2_B_2002_WMK | rare | NAG | 0.985 | 3 |
| 6PLT_C_504_UNL | UNL | NAG | 0.985 | 6 |
| 8EW3_B_602_UQ1 | rare | NAG | 0.984 | 2 |
| 6PLT_A_509_UNL | UNL | NAG | 0.983 | 3 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
