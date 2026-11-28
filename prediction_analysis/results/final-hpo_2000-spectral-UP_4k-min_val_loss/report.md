# Prediction analysis: final-hpo_2000-spectral-UP_4k-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 80.519%; macro precision: 62.000%; macro recall: 53.468%; macro F1: 53.907%.
- Mean predicted-class confidence: 0.812; correct: 0.881; incorrect: 0.526.
- 10-bin expected calibration error: 0.0135. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 1.83; median: 1.0; rank ≤1: 80.5%, rank ≤3: 92.9%, rank ≤5: 95.7%, rank ≤10: 98.1%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 13.25 | 0.279 |
| PEV | 35 | 0.000 | 0.000 | 34.86 | 0.532 |
| CU | 17 | 0.000 | 0.000 | 7.94 | 0.601 |
| PEF | 13 | 0.000 | 0.000 | 8.77 | 0.394 |
| FE2 | 10 | 0.000 | 0.000 | 9.00 | 0.327 |
| DTP | 8 | 0.000 | 0.000 | 36.00 | 0.500 |
| FUC | 8 | 0.000 | 0.000 | 3.25 | 0.460 |
| NI | 6 | 0.000 | 0.000 | 11.17 | 0.429 |
| GLY | 5 | 0.000 | 0.000 | 2.40 | 0.353 |
| PLC | 2 | 0.000 | 0.000 | 12.50 | 0.616 |
| FE | 43 | 0.047 | 0.100 | 5.86 | 0.394 |
| PLM | 20 | 0.050 | 0.200 | 3.70 | 0.451 |
| PGV | 29 | 0.069 | 0.250 | 6.14 | 0.461 |
| SEP | 9 | 0.111 | 1.000 | 4.00 | 0.542 |
| PGW | 17 | 0.118 | 0.154 | 3.71 | 0.376 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| FMN | 44 | 1.000 | 1.000 | 1.00 | 0.968 |
| NDP | 34 | 1.000 | 0.791 | 1.00 | 0.934 |
| IMP | 21 | 1.000 | 0.875 | 1.00 | 0.979 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.996 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.996 |
| HEA | 8 | 1.000 | 0.800 | 1.00 | 0.766 |
| 7ID | 4 | 1.000 | 0.667 | 1.00 | 0.830 |
| CPL | 4 | 1.000 | 0.500 | 1.00 | 0.496 |
| UTP | 1 | 1.000 | 0.333 | 1.00 | 0.768 |
| CYC | 115 | 0.991 | 0.905 | 1.01 | 0.982 |
| SF4 | 329 | 0.991 | 0.964 | 1.05 | 0.975 |
| NAG | 4480 | 0.990 | 0.947 | 1.01 | 0.965 |
| CLA | 1106 | 0.986 | 0.858 | 1.03 | 0.897 |
| NAI | 42 | 0.976 | 0.976 | 1.02 | 0.859 |
| CDL | 169 | 0.953 | 0.870 | 1.15 | 0.932 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 22 | 110 | 132 | 5.1% |
| CA ↔ MG | 100 | 22 | 122 | 4.8% |
| BMA ↔ MAN | 60 | 57 | 117 | 4.6% |
| MG ↔ ZN | 9 | 94 | 103 | 4.0% |
| CHL ↔ CLA | 97 | 4 | 101 | 3.9% |
| AGS ↔ ATP | 29 | 32 | 61 | 2.4% |
| 3PE ↔ rare | 28 | 25 | 53 | 2.1% |
| BCL ↔ CLA | 49 | 2 | 51 | 2.0% |
| UNL ↔ rare | 46 | 5 | 51 | 2.0% |
| NAG ↔ UNL | 0 | 46 | 46 | 1.8% |
| ATP ↔ rare | 13 | 30 | 43 | 1.7% |
| MAN ↔ NAG | 31 | 11 | 42 | 1.6% |
| BCR ↔ rare | 15 | 26 | 41 | 1.6% |
| CA ↔ ZN | 20 | 20 | 40 | 1.6% |
| MAN ↔ rare | 10 | 26 | 36 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7YMI_B_612_CL7 | rare | CLA | 0.996 | 4 |
| 8DQV_B_402_F3S | F3S | rare | 0.994 | 2 |
| 8DQV_D_403_F3S | F3S | rare | 0.992 | 2 |
| 7YMI_b_613_CL7 | rare | CLA | 0.992 | 4 |
| 5JCO_J_501_G2P | rare | GTP | 0.987 | 3 |
| 7Z5J_I_304_IWO | rare | NAG | 0.986 | 2 |
| 8ASL_b_3052_EQ3 | rare | BCR | 0.985 | 2 |
| 7SSS_C_401_PEV | PEV | rare | 0.985 | 41 |
| 6KAD_N_605_CHL | CHL | CLA | 0.985 | 2 |
| 7B2L_M_302_PIO | PIO | rare | 0.984 | 5 |
| 6PLS_E_614_UNL | UNL | NAG | 0.983 | 83 |
| 7Y8A_B_845_8CT | rare | BCR | 0.983 | 2 |
| 8GFA_B_1205_POV | POV | PC1 | 0.982 | 42 |
| 7XXF_I_101_I7D | rare | SPO | 0.982 | 2 |
| 7Z5J_A_304_IWO | rare | NAG | 0.982 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
