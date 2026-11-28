# Prediction analysis: final-hpo_2000-spectral-UP_4k.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 82.311%; macro precision: 69.493%; macro recall: 55.822%; macro F1: 58.925%.
- Mean predicted-class confidence: 0.728; correct: 0.788; incorrect: 0.453.
- 10-bin expected calibration error: 0.0946. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.65; median: 1.0; rank ≤1: 82.3%, rank ≤3: 93.2%, rank ≤5: 95.3%, rank ≤10: 97.0%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 28.71 | 0.379 |
| NA | 35 | 0.000 | 0.000 | 4.34 | 0.443 |
| PEV | 35 | 0.000 | 0.000 | 47.46 | 0.415 |
| CU | 17 | 0.000 | 0.000 | 8.29 | 0.542 |
| PEF | 13 | 0.000 | 0.000 | 25.62 | 0.279 |
| FE2 | 10 | 0.000 | 0.000 | 8.60 | 0.306 |
| DTP | 8 | 0.000 | 0.000 | 53.75 | 0.385 |
| NI | 6 | 0.000 | 0.000 | 9.33 | 0.359 |
| GLY | 5 | 0.000 | 0.000 | 3.80 | 0.368 |
| PLC | 2 | 0.000 | 0.000 | 64.00 | 0.317 |
| FE | 43 | 0.047 | 0.250 | 4.86 | 0.500 |
| SEP | 9 | 0.111 | 1.000 | 7.11 | 0.544 |
| C14 | 32 | 0.125 | 0.800 | 25.19 | 0.313 |
| PGV | 29 | 0.138 | 0.667 | 20.66 | 0.304 |
| 6OU | 38 | 0.158 | 0.300 | 11.50 | 0.368 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.935 | 1.00 | 0.816 |
| FMN | 44 | 1.000 | 1.000 | 1.00 | 0.845 |
| IMP | 21 | 1.000 | 0.778 | 1.00 | 0.885 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.870 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.921 |
| HEA | 8 | 1.000 | 1.000 | 1.00 | 0.572 |
| 7ID | 4 | 1.000 | 1.000 | 1.00 | 0.511 |
| CPL | 4 | 1.000 | 0.800 | 1.00 | 0.730 |
| UTP | 1 | 1.000 | 1.000 | 1.00 | 0.578 |
| NAG | 4480 | 0.990 | 0.955 | 1.01 | 0.844 |
| SF4 | 329 | 0.988 | 0.988 | 1.04 | 0.851 |
| CLA | 1106 | 0.986 | 0.894 | 1.03 | 0.817 |
| NAI | 42 | 0.976 | 1.000 | 1.02 | 0.684 |
| 2MR | 21 | 0.952 | 1.000 | 1.24 | 0.868 |
| GTP | 589 | 0.947 | 0.974 | 1.16 | 0.805 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 23 | 92 | 115 | 4.9% |
| BMA ↔ MAN | 51 | 60 | 111 | 4.8% |
| MG ↔ ZN | 20 | 81 | 101 | 4.3% |
| CA ↔ MG | 60 | 27 | 87 | 3.7% |
| CHL ↔ CLA | 72 | 7 | 79 | 3.4% |
| UNL ↔ rare | 69 | 2 | 71 | 3.0% |
| 3PE ↔ rare | 30 | 25 | 55 | 2.4% |
| ATP ↔ rare | 20 | 35 | 55 | 2.4% |
| CA ↔ ZN | 31 | 19 | 50 | 2.1% |
| NAG ↔ UNL | 1 | 41 | 42 | 1.8% |
| MAN ↔ NAG | 28 | 13 | 41 | 1.8% |
| CU ↔ MN | 0 | 40 | 40 | 1.7% |
| AGS ↔ ATP | 20 | 19 | 39 | 1.7% |
| BCL ↔ CLA | 36 | 1 | 37 | 1.6% |
| LMT ↔ rare | 26 | 5 | 31 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 6X89_AL_401_PC1 | PC1 | rare | 0.971 | 10 |
| 6V1X_C_601_QNJ | rare | CLR | 0.954 | 72 |
| 8D1F_D_501_CA | CA | MG | 0.951 | 3 |
| 8D1F_A_502_CA | CA | MG | 0.949 | 3 |
| 7PIN_Y1_620_LUT | LUT | rare | 0.944 | 17 |
| 8BRD_E_301_NA | NA | CA | 0.940 | 5 |
| 6W22_C_802_ATP | ATP | AGS | 0.935 | 74 |
| 6W22_D_802_ATP | ATP | AGS | 0.935 | 2 |
| 8D1F_C_501_CA | CA | MG | 0.931 | 2 |
| 7Y8A_A_848_8CT | rare | BCR | 0.929 | 2 |
| 8D1F_E_501_CA | CA | MG | 0.925 | 3 |
| 7Y8A_A_854_IHT | rare | BCR | 0.923 | 2 |
| 7YMI_B_612_CL7 | rare | CLA | 0.921 | 89 |
| 6PLS_B_602_UNL | UNL | rare | 0.921 | 5 |
| 7SSS_C_401_PEV | PEV | rare | 0.921 | 86 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
