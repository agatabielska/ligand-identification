# Prediction analysis: final-hpo_2000-spectral-UP_10k.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 82.804%; macro precision: 67.784%; macro recall: 57.637%; macro F1: 59.578%.
- Mean predicted-class confidence: 0.731; correct: 0.788; incorrect: 0.460.
- 10-bin expected calibration error: 0.0968. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.71; median: 1.0; rank ≤1: 82.8%, rank ≤3: 93.0%, rank ≤5: 95.3%, rank ≤10: 97.0%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 60.20 | 0.483 |
| NA | 35 | 0.000 | 0.000 | 5.89 | 0.479 |
| PEV | 35 | 0.000 | 0.000 | 40.11 | 0.462 |
| CU | 17 | 0.000 | 0.000 | 7.00 | 0.612 |
| PGW | 17 | 0.000 | 0.000 | 7.71 | 0.289 |
| PEF | 13 | 0.000 | 0.000 | 15.38 | 0.367 |
| FE2 | 10 | 0.000 | 0.000 | 9.10 | 0.468 |
| DTP | 8 | 0.000 | 0.000 | 56.25 | 0.344 |
| NI | 6 | 0.000 | 0.000 | 8.33 | 0.342 |
| GLY | 5 | 0.000 | 0.000 | 2.20 | 0.670 |
| PLC | 2 | 0.000 | 0.000 | 64.50 | 0.378 |
| FE | 43 | 0.047 | 0.133 | 7.19 | 0.502 |
| PGV | 29 | 0.103 | 0.750 | 17.03 | 0.253 |
| SEP | 9 | 0.111 | 0.250 | 3.00 | 0.626 |
| C14 | 32 | 0.156 | 1.000 | 31.31 | 0.377 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.958 | 1.00 | 0.811 |
| IMP | 21 | 1.000 | 0.913 | 1.00 | 0.865 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.841 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.913 |
| 7ID | 4 | 1.000 | 0.800 | 1.00 | 0.522 |
| CPL | 4 | 1.000 | 0.571 | 1.00 | 0.779 |
| AJP | 2 | 1.000 | 0.286 | 1.00 | 0.580 |
| UTP | 1 | 1.000 | 1.000 | 1.00 | 0.355 |
| NAG | 4480 | 0.990 | 0.960 | 1.02 | 0.846 |
| SF4 | 329 | 0.985 | 0.988 | 1.03 | 0.855 |
| CLA | 1106 | 0.981 | 0.899 | 1.04 | 0.824 |
| FMN | 44 | 0.977 | 1.000 | 1.02 | 0.838 |
| NAI | 42 | 0.976 | 1.000 | 1.02 | 0.628 |
| CDL | 169 | 0.964 | 0.858 | 1.08 | 0.799 |
| HIC | 22 | 0.955 | 0.875 | 1.05 | 0.906 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| BMA ↔ MAN | 56 | 58 | 114 | 5.0% |
| NAG ↔ rare | 30 | 80 | 110 | 4.9% |
| MG ↔ ZN | 20 | 68 | 88 | 3.9% |
| CA ↔ MG | 53 | 30 | 83 | 3.7% |
| CHL ↔ CLA | 61 | 13 | 74 | 3.3% |
| CA ↔ ZN | 34 | 27 | 61 | 2.7% |
| UNL ↔ rare | 57 | 4 | 61 | 2.7% |
| 3PE ↔ rare | 26 | 27 | 53 | 2.3% |
| K ↔ MN | 0 | 48 | 48 | 2.1% |
| AGS ↔ ATP | 21 | 21 | 42 | 1.9% |
| ATP ↔ rare | 14 | 28 | 42 | 1.9% |
| NAG ↔ UNL | 1 | 37 | 38 | 1.7% |
| LMT ↔ rare | 26 | 8 | 34 | 1.5% |
| MAN ↔ NAG | 18 | 14 | 32 | 1.4% |
| PC1 ↔ rare | 19 | 13 | 32 | 1.4% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7XVE_A_2016_LPE | LPE | rare | 0.950 | 7 |
| 7PIN_Y1_620_LUT | LUT | rare | 0.944 | 5 |
| 7SGR_I_801_6OU | 6OU | rare | 0.937 | 13 |
| 7Y8A_A_854_IHT | rare | BCR | 0.929 | 2 |
| 7OH6_A_1502_ALF | rare | FES | 0.925 | 2 |
| 8SET_A_5101_CMP | CMP | rare | 0.923 | 3 |
| 6W22_C_801_ATP | ATP | AGS | 0.921 | 2 |
| 6X89_AL_401_PC1 | PC1 | rare | 0.918 | 48 |
| 8OM1_N_1_FME | FME | rare | 0.918 | 2 |
| 8E4P_B_1201_Y01 | Y01 | CLR | 0.917 | 2 |
| 7Y8A_A_848_8CT | rare | BCR | 0.916 | 2 |
| 7XXF_7_401_U10 | rare | BCR | 0.916 | 3 |
| 8OM1_A_201_LMT | LMT | rare | 0.914 | 2 |
| 8E4P_D_1201_Y01 | Y01 | CLR | 0.913 | 2 |
| 7KNS_B_902_PO4 | PO4 | rare | 0.912 | 4 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
