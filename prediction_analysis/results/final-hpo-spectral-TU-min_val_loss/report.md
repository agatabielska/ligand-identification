# Prediction analysis: final-hpo-spectral-TU-min_val_loss.csv

## Overall

- Samples: 13,172; classes: 91.
- Accuracy: 73.740%; macro precision: 52.741%; macro recall: 42.095%; macro F1: 42.877%.
- Mean predicted-class confidence: 0.752; correct: 0.840; incorrect: 0.506.
- 10-bin expected calibration error: 0.0146. A smaller value means confidence is better aligned with observed accuracy.
- Mean rank of the true class: 2.22; median: 1.0; rank ≤1: 73.7%, rank ≤3: 88.9%, rank ≤5: 93.2%, rank ≤10: 96.8%.

Confidence is the CSV's predicted-class probability. The CSV does not contain the full probability vector, so this report evaluates top-class calibration, not multiclass calibration.

## Hardest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |
|---|---:|---:|---:|---:|---:|
| MN | 56 | 0.000 | 0.000 | 12.66 | 0.337 |
| NA | 35 | 0.000 | 0.000 | 6.23 | 0.317 |
| PEV | 35 | 0.000 | 0.000 | 40.17 | 0.437 |
| 9TY | 24 | 0.000 | 0.000 | 8.21 | 0.717 |
| PLM | 20 | 0.000 | 0.000 | 6.95 | 0.375 |
| CU | 17 | 0.000 | 0.000 | 9.06 | 0.547 |
| D12 | 12 | 0.000 | 0.000 | 14.50 | 0.326 |
| FE2 | 10 | 0.000 | 0.000 | 11.10 | 0.267 |
| SEP | 9 | 0.000 | 0.000 | 6.11 | 0.711 |
| DTP | 8 | 0.000 | 0.000 | 8.38 | 0.499 |
| FUC | 8 | 0.000 | 0.000 | 4.75 | 0.746 |
| NI | 6 | 0.000 | 0.000 | 11.17 | 0.404 |
| GLY | 5 | 0.000 | 0.000 | 7.40 | 0.339 |
| 7ID | 4 | 0.000 | 0.000 | 7.75 | 0.197 |
| PLC | 2 | 0.000 | 0.000 | 15.50 | 0.494 |

## Easiest classes

| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |
|---|---:|---:|---:|---:|---:|
| CYC | 115 | 1.000 | 0.839 | 1.00 | 0.979 |
| FMN | 44 | 1.000 | 0.917 | 1.00 | 0.990 |
| NAI | 42 | 1.000 | 1.000 | 1.00 | 0.860 |
| IMP | 21 | 1.000 | 0.778 | 1.00 | 0.897 |
| 6EU | 13 | 1.000 | 1.000 | 1.00 | 0.998 |
| SAH | 12 | 1.000 | 1.000 | 1.00 | 0.999 |
| NAG | 4480 | 0.981 | 0.838 | 1.04 | 0.898 |
| NDP | 34 | 0.971 | 0.868 | 1.35 | 0.974 |
| CLA | 1106 | 0.962 | 0.809 | 1.11 | 0.902 |
| SF4 | 329 | 0.957 | 0.969 | 1.17 | 0.971 |
| CDL | 169 | 0.917 | 0.795 | 1.37 | 0.925 |
| F3S | 12 | 0.917 | 0.917 | 1.08 | 0.852 |
| 2MR | 21 | 0.905 | 0.905 | 1.57 | 0.909 |
| GDP | 297 | 0.899 | 0.890 | 1.33 | 0.845 |
| GTP | 589 | 0.878 | 0.949 | 1.33 | 0.857 |

## Most confused class pairs

| Class pair | A → B | B → A | Total | Share of errors |
|---|---:|---:|---:|---:|
| NAG ↔ rare | 33 | 269 | 302 | 8.7% |
| MAN ↔ NAG | 175 | 24 | 199 | 5.8% |
| BMA ↔ NAG | 145 | 6 | 151 | 4.4% |
| CHL ↔ CLA | 108 | 0 | 108 | 3.1% |
| BCL ↔ CLA | 98 | 4 | 102 | 2.9% |
| CA ↔ MG | 57 | 40 | 97 | 2.8% |
| MG ↔ ZN | 11 | 69 | 80 | 2.3% |
| NAG ↔ UNL | 1 | 78 | 79 | 2.3% |
| AGS ↔ ATP | 30 | 33 | 63 | 1.8% |
| CA ↔ ZN | 22 | 39 | 61 | 1.8% |
| ATP ↔ rare | 20 | 34 | 54 | 1.6% |
| 3PE ↔ rare | 34 | 19 | 53 | 1.5% |
| CLR ↔ Y01 | 20 | 32 | 52 | 1.5% |
| ADP ↔ ATP | 31 | 16 | 47 | 1.4% |
| LMT ↔ rare | 37 | 7 | 44 | 1.3% |

## Highest-confidence failures

| Sample | True class | Predicted class | Confidence | True-class rank |
|---|---|---|---:|---:|
| 7F1O_F_501_LDP | rare | NAG | 0.990 | 2 |
| 8EOG_D_1108_LEU | rare | NAG | 0.989 | 2 |
| 6KGX_aI_202_PEB | PEB | CYC | 0.986 | 2 |
| 8GZ2_B_2002_WMK | rare | NAG | 0.986 | 2 |
| 6PLT_A_511_UNL | UNL | NAG | 0.985 | 6 |
| 8G3O_c_5_MAN | MAN | NAG | 0.983 | 2 |
| 7BVG_C_1_BXY | rare | NAG | 0.983 | 2 |
| 6KGX_a2_202_PEB | PEB | CYC | 0.983 | 2 |
| 6X89_A9_401_NAP | NAP | NDP | 0.982 | 2 |
| 7Z6Q_A_805_BCL | BCL | CDL | 0.982 | 13 |
| 7Z6Q_E_407_BCL | BCL | CLA | 0.981 | 3 |
| 8G3O_s_5_MAN | MAN | NAG | 0.980 | 3 |
| 6PLT_B_511_UNL | UNL | NAG | 0.980 | 7 |
| 8AGC_L_8_GLC | rare | NAG | 0.980 | 3 |
| 7MXE_h_3_BMA | BMA | NAG | 0.977 | 2 |

## Generated files

- `summary.json`: overall metrics.
- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.
- `confusion_pairs.csv`: most frequent bidirectional class confusions.
- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.
- `confidence_bins.csv`: calibration by confidence range.
- `overview.png`: confidence, calibration, class recall, and rank plots.
- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.

Class difficulty depends on support; inspect the support column alongside recall.
