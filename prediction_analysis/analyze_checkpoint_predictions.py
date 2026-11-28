"""Generate per-checkpoint analysis reports from prediction CSV files."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "checkpoint_predictions"
DEFAULT_OUTPUT_DIR = ROOT / "prediction_analysis" / "results"
REQUIRED_COLUMNS = {
    "sample_identifier",
    "true_class",
    "predicted_class",
    "model_certainty",
    "true_class_rank",
}
RANK_CUTOFFS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing checkpoint prediction CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder in which one analysis subfolder per CSV is created.",
    )
    parser.add_argument(
        "--csv",
        action="append",
        dest="csv_files",
        help="Analyze a specific CSV (relative to --input-dir or an absolute path). "
        "May be supplied multiple times; by default all *.csv files are analyzed.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict]:
    required = REQUIRED_COLUMNS
    records = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            try:
                sample_identifier = row["sample_identifier"].strip()
                true_class = row["true_class"].strip()
                predicted_class = row["predicted_class"].strip()
                certainty = float(row["model_certainty"])
                rank_value = float(row["true_class_rank"])
                if not sample_identifier or not true_class or not predicted_class:
                    raise ValueError("sample identifier and class names must not be empty")
                if not math.isfinite(certainty) or not 0 <= certainty <= 1:
                    raise ValueError(f"model_certainty must be finite and in [0, 1], got {certainty}")
                if not math.isfinite(rank_value) or not rank_value.is_integer() or rank_value < 1:
                    raise ValueError(f"true_class_rank must be a positive integer, got {rank_value}")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid prediction row: {error}") from error

            records.append(
                {
                    "sample_identifier": sample_identifier,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "model_certainty": certainty,
                    "true_class_rank": int(rank_value),
                    "correct": true_class == predicted_class,
                }
            )

    if not records:
        raise ValueError(f"{path} contains no prediction rows")
    return records


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_predictions(csv_path: Path, output_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = read_predictions(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(
        {record["true_class"] for record in records}
        | {record["predicted_class"] for record in records}
    )
    class_records = {name: [] for name in classes}
    predicted_records = {name: [] for name in classes}
    confusion_counts = Counter()
    for record in records:
        class_records[record["true_class"]].append(record)
        predicted_records[record["predicted_class"]].append(record)
        if not record["correct"]:
            confusion_counts[(record["true_class"], record["predicted_class"])] += 1

    correct_records = [record for record in records if record["correct"]]
    incorrect_records = [record for record in records if not record["correct"]]
    accuracy = len(correct_records) / len(records)
    ranks = [record["true_class_rank"] for record in records]
    confidences = [record["model_certainty"] for record in records]

    per_class = []
    for name in classes:
        actual = class_records[name]
        predicted = predicted_records[name]
        class_correct = [row for row in actual if row["correct"]]
        class_incorrect = [row for row in actual if not row["correct"]]
        support = len(actual)
        predicted_count = len(predicted)
        true_positives = len(class_correct)
        recall = true_positives / support if support else 0.0
        precision = true_positives / predicted_count if predicted_count else 0.0
        ranks_for_class = [row["true_class_rank"] for row in actual]
        per_class.append(
            {
                "class": name,
                "support": support,
                "predicted_count": predicted_count,
                "true_positives": true_positives,
                "recall": recall,
                "precision": precision,
                "mean_true_class_rank": mean(ranks_for_class),
                "median_true_class_rank": median(ranks_for_class) if ranks_for_class else None,
                **{
                    f"true_class_rank_at_most_{cutoff}": (
                        sum(rank <= cutoff for rank in ranks_for_class) / support if support else 0.0
                    )
                    for cutoff in RANK_CUTOFFS
                },
                "mean_confidence_on_true_class": mean(
                    [row["model_certainty"] for row in actual]
                ),
                "mean_confidence_when_correct": mean(
                    [row["model_certainty"] for row in class_correct]
                ),
                "mean_confidence_when_wrong": mean(
                    [row["model_certainty"] for row in class_incorrect]
                ),
                "mean_confidence_when_predicted": mean(
                    [row["model_certainty"] for row in predicted]
                ),
            }
        )

    macro_recall = mean([row["recall"] for row in per_class if row["support"]])
    macro_precision = mean([row["precision"] for row in per_class if row["support"]])
    macro_f1 = mean(
        [
            2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"]
            else 0.0
            for row in per_class
            if row["support"]
        ]
    )

    confidence_bins = []
    expected_calibration_error = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        bucket = [
            row
            for row in records
            if lower <= row["model_certainty"] < upper
            or (bin_index == 9 and row["model_certainty"] == 1)
        ]
        bin_confidence = mean([row["model_certainty"] for row in bucket])
        bin_accuracy = mean([float(row["correct"]) for row in bucket])
        if bucket:
            expected_calibration_error += (
                len(bucket) / len(records) * abs(bin_confidence - bin_accuracy)
            )
        confidence_bins.append(
            {
                "confidence_range": f"[{lower:.1f}, {upper:.1f}]"
                if bin_index == 9
                else f"[{lower:.1f}, {upper:.1f})",
                "sample_count": len(bucket),
                "mean_confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "calibration_gap": (
                    abs(bin_confidence - bin_accuracy)
                    if bucket
                    else None
                ),
            }
        )

    class_names = [row["class"] for row in per_class]
    class_index = {name: index for index, name in enumerate(class_names)}
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for record in records:
        confusion_matrix[
            class_index[record["true_class"]], class_index[record["predicted_class"]]
        ] += 1

    paired_confusions = defaultdict(lambda: [0, 0])
    for (true_class, predicted_class), count in confusion_counts.items():
        class_a, class_b = sorted((true_class, predicted_class))
        direction_index = 0 if (true_class, predicted_class) == (class_a, class_b) else 1
        paired_confusions[(class_a, class_b)][direction_index] += count
    total_errors = len(incorrect_records)
    confusion_pairs = [
        {
            "class_a": class_a,
            "class_b": class_b,
            "a_misclassified_as_b": counts[0],
            "b_misclassified_as_a": counts[1],
            "total_confusions": sum(counts),
            "fraction_of_all_errors": sum(counts) / total_errors if total_errors else 0.0,
        }
        for (class_a, class_b), counts in paired_confusions.items()
    ]
    confusion_pairs.sort(
        key=lambda row: (-row["total_confusions"], row["class_a"], row["class_b"])
    )

    summary = {
        "prediction_file": csv_path.name,
        "sample_count": len(records),
        "class_count": len(classes),
        "correct_count": len(correct_records),
        "incorrect_count": len(incorrect_records),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mean_confidence": mean(confidences),
        "mean_confidence_when_correct": mean(
            [row["model_certainty"] for row in correct_records]
        ),
        "mean_confidence_when_wrong": mean(
            [row["model_certainty"] for row in incorrect_records]
        ),
        "expected_calibration_error_10_bins": expected_calibration_error,
        "mean_true_class_rank": mean(ranks),
        "median_true_class_rank": median(ranks),
        **{
            f"true_class_rank_at_most_{cutoff}": sum(rank <= cutoff for rank in ranks)
            / len(ranks)
            for cutoff in RANK_CUTOFFS
        },
        "most_common_predicted_class": max(
            classes,
            key=lambda name: (len(predicted_records[name]), name),
        ),
        "most_common_true_class": max(
            classes,
            key=lambda name: (len(class_records[name]), name),
        ),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
        file.write("\n")

    class_fields = [
        "class",
        "support",
        "predicted_count",
        "true_positives",
        "recall",
        "precision",
        "mean_true_class_rank",
        "median_true_class_rank",
        *(f"true_class_rank_at_most_{cutoff}" for cutoff in RANK_CUTOFFS),
        "mean_confidence_on_true_class",
        "mean_confidence_when_correct",
        "mean_confidence_when_wrong",
        "mean_confidence_when_predicted",
    ]
    write_csv(output_dir / "per_class.csv", class_fields, per_class)
    write_csv(
        output_dir / "confidence_bins.csv",
        [
            "confidence_range",
            "sample_count",
            "mean_confidence",
            "accuracy",
            "calibration_gap",
        ],
        confidence_bins,
    )
    write_csv(
        output_dir / "confusion_pairs.csv",
        [
            "class_a",
            "class_b",
            "a_misclassified_as_b",
            "b_misclassified_as_a",
            "total_confusions",
            "fraction_of_all_errors",
        ],
        confusion_pairs,
    )

    failures = sorted(
        incorrect_records,
        key=lambda row: (
            -row["model_certainty"],
            -row["true_class_rank"],
            row["sample_identifier"],
        ),
    )
    failure_fields = [
        "sample_identifier",
        "true_class",
        "predicted_class",
        "model_certainty",
        "true_class_rank",
    ]
    write_csv(
        output_dir / "worst_failures.csv",
        failure_fields,
        [{field: row[field] for field in failure_fields} for row in failures],
    )

    hardest_classes = sorted(
        (row for row in per_class if row["support"]),
        key=lambda row: (row["recall"], -row["support"], row["class"]),
    )
    easiest_classes = sorted(
        (row for row in per_class if row["support"]),
        key=lambda row: (-row["recall"], -row["support"], row["class"]),
    )
    wrong_confidence = summary["mean_confidence_when_wrong"]
    wrong_confidence_text = f"{wrong_confidence:.3f}" if wrong_confidence is not None else "n/a"
    report_lines = [
        f"# Prediction analysis: {csv_path.name}",
        "",
        "## Overall",
        "",
        f"- Samples: {len(records):,}; classes: {len(classes):,}.",
        f"- Accuracy: {accuracy:.3%}; macro precision: {macro_precision:.3%}; "
        f"macro recall: {macro_recall:.3%}; macro F1: {macro_f1:.3%}.",
        f"- Mean predicted-class confidence: {summary['mean_confidence']:.3f}; "
        f"correct: {summary['mean_confidence_when_correct']:.3f}; "
        f"incorrect: {wrong_confidence_text}.",
        f"- 10-bin expected calibration error: {expected_calibration_error:.4f}. "
        "A smaller value means confidence is better aligned with observed accuracy.",
        f"- Mean rank of the true class: {summary['mean_true_class_rank']:.2f}; "
        f"median: {summary['median_true_class_rank']:.1f}; "
        + ", ".join(
            f"rank ≤{cutoff}: {summary[f'true_class_rank_at_most_{cutoff}']:.1%}"
            for cutoff in RANK_CUTOFFS
        )
        + ".",
        "",
        "Confidence is the CSV's predicted-class probability. The CSV does not contain "
        "the full probability vector, so this report evaluates top-class calibration, "
        "not multiclass calibration.",
        "",
        "## Hardest classes",
        "",
        "| Class | Support | Recall | Precision | Mean true-class rank | Confidence when wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in hardest_classes[:15]:
        rank_text = (
            f"{row['mean_true_class_rank']:.2f}"
            if row["mean_true_class_rank"] is not None
            else "n/a"
        )
        wrong_confidence = row["mean_confidence_when_wrong"]
        wrong_text = f"{wrong_confidence:.3f}" if wrong_confidence is not None else "n/a"
        report_lines.append(
            f"| {row['class'].replace('|', '\\|')} | {row['support']} | "
            f"{row['recall']:.3f} | {row['precision']:.3f} | {rank_text} | {wrong_text} |"
        )

    report_lines.extend(
        [
            "",
            "## Easiest classes",
            "",
            "| Class | Support | Recall | Precision | Mean true-class rank | Confidence when correct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in easiest_classes[:15]:
        rank_text = (
            f"{row['mean_true_class_rank']:.2f}"
            if row["mean_true_class_rank"] is not None
            else "n/a"
        )
        correct_confidence = row["mean_confidence_when_correct"]
        correct_text = f"{correct_confidence:.3f}" if correct_confidence is not None else "n/a"
        report_lines.append(
            f"| {row['class'].replace('|', '\\|')} | {row['support']} | "
            f"{row['recall']:.3f} | {row['precision']:.3f} | {rank_text} | {correct_text} |"
        )

    report_lines.extend(
        [
            "",
            "## Most confused class pairs",
            "",
            "| Class pair | A → B | B → A | Total | Share of errors |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in confusion_pairs[:15]:
        report_lines.append(
            f"| {row['class_a'].replace('|', '\\|')} ↔ {row['class_b'].replace('|', '\\|')} "
            f"| {row['a_misclassified_as_b']} | {row['b_misclassified_as_a']} "
            f"| {row['total_confusions']} | {row['fraction_of_all_errors']:.1%} |"
        )

    report_lines.extend(
        [
            "",
            "## Highest-confidence failures",
            "",
            "| Sample | True class | Predicted class | Confidence | True-class rank |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in failures[:15]:
        report_lines.append(
            f"| {row['sample_identifier'].replace('|', '\\|')} "
            f"| {row['true_class'].replace('|', '\\|')} "
            f"| {row['predicted_class'].replace('|', '\\|')} "
            f"| {row['model_certainty']:.3f} | {row['true_class_rank']} |"
        )

    report_lines.extend(
        [
            "",
            "## Generated files",
            "",
            "- `summary.json`: overall metrics.",
            "- `per_class.csv`: class-level recall, precision, confidence, and true-class rank.",
            "- `confusion_pairs.csv`: most frequent bidirectional class confusions.",
            "- `worst_failures.csv`: all incorrect samples, ordered by predicted confidence.",
            "- `confidence_bins.csv`: calibration by confidence range.",
            "- `overview.png`: confidence, calibration, class recall, and rank plots.",
            "- `confusion_matrix_top_classes.png`: normalized confusion matrix for the 20 most-supported classes.",
            "",
            "Class difficulty depends on support; inspect the support column alongside recall.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(
        [
            [row["model_certainty"] for row in correct_records],
            [row["model_certainty"] for row in incorrect_records],
        ],
        bins=np.linspace(0, 1, 21),
        label=["Correct", "Incorrect"],
        stacked=True,
    )
    axes[0, 0].set(title="Predicted-class confidence", xlabel="Confidence", ylabel="Samples")
    axes[0, 0].legend()

    populated_bins = [row for row in confidence_bins if row["sample_count"]]
    axes[0, 1].plot(
        [row["mean_confidence"] for row in populated_bins],
        [row["accuracy"] for row in populated_bins],
        marker="o",
        label="Observed accuracy",
    )
    axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal calibration")
    axes[0, 1].set(
        title="Top-class calibration",
        xlabel="Mean confidence",
        ylabel="Accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0, 1].legend()

    axes[1, 0].scatter(
        [row["support"] for row in per_class if row["support"]],
        [row["recall"] for row in per_class if row["support"]],
        alpha=0.65,
    )
    axes[1, 0].set(
        title="Class recall vs support",
        xlabel="True-class sample count",
        ylabel="Recall",
        ylim=(-0.03, 1.03),
    )
    axes[1, 0].set_xscale("log")

    max_rank = max(ranks)
    rank_bins = np.arange(0.5, max_rank + 1.5, 1)
    axes[1, 1].hist(ranks, bins=rank_bins)
    axes[1, 1].set(
        title="Rank of the true class",
        xlabel="True-class rank",
        ylabel="Samples",
        xlim=(0.5, min(max_rank + 0.5, 30.5)),
    )
    fig.suptitle(csv_path.name)
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=140)
    plt.close(fig)

    top_classes = sorted(classes, key=lambda name: (-len(class_records[name]), name))[:20]
    indices = [class_index[name] for name in top_classes]
    matrix = confusion_matrix[np.ix_(indices, indices)].astype(float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(max(8, len(top_classes) * 0.55), max(7, len(top_classes) * 0.5)))
    image = ax.imshow(normalized_matrix, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_classes)), top_classes, rotation=90)
    ax.set_yticks(range(len(top_classes)), top_classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Row-normalized confusion: 20 most-supported classes")
    fig.colorbar(image, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_top_classes.png", dpi=140)
    plt.close(fig)

    return summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.csv_files:
        csv_paths = [
            Path(name) if Path(name).is_absolute() else input_dir / name
            for name in args.csv_files
        ]
    else:
        csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No prediction CSV files found in {input_dir}")

    for csv_path in csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        result_dir = output_dir / csv_path.stem
        summary = analyze_predictions(csv_path, result_dir)
        print(
            f"{csv_path.name}: {summary['sample_count']} samples, "
            f"accuracy={summary['accuracy']:.3%} -> {result_dir}"
        )


if __name__ == "__main__":
    main()
