from src.utils.sampling_strategies import (
    UniformSelectionTransform,
    ProbabilisticSelectionTransform,
    SpatialNormalization3,
)
from concurrent.futures import ThreadPoolExecutor, wait
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import List, Tuple
from pathlib import Path
import numpy as np
import argparse


def load_folder(folder_paths: List[Path]) -> Tuple[List[Path], List[int]]:
    for folder_path in folder_paths:
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

    file_paths = []
    labels = []
    class_dirs = []
    for folder_path in folder_paths:
        class_dirs.extend(
            [
                d
                for d in folder_path.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

    unique_class_names = sorted(list(set(d.name.split("/")[-1] for d in class_dirs)))
    class_name_to_label = {name: idx for idx, name in enumerate(unique_class_names)}
    dir_to_label = {
        d.name: class_name_to_label[d.name.split("/")[-1]] for d in class_dirs
    }

    for class_dir in class_dirs:
        label = dir_to_label[class_dir.name]
        files = list(class_dir.glob("*.npz"))
        file_paths.extend(files)
        labels.extend([label] * len(files))

    return file_paths, labels


def preprocess(
    file_paths: List[Path],
    file_labels: List[int],
    output_path: Path,
    transform_stack="probabilistic",
) -> None:
    if transform_stack == "uniform":
        transforms = [UniformSelectionTransform(max_blob_size=2000, method="max")]
    elif transform_stack == "probabilistic":
        transforms = [ProbabilisticSelectionTransform(max_blob_size=2000)]
    elif transform_stack == "normalization":
        transforms = [
            SpatialNormalization3(),
            ProbabilisticSelectionTransform(max_blob_size=2000),
        ]
    else:
        raise ValueError(f"Unknown transform stack: {transform_stack}")

    for file_path, label in zip(file_paths, file_labels):
        data = np.load(file_path)
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(
                f"Expected one key in NPZ file, found {len(keys)} keys in {file_path}"
            )

        blob = data[keys[0]]
        for transform in transforms:
            blob = transform.preprocess(blob)
        idx = np.argwhere(blob > 0)
        values = blob[blob > 0]
        processed_data = {"indices": idx, "values": values, "shape": blob.shape}
        (output_path / str(label)).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path / str(label) / file_path.name, **processed_data)

    print(f"Processed {len(file_paths)} files and saved to {output_path}")


def chunk(items, labels, size):
    for i in range(0, len(items), size):
        yield items[i : i + size], labels[i : i + size]


def handle_single_example(
    file_paths: List[Path], file_labels: List[int], strategy: str = "remove"
) -> Tuple[List[Path], List[int]]:
    new_paths = []
    new_labels = []
    counter = Counter(file_labels)

    for path, label in zip(file_paths, file_labels):
        if counter[label] == 1:
            print(f"Class {label} has a single example: {path}")
            if strategy == "remove":
                continue
            elif strategy == "duplicate":
                new_paths.append(path)
                new_labels.append(label)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        new_paths.append(path)
        new_labels.append(label)

    return new_paths, new_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess data and save it to a file."
    )
    parser.add_argument(
        "--train-folders",
        type=str,
        required=True,
        help="List of paths to the training data folders. Folders should be comma separated.",
    )
    parser.add_argument(
        "--test-folders",
        type=str,
        required=True,
        help="List of paths to the testing data folders. Folders should be comma separated.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of files to process in each chunk.",
    )
    parser.add_argument(
        "--transform-stack",
        type=str,
        default="probabilistic",
        help="Transform stack to apply during preprocessing.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder where preprocessed data will be saved.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Proportion of test data."
    )
    parser.add_argument(
        "--single-example-strategy",
        type=str,
        choices=["remove", "duplicate"],
        default="remove",
        help="Strategy to handle classes with a single example.",
    )
    args = parser.parse_args()

    train_folders = [Path(p.strip()) for p in args.train_folders.split(",")]
    test_folders = [Path(p.strip()) for p in args.test_folders.split(",")]
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    train_files, train_labels = load_folder(train_folders)
    test_files, test_labels = load_folder(test_folders)

    train_files, train_labels = handle_single_example(
        train_files, train_labels, strategy=args.single_example_strategy
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files,
        train_labels,
        test_size=args.test_size,
        stratify=train_labels,
        random_state=42,
    )

    print(
        f"Found {len(train_files)} training files and {len(test_files)} testing files."
    )

    futures = []
    with ThreadPoolExecutor() as pool:
        for sublist, sublabels in chunk(train_files, train_labels, args.chunk_size):
            futures.append(
                pool.submit(
                    preprocess,
                    sublist,
                    sublabels,
                    output_folder / "train",
                    transform_stack=args.transform_stack,
                )
            )
        for sublist, sublabels in chunk(val_files, val_labels, args.chunk_size):
            futures.append(
                pool.submit(
                    preprocess,
                    sublist,
                    sublabels,
                    output_folder / "val",
                    transform_stack=args.transform_stack,
                )
            )
        for sublist, sublabels in chunk(test_files, test_labels, args.chunk_size):
            futures.append(
                pool.submit(
                    preprocess,
                    sublist,
                    sublabels,
                    output_folder / "test",
                    transform_stack=args.transform_stack,
                )
            )
    wait(futures)
