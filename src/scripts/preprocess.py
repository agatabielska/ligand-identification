from src.utils.sampling_strategies import ProbabilisticSelectionTransform
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from pathlib import Path
import numpy as np
import argparse


def load_folder(folder_path: Path) -> Tuple[List[Path], List[int]]:
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    file_paths = []
    labels = []
    class_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda x: x.name)
    name_to_label = {d.name: idx for idx, d in enumerate(class_dirs)}

    for class_dir in class_dirs:
        label = name_to_label[class_dir.name]
        files = list(class_dir.glob("*.npz"))
        file_paths.extend(files)
        labels.extend([label] * len(files))

    return file_paths, labels


def preprocess(
    file_paths: List[Path], file_labels: List[int], output_path: Path
) -> None:
    # TODO: Change this if needed
    transform = ProbabilisticSelectionTransform(max_blob_size=2000)
    for file_path, label in zip(file_paths, file_labels):
        data = np.load(file_path)
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(
                f"Expected one key in NPZ file, found {len(keys)} keys in {file_path}"
            )

        blob = data[keys[0]]
        blob = transform.preprocess(blob)
        idx = np.argwhere(blob != 0)
        values = blob[blob != 0]
        processed_data = {"indices": idx, "values": values, "shape": blob.shape}
        (output_path / str(label)).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path / str(label) / file_path.name, **processed_data)

    print(f"Processed {len(file_paths)} files and saved to {output_path}")


def chunk(items, labels, size):
    for i in range(0, len(items), size):
        yield items[i : i + size], labels[i : i + size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess data and save it to a file."
    )
    parser.add_argument(
        "--train-folder",
        type=str,
        required=True,
        help="Path to the training data folder.",
    )
    parser.add_argument(
        "--test-folder",
        type=str,
        required=True,
        help="Path to the testing data folder.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of files to process in each chunk.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder where preprocessed data will be saved.",
    )
    args = parser.parse_args()

    train_folder = Path(args.train_folder)
    test_folder = Path(args.test_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    train_files, train_labels = load_folder(train_folder)
    test_files, test_labels = load_folder(test_folder)

    print(
        f"Found {len(train_files)} training files and {len(test_files)} testing files."
    )

    with ThreadPoolExecutor() as pool:
        for sublist, sublabels in chunk(train_files, train_labels, args.chunk_size):
            pool.submit(preprocess, sublist, sublabels, output_folder / "train")
        for sublist, sublabels in chunk(test_files, test_labels, args.chunk_size):
            pool.submit(preprocess, sublist, sublabels, output_folder / "test")
