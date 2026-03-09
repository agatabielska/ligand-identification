from src.utils.sampling_strategies import (
    UniformSelectionTransform,
    ProbabilisticSelectionTransform,
    SpatialNormalization3,
)
from concurrent.futures import ThreadPoolExecutor, wait
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple
from pathlib import Path
import numpy as np
import argparse


def extract_pdb_id(file_path: Path) -> str:
    """Extract PDB ID (molecule) from filename - first part of file name."""
    return file_path.stem.split("_")[0].upper()


def load_folders(folder_paths: list[Path]) -> Tuple[List[Path], List[str], List[str]]:
    """
    Load all NPZ files from the given folders which are ligand classes, and extracts molecule IDs from filenames.
    Returns three parallel lists: file paths, ligand class labels, and molecule IDs (PDB IDs) for stratified group splitting.
    """

    for folder_path in folder_paths:
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")

    all_files_paths:  List[Path] = []
    ligand_names: List[str]  = []
    pdb_ids: List[str]  = []

    for folder_path in folder_paths:
        for ligand_dir in sorted(folder_path.iterdir()):
            if not ligand_dir.is_dir():
                continue
            ligand_name = ligand_dir.name
            for f in sorted(ligand_dir.glob("*.npz")):
                all_files_paths.append(f)
                ligand_names.append(ligand_name)
                pdb_ids.append(extract_pdb_id(f))

    return all_files_paths, ligand_names, pdb_ids


def stratified_group_split(
    files: List[Path],
    labels: List[str],
    groups: List[str],
    n_splits: int,
    random_state: int,
) -> Tuple[List[Path], List[str], List[Path], List[str], List[Path], List[str]]:
    
    files_arr  = np.array(files,  dtype=object)
    ligand_names = np.array(labels, dtype=object)
    pdb_ids = np.array(groups, dtype=object)

    # Splits based on groups (PDB IDs) and stratified by ligand names
    iterator = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(iterator.split(files_arr, ligand_names, pdb_ids))

    # Split into n_splits folds, takes fold 0 as test, fold 1 as val, and the rest as train
    test_idx  = folds[0][1]
    val_idx   = folds[1][1]

    # Train: remaining files not in test or val
    exclude   = set(test_idx) | set(val_idx)
    train_idx = [i for i in range(len(files_arr)) if i not in exclude]

    train_files  = files_arr[train_idx]
    train_labels = ligand_names[train_idx]
    val_files    = files_arr[val_idx]
    val_labels   = ligand_names[val_idx]
    test_files   = files_arr[test_idx]
    test_labels  = ligand_names[test_idx]

    return (
        list(train_files), list(train_labels),
        list(val_files),   list(val_labels),
        list(test_files),  list(test_labels),
    )




def preprocess(
    file_paths: List[Path],
    file_labels: List[str],
    output_path: Path,
    transform_stack: str = "probabilistic",
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

    for file_path, ligand_name in zip(file_paths, file_labels):
        data = np.load(file_path)
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(
                f"Expected one key in NPZ file, found {len(keys)} keys in {file_path}"
            )

        blob = data[keys[0]]
        for transform in transforms:
            blob = transform.preprocess(blob)
        idx    = np.argwhere(blob > 0)
        values = blob[blob > 0]
        processed_data = {"indices": idx, "values": values, "shape": blob.shape}

        (output_path / ligand_name).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path / ligand_name / file_path.name, **processed_data)

    print(f"Processed {len(file_paths)} files and saved to {output_path}")


def chunk(items, labels, size):
    for i in range(0, len(items), size):
        yield items[i : i + size], labels[i : i + size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess data and save it to a file."
    )
    parser.add_argument(
        "--folders",
        type=str,
        required=True,
        nargs="+",
        help="Paths to data folders (each containing ligand class subdirs).",
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
        "--n-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedGroupKFold.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load all files into flat lists
    all_files, all_labels, all_groups = load_folders([Path(p) for p in args.folders])
    print(f"Total files   : {len(all_files)}")
    print(f"Total classes : {len(set(all_labels))}")
    print(f"Unique PDB IDs: {len(set(all_groups))}")

    # Splits into train/val/test with stratification by ligand class and grouping by PDB ID
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        stratified_group_split(
            all_files, all_labels, all_groups,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train_files)} files  ({len(set(train_labels))} classes)")
    print(f"  Val   : {len(val_files)} files  ({len(set(val_labels))} classes)")
    print(f"  Test  : {len(test_files)} files  ({len(set(test_labels))} classes)")

    futures = []
    with ThreadPoolExecutor() as pool:
        for sublist, sublabels in chunk(train_files, train_labels, args.chunk_size):
            futures.append(pool.submit(
                preprocess, sublist, sublabels,
                output_folder / "train",
                transform_stack=args.transform_stack,
            ))
        for sublist, sublabels in chunk(val_files, val_labels, args.chunk_size):
            futures.append(pool.submit(
                preprocess, sublist, sublabels,
                output_folder / "val",
                transform_stack=args.transform_stack,
            ))
        for sublist, sublabels in chunk(test_files, test_labels, args.chunk_size):
            futures.append(pool.submit(
                preprocess, sublist, sublabels,
                output_folder / "test",
                transform_stack=args.transform_stack,
            ))
    wait(futures)