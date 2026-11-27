"""Export per-sample test predictions for the finalized HPO checkpoints.

Run with ``uv run -m src.scripts.evaluate_checkpoints`` after all checkpoint
directories contain their ``final-epoch=*.ckpt`` file.
"""

import argparse
import csv
import gc
import re
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.e3nn.configurable_spectral import ConfigurableSpectralModel
from src.models.radialCNN.model import RadialShellCNN1d
from src.pipeline.data_loader import (
    get_dataset,
    transform_e3nn,
    transform_radialCNN,
)
from src.pipeline.global_statistics import complete_input_statistics


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
CHECKPOINT_PATTERN = re.compile(
    r"^final-hpo-(?P<model>radialCNN|spectral)-(?P<data>TP|TU|UP)"
    r"(?:-min_val_loss)?$"
)
DATASETS = {
    "TP": "thresholded_transform_probabilistic",
    "TU": "thresholded_transform_uniform",
    "UP": "unthresholded_transform_probabilistic",
}
MODEL_CLASSES = {
    "radialCNN": RadialShellCNN1d,
    "spectral": ConfigurableSpectralModel,
}
CSV_COLUMNS = (
    "sample_identifier",
    "true_class",
    "predicted_class",
    "model_certainty",
    "true_class_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "checkpoint_predictions"
    )
    return parser.parse_args()


def checkpoint_jobs(checkpoints_dir: Path):
    directories = sorted(path for path in checkpoints_dir.iterdir() if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")

    jobs = []
    for directory in directories:
        match = CHECKPOINT_PATTERN.fullmatch(directory.name)
        if match is None:
            raise ValueError(f"Unrecognized checkpoint directory: {directory}")
        checkpoints = sorted(directory.glob("final-epoch=*.ckpt"))
        if len(checkpoints) != 1:
            raise FileNotFoundError(
                f"Expected exactly one final-epoch=*.ckpt in {directory}, found {len(checkpoints)}"
            )
        jobs.append((directory.name, match.group("model"), match.group("data"), checkpoints[0]))
    return jobs


def build_test_loader(cfg, data_dir: Path, model_type: str):
    split_dirs = [data_dir / split for split in ("train", "val", "test")]
    missing = [path for path in split_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Dataset must contain train/, val/, and test/ directories; missing: "
            + ", ".join(map(str, missing))
        )

    classes = sorted(
        {
            class_dir.name
            for split_dir in split_dirs
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        }
    )
    label_to_idx = {name: index for index, name in enumerate(classes)}

    if model_type == "radialCNN":
        transform = lambda npz: transform_radialCNN(npz, cfg.data.max_points)
        global_transform = None
    else:
        transform = lambda npz: transform_e3nn(npz, cfg.data.max_points)
        global_transform = complete_input_statistics

    cfg.machine.cache_dataset = False
    cfg.machine.num_workers = 0
    dataset = get_dataset(
        data_dir / "test",
        cfg,
        transform,
        label_to_idx,
        global_transform=global_transform,
    )
    if not dataset.samples:
        raise ValueError(f"No test samples found in {data_dir / 'test'}")
    dataset.samples.sort(key=lambda sample: (sample[0].stem, sample[0].parent.name))
    loader = DataLoader(
        dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader, classes


@torch.inference_mode()
def export_predictions(
    model: pl.LightningModule,
    loader: DataLoader,
    classes: list[str],
    model_type: str,
    checkpoint_name: str,
    output_dir: Path,
    device: torch.device,
) -> Path:
    model.to(device).eval()
    rows = []
    dataset_samples = loader.dataset.samples
    sample_offset = 0

    for batch in loader:
        x, y, *rest = batch
        x, y = x.to(device), y.to(device).reshape(-1)
        if model_type == "spectral":
            global_values = rest[0].to(device)
            if model.global_branch is not None:
                global_values = (global_values - model.global_mean) / model.global_std
                global_values = global_values[:, model.global_columns]
            logits = model(x, global_values=global_values).float()
        else:
            logits = model(x).float()

        probabilities = F.softmax(logits, dim=1)
        predicted = probabilities.argmax(dim=1)
        ranks = (
            torch.argsort(logits, dim=1, descending=True, stable=True)
            .eq(y[:, None])
            .to(torch.int64)
            .argmax(dim=1)
            .add(1)
        )

        for offset in range(len(y)):
            sample_path = dataset_samples[sample_offset][0]
            true_index = int(y[offset].item())
            predicted_index = int(predicted[offset].item())
            rows.append(
                {
                    "sample_identifier": sample_path.stem,
                    "true_class": classes[true_index],
                    "predicted_class": classes[predicted_index],
                    "model_certainty": float(probabilities[offset, predicted_index].item()),
                    "true_class_rank": int(ranks[offset].item()),
                }
            )
            sample_offset += 1

    rows.sort(key=lambda row: (row["sample_identifier"], row["true_class"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{checkpoint_name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def evaluate_checkpoint(
    checkpoint_name: str,
    model_type: str,
    dataset_key: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    data_dir = args.data_root / DATASETS[dataset_key]
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"model={model_type}",
                f"paths.data_dir={data_dir}",
                "machine.cache_dataset=false",
                "machine.num_workers=0",
                "machine.use_gpu_loader=false",
            ],
        )
    cfg.machine.cache_dataset = False
    cfg.machine.num_workers = 0
    loader, classes = build_test_loader(cfg, data_dir, model_type)
    model = MODEL_CLASSES[model_type].load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    output_path = export_predictions(
        model,
        loader,
        classes,
        model_type,
        checkpoint_name,
        args.output_dir,
        device,
    )
    print(f"Wrote {len(loader.dataset)} predictions to {output_path}")


def main() -> None:
    args = parse_args()
    jobs = checkpoint_jobs(args.checkpoints_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for checkpoint_name, model_type, dataset_key, checkpoint_path in jobs:
        evaluate_checkpoint(
            checkpoint_name,
            model_type,
            dataset_key,
            checkpoint_path,
            args,
            device,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
