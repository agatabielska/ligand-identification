from src.models.clifford.model import CliffordSteerableNetwork
from src.models.e3nn.model import E3NNPointCloudModel
from src.pipeline.dataset import BlobDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from src.utils.set_seed import set_seed
from pathlib import Path
import numpy as np
import hydra
import torch
import torch.nn.functional as F
import json


def load_class_mapping(mapping_file: Path):
    """Load class mapping from JSON file."""
    if not mapping_file.exists():
        print(f"Class mapping file not found at {mapping_file}")
        return None
    
    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    return {int(k): v for k, v in mapping.items()}


def transform_clifford(npz):
    indices = npz["indices"]
    points = np.pad(
        indices,
        ((0, 2000 - indices.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    points = points.reshape(5, 20, 20, 3).transpose(3, 0, 1, 2)
    points = points.astype(np.float32)
    return torch.from_numpy(points)


def transform_e3nn(npz):
    indices = npz["indices"]
    values = npz["values"]
    coords = indices.astype(np.float32)
    values = values.astype(np.float32)
    points = np.column_stack([coords, values])
    current_points = points.shape[0]

    if current_points < 2000:
        padding = np.zeros((2000 - current_points, 4), dtype=np.float32)
        points = np.vstack([points, padding])

    return torch.from_numpy(points)


def get_dataset(path: str, cfg, transform):
    return BlobDataset(
        path=path,
        transform=transform,
        normalize=cfg.train.normalize_data,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
    )


def get_dataloader(dataset, cfg, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=shuffle,
        num_workers=cfg.machine.num_workers if not cfg.machine.cache_dataset else 1,
        pin_memory=cfg.machine.pin_memory,
        persistent_workers=True if cfg.machine.num_workers > 0 else False,
    )


def verbose_predictions(model, dataloader, device, dataset, class_mapping=None):
    """
    Run predictions with detailed output for each sample.
    """
    model.eval()
    model.to(device)
    
    all_correct = 0
    all_total = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(logits, dim=1)
            predicted_probs = torch.max(probs, dim=1)[0]
            
            top10_probs, top10_classes = torch.topk(probs, k=min(10, probs.shape[1]), dim=1)
            
            batch_size = x.shape[0]
            for i in range(batch_size):
                sample_idx = batch_idx * dataloader.batch_size + i
                
                try:
                    filename = dataset.file_list[sample_idx]
                    sample_name = Path(filename).stem
                except:
                    sample_name = f"sample_{sample_idx}"
                
                true_class = y[i].item()
                pred_class = predicted_classes[i].item()
                pred_prob = predicted_probs[i].item()
                is_correct = pred_class == true_class
                
                all_total += 1
                if is_correct:
                    all_correct += 1
                
                true_name = class_mapping.get(true_class, str(true_class)) if class_mapping else str(true_class)
                pred_name = class_mapping.get(pred_class, str(pred_class)) if class_mapping else str(pred_class)
                
                status = "CORRECT" if is_correct else "WRONG"
                print(f"\n[{sample_idx + 1}] {sample_name}")
                print(f"Result: {status}")
                print(f"True: {true_name} (class {true_class}) | Predicted: {pred_name} (class {pred_class}")
                print(f"Top 10:")
                
                for rank, (prob, cls) in enumerate(zip(top10_probs[i], top10_classes[i]), 1):
                    cls_idx = cls.item()
                    cls_name = class_mapping.get(cls_idx, str(cls_idx)) if class_mapping else str(cls_idx)
                    marker = "*" if cls_idx == true_class else " "
                    print(f"  {rank:2d}. {cls_name:30s} (class {cls_idx:3d}): {prob.item():.4f} {marker}")
    
    accuracy = all_correct / all_total if all_total > 0 else 0
    print(f"\nTotal: {all_correct}/{all_total} correct ({accuracy:.2%})")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not cfg.test.checkpoint_path:
        raise ValueError("Checkpoint path not set in config (test.checkpoint_path)")

    set_seed(cfg.random_seed)

    # Load class mapping, if not provided in config, try default location - parent of test data folder
    class_mapping = None
    if cfg.test.class_mapping_path:
        mapping_file = Path(cfg.test.class_mapping_path)
        class_mapping = load_class_mapping(mapping_file)
        if class_mapping:
            print(f"Loaded class mapping with {len(class_mapping)} classes from {mapping_file}")
    else:
        data_root = Path(cfg.paths.test_data).parent
        mapping_file = data_root / "class_mapping.json"
        class_mapping = load_class_mapping(mapping_file)
        if class_mapping:
            print(f"Loaded class mapping with {len(class_mapping)} classes from {mapping_file}")
        else:
            print("No class mapping provided or found, using numeric labels")

    wandb_logger = None
    if cfg.get('use_wandb', False):
        wandb_logger = WandbLogger(
            project="ligand-identification",
            name="test_run",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    if cfg.model.type == "clifford":
        transform = transform_clifford
    elif cfg.model.type == "e3nn":
        transform = transform_e3nn
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    test_dataset = get_dataset(cfg.paths.test_data, cfg, transform)
    test_dataloader = get_dataloader(test_dataset, cfg, shuffle=False)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=cfg.machine.devices,
        logger=wandb_logger,
    )

    checkpoint_path = Path(cfg.test.checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if cfg.model.type == "clifford":
        model = CliffordSteerableNetwork.load_from_checkpoint(
            str(checkpoint_path),
            p=cfg.model.p,
            q=cfg.model.q,
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.train.out_channels,
            n_shells=cfg.model.n_shells,
            kernel_size=cfg.model.kernel_size,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.model.type == "e3nn":
        model = E3NNPointCloudModel.load_from_checkpoint(
            str(checkpoint_path),
            num_classes=cfg.train.out_channels,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    print(f"Model loaded. Testing on {len(test_dataset)} samples...")

    if cfg.test.verbose:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verbose_predictions(model, test_dataloader, device, test_dataset, class_mapping)

    results = trainer.test(model, dataloaders=test_dataloader)

    return results


if __name__ == "__main__":
    main()