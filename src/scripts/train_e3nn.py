import os
import sys
from pathlib import Path

# Add project root to path and change to project root directory
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Now import after path is set
import torch
import numpy as np
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.models.e3nn.model import E3NNPointCloudModel
from src.pipeline.dataset import BlobDataset

def transform(npz):
    """
    Transform preprocessed NPZ file to point cloud format.

    Args:
        npz: NPZ file containing 'indices', 'values', and 'shape'

    Returns:
        torch.Tensor: (max_points, 4) tensor with [x, y, z, density]
    """
    MAX_POINTS = 2000

    indices = npz["indices"]
    values = npz["values"]

    if len(indices) == 0:
        return torch.zeros((MAX_POINTS, 4), dtype=torch.float32)

    # Normalize coordinates
    coords = indices.astype(np.float32)

    # Compute center of mass
    center = coords.mean(axis=0)
    centered_coords = coords - center

    # Robust scaling using 95th percentile
    scale = np.percentile(np.linalg.norm(centered_coords, axis=1), 95)
    scale = max(scale, 1.0)  # Prevent division by zero
    norm_coords = centered_coords / scale

    # Combine coordinates and values
    points = np.column_stack([norm_coords, values])

    # Pad or truncate to MAX_POINTS
    current_points = points.shape[0]
    if current_points < MAX_POINTS:
        padding = np.zeros((MAX_POINTS - current_points, 4), dtype=np.float32)
        points = np.vstack([points, padding])
    elif current_points > MAX_POINTS:
        points = points[:MAX_POINTS, :]

    return torch.from_numpy(points.astype(np.float32))


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize logger
    wandb_logger = WandbLogger(
        project="ligand-identification-e3nn",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Load datasets
    train_dataset = BlobDataset(
        path=cfg.paths.train_data,
        transform=transform,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
    )

    val_dataset = BlobDataset(
        path=cfg.paths.val_data,
        transform=transform,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=True,
        num_workers=cfg.machine.num_workers if not cfg.machine.cache_dataset else 1,
        pin_memory=cfg.machine.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=False,
        num_workers=cfg.machine.num_workers if not cfg.machine.cache_dataset else 1,
        pin_memory=cfg.machine.pin_memory,
    )

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.paths.model_checkpoint,
        filename="e3nn-model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
        save_top_k=3,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=cfg.machine.devices,
        log_every_n_steps=10,
        logger=wandb_logger,
        gradient_clip_val=1.0,  # Gradient clipping
    )

    # Initialize E3NN model
    model = E3NNPointCloudModel(
        num_classes=cfg.train.out_channels,
        max_points=2000,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.get('train', {}).get('weight_decay', 1e-4),
    )

    # Print model summary
    print("\n" + "=" * 70)
    print("E3NN Point Cloud Model")
    print("=" * 70)
    print(f"Number of classes: {cfg.train.out_channels}")
    print(f"Max points: 2000")
    print(f"Learning rate: {cfg.train.learning_rate}")
    print(f"\nIrreps structure:")
    print(f"  Input:    {model.irreps_in}")
    print(f"  Hidden1:  {model.irreps_hidden1}")
    print(f"  Hidden2:  {model.irreps_hidden2}")
    print(f"  Hidden3:  {model.irreps_hidden3}")
    print(f"  Output:   {model.irreps_scalar}")
    print(f"\nPooling strategy: Multi-scale (max + mean + std)")
    print(f"  Pooled features: 256 × 3 = 768")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70 + "\n")

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()