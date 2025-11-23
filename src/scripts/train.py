from src.models.clifford.model import CliffordSteerableNetwork
from src.pipeline.dataset import BlobDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
import numpy as np
import hydra
import torch
import sys


def transform(npz):
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


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
    sys.exit(0)

    train_dataset = BlobDataset(
        path="data/processed/train",
        transform=transform,
        cache=cfg.machine.cache_dataset,
    )

    val_dataset = BlobDataset(
        path="data/processed/test", transform=transform, cache=cfg.machine.cache_dataset
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=True,
        num_workers=cfg.machine.num_workers,
        pin_memory=cfg.machine.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=False,
        num_workers=cfg.machine.num_workers,
        pin_memory=cfg.machine.pin_memory,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=cfg.machine.devices,
        log_every_n_steps=10,
    )

    if cfg.model.type == "clifford":
        model = CliffordSteerableNetwork(
            p=cfg.model.p,
            q=cfg.model.q,
            in_channels=cfg.model.in_channels,
            hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.train.out_channels,
            n_shells=cfg.model.n_shells,
            kernel_size=cfg.model.kernel_size,
            learning_rate=cfg.train.learning_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
