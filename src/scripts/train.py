from src.models.clifford.model import CliffordSteerableNetwork
from src.pipeline.dataset import BlobDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import torch


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


train_dataset = BlobDataset(
    path="data/processed/train", transform=transform, cache=True
)
val_dataset = BlobDataset(path="data/processed/test", transform=transform, cache=True)

train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=64, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=64, pin_memory=True
)


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    mode="min",
)

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[checkpoint_callback],
    accelerator="auto",
    devices=1,
    log_every_n_steps=10,
)

model = CliffordSteerableNetwork(
    p=3,
    q=0,
    in_channels=3,
    hidden_channels=[32, 64, 128],
    out_channels=219,
    n_shells=3,
    kernel_size=3,
    learning_rate=1e-3,
)


trainer.fit(model, train_dataloader, val_dataloader)
