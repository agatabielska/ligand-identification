from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset

from src.models.clifford.model import CliffordSteerableNetwork
from src.models.e3nn.configurable_spectral import ConfigurableSpectralModel
from src.pipeline.data_loader import build_data, validate_data_dir
from src.utils.set_seed import set_seed
    
    
def build_spectral_training(cfg: DictConfig):
    loaders = build_data(cfg, "spectral")
    train_loader, validation_loader, test_loader = loaders

    model_cfg, train_cfg = cfg.model, cfg.train
    model = ConfigurableSpectralModel(
        num_classes=int(train_cfg.out_channels),
        **OmegaConf.to_container(model_cfg, resolve=True),
    )
    model.set_class_counts(torch.bincount(train_loader.y, minlength=model.num_classes))

    # train loader is of form (x, y, global_values), where global_values is a tensor of shape (N, 12)
    global_train = torch.cat([batch[2] for batch in train_loader], dim=0)
    global_mean = global_train.mean(0)
    global_std = global_train.std(0).clamp_min(1e-6)
    model.global_mean.copy_(global_mean)
    model.global_std.copy_(global_std)
    return model, loaders


def build_clifford_training(cfg: DictConfig):
    train_loader, validation_loader, test_loader = build_data(cfg, "clifford")
    model_cfg, train_cfg = cfg.model, cfg.train
    model = CliffordSteerableNetwork(
        p=model_cfg.p,
        q=model_cfg.q,
        in_channels=model_cfg.in_channels,
        hidden_channels=model_cfg.hidden_channels,
        out_channels=train_cfg.out_channels,
        n_shells=model_cfg.n_shells,
        kernel_size=model_cfg.kernel_size,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    return model, (train_loader, validation_loader, test_loader)


TRAINING_BUILDERS = {
    "spectral": build_spectral_training,
    "clifford": build_clifford_training,
}


def run_training(cfg: DictConfig):
    model_type = cfg.model.type
    if model_type not in TRAINING_BUILDERS:
        raise ValueError(f"Unknown model type: {model_type}")

    model, (train_loader, validation_loader, test_loader) = TRAINING_BUILDERS[model_type](cfg)
    logger_kwargs = {
        "project": "ligand-identification",
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cfg.wandb.run_name is not None:
        logger_kwargs["name"] = cfg.wandb.run_name
    logger = WandbLogger(**logger_kwargs)
    checkpoint_dir = Path(cfg.paths.model_checkpoint) / logger.experiment.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = cfg.train.early_stopping
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=[
            ModelCheckpoint(
                monitor=early_stopping.monitor,
                mode=early_stopping.mode,
                dirpath=str(checkpoint_dir),
                filename="model-{epoch:02d}-{val_loss:.2f}",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor=early_stopping.monitor,
                mode=early_stopping.mode,
                patience=early_stopping.patience,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        accelerator="auto",
        devices=cfg.machine.devices,
        precision=cfg.train.get("precision", "32-true"),
        log_every_n_steps=10,
        logger=logger,
        gradient_clip_val=cfg.train.gradient_clip_val,
    )
    trainer.fit(model, train_loader, validation_loader)
    trainer.test(ckpt_path="best", dataloaders=test_loader)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.random_seed)
    validate_data_dir(Path(cfg.paths.data_dir))
    run_training(cfg)


if __name__ == "__main__":
    main()
