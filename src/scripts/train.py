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
from src.pipeline.global_statistics import (
    complete_input_statistics_for_splits,
    point_statistics,
)
from src.utils.set_seed import set_seed


def build_clifford_model(cfg: DictConfig):
    model_cfg, train_cfg = cfg.model, cfg.train
    return CliffordSteerableNetwork(
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


def _spectral_global_features(cfg: DictConfig, loaders, splits=("train", "val", "test")):
    if cfg.global_statistics.source == "points_2000":
        return tuple(point_statistics(loader.x) for loader in loaders)
    if cfg.global_statistics.source == "complete_input":
        data_dir = Path(cfg.paths.data_dir)
        classes = sorted(
            directory.name
            for split in ("train", "val", "test")
            for directory in (data_dir / split).iterdir()
            if directory.is_dir()
        )
        label_to_idx = {name: index for index, name in enumerate(classes)}
        values = complete_input_statistics_for_splits(
            data_dir, label_to_idx, int(cfg.machine.num_workers), splits
        )
        return tuple(values[split][0] for split in splits)

def _spectral_loader(x, y, globals_, batch_size, shuffle):
    return DataLoader(
        TensorDataset(x.cpu(), y.cpu().view(-1), globals_.cpu()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_spectral_training(cfg: DictConfig):
    loaders = build_data(cfg, "spectral")
    train_loader, validation_loader, test_loader = loaders

    model_cfg, train_cfg = cfg.model, cfg.train
    model = ConfigurableSpectralModel(
        num_classes=int(train_cfg.out_channels),
        **OmegaConf.to_container(model_cfg, resolve=True),
    )
    model.set_class_counts(torch.bincount(train_loader.y, minlength=model.num_classes))
    if cfg.global_statistics.source == "points_2000":
        model.set_global_stats(train_loader.x)
        return model, loaders

    if cfg.global_statistics.source != "complete_input":
        raise ValueError(f"Unknown global statistics source: {cfg.global_statistics.source!r}")

    global_train = _spectral_global_features(cfg, (train_loader,), ("train",))[0]
    global_mean = global_train.mean(0)
    global_std = global_train.std(0).clamp_min(1e-6)
    model.global_mean.copy_(global_mean)
    model.global_std.copy_(global_std)
    global_validation, global_test = _spectral_global_features(
        cfg, (validation_loader, test_loader), ("val", "test")
    )
    return model, (
        _spectral_loader(train_loader.x, train_loader.y, global_train, cfg.machine.batch_size, True),
        _spectral_loader(
            validation_loader.x,
            validation_loader.y,
            global_validation,
            cfg.machine.batch_size,
            False,
        ),
        _spectral_loader(
            test_loader.x,
            test_loader.y,
            global_test,
            cfg.machine.batch_size,
            False,
        ),
    )


def build_clifford_training(cfg: DictConfig):
    train_loader, validation_loader, test_loader = build_data(cfg, "clifford")
    return build_clifford_model(cfg), (train_loader, validation_loader, test_loader)


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
