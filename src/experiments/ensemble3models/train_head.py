"""Train a classifier on frozen, then fine-tuned, spectral e3nn backbones."""

from pathlib import Path
from datetime import datetime, timezone

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.experiments.train_3_models import build_config
from src.experiments.ensemble3models.losses import build_loss
from src.models.e3nn.ensemble import best_ckpt, load_model
from src.models.e3nn.configurable_spectral import ConfigurableSpectralModel
from src.pipeline.data_loader import build_data, validate_data_dir
from src.utils.set_seed import set_seed


ROOT = Path(__file__).resolve().parents[3]
set_seed(42)

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch, **kw):
        super().__init__(**kw)
        self.start_epoch = start_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.start_epoch:
            self.wait_count = 0
            self.stopped_epoch = 0
            initial_score = torch.inf if self.mode == "min" else -torch.inf
            self.best_score = torch.tensor(initial_score, device=pl_module.device)

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def _resolve_backbone_dirs(experiment_cfg: DictConfig) -> list[Path]:
    """Resolve one complete ensemble run, preferring the newest run by default."""
    checkpoint_root = ROOT / "checkpoints"
    run_id = experiment_cfg.get("run_id")
    entries = list(experiment_cfg.models)
    if run_id:
        return [
            checkpoint_root / f"{run_id}-{entry.name}"
            if "checkpoint" not in entry
            else Path(entry.checkpoint)
            for entry in entries
        ]

    if any("checkpoint" in entry for entry in entries):
        return [
            Path(entry.checkpoint) if "checkpoint" in entry else checkpoint_root / entry.name
            for entry in entries
        ]

    candidates = []
    for model_dir in checkpoint_root.glob(f"*-{entries[0].name}"):
        prefix = model_dir.name[: -len(entries[0].name) - 1]
        dirs = [checkpoint_root / f"{prefix}-{entry.name}" for entry in entries]
        if all(directory.is_dir() and any(directory.glob("model-*.ckpt")) for directory in dirs):
            candidates.append((max(directory.stat().st_mtime for directory in dirs), dirs))
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    return [checkpoint_root / entry.name for entry in entries]


class SynchronizedEnsembleLoader:
    """Batch aligned samples from one spectral loader per backbone source."""

    def __init__(self, loaders, shuffle: bool):
        self.loaders = loaders
        self.shuffle = shuffle
        sizes = [len(loader.y) for loader in loaders]
        if len(set(sizes)) != 1:
            raise ValueError(f"Ensemble data sources have different sample counts: {sizes}")
        labels = loaders[0].y
        if any(not torch.equal(labels, loader.y) for loader in loaders[1:]):
            raise ValueError("Ensemble data sources must have identical label ordering.")
        self.y = labels
        self.batch_size = loaders[0].batch_size
        self.device = loaders[0].device

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        index_device = self.device if self.loaders[0].on_gpu else "cpu"
        indices = (
            torch.randperm(len(self.y), device=index_device)
            if self.shuffle
            else torch.arange(len(self.y), device=index_device)
        )
        for start in range(0, len(self.y), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            sources = []
            for loader in self.loaders:
                x = loader.x[batch_indices]
                g = loader.g[batch_indices] if loader.g is not None else None
                if not loader.on_gpu:
                    x = x.to(loader.device, non_blocking=True)
                    g = g.to(loader.device, non_blocking=True) if g is not None else None
                sources.append((x, g))
            yield tuple(sources), self.y[batch_indices].to(self.device)


class StackedHead(pl.LightningModule):
    def __init__(
        self, backbones, num_classes, learning_rate, weight_decay, unfreeze_epoch,
        loss_cfg=None, class_counts=None, feature_noise=0.2, backbone_lr_scale=0.1
    ):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.unfreeze_epoch = unfreeze_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_cfg = loss_cfg
        self.feature_noise = feature_noise
        counts = torch.as_tensor(class_counts, dtype=torch.long)
        prior = (counts.float() + 1.0) / (counts.sum() + len(counts))
        self.register_buffer("log_prior", prior.log())
        self.loss_fn = build_loss(loss_cfg, counts, self.log_prior)
        feature_size = sum(
            model.net.d_in + (model.glob_dim if model.global_columns else 0)
            for model in backbones
        )
        self.backbone_lr_scale = backbone_lr_scale
        # replaces self.classifier
        self.norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(0.2)
        self.correction = nn.Linear(feature_size, num_classes)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self._set_backbones_trainable(False)

    def _set_backbones_trainable(self, trainable):
        for backbone in self.backbones:
            for p in backbone.parameters():
                p.requires_grad = trainable
            backbone.eval()

    def on_train_epoch_start(self):
        self._set_backbones_trainable(self.current_epoch >= self.unfreeze_epoch)

    def forward(self, batch):
        sources, _ = batch
        frozen = self.current_epoch < self.unfreeze_epoch and self.training
        feats, base_logits = [], []
        for model, (x, g) in zip(self.backbones, sources):
            f = model.encode_features(x, self._prepare_globals(model, g))
            if self.current_epoch < self.unfreeze_epoch:
                f = f.detach()

            if self.training:
                f = f + self.feature_noise * f.std(dim=0, keepdim=True) * torch.randn_like(f)
            feats.append(f)
            base_logits.append(model.head(f))
        base = torch.stack(base_logits).mean(0)
        z = self.dropout(self.norm(torch.cat(feats, -1)))
        return base + self.correction(z)

    def _step(self, batch, prefix):
        _, y = batch
        logits = self(batch)
        loss = self.loss_fn(logits, y.view(-1))
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, batch_size=y.size(0))
        self.log(f"{prefix}_acc", (logits.argmax(1) == y.view(-1)).float().mean(),
                 on_step=False, on_epoch=True, batch_size=y.size(0))
        return loss

    @staticmethod
    def _prepare_globals(model, global_values):
        if model.global_branch is None:
            return None
        global_values = (global_values - model.global_mean) / model.global_std
        return global_values[:, model.global_columns]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        head_params = list(self.norm.parameters()) + list(self.correction.parameters())
        optimizer = torch.optim.AdamW([
            {"params": head_params, "lr": self.learning_rate},
            {"params": self.backbones.parameters(),
             "lr": self.learning_rate * self.backbone_lr_scale},
        ], weight_decay=self.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2),
            "monitor": "val_loss"}}


def run_training(experiment_cfg: DictConfig):
    ensemble_cfg = experiment_cfg.ensemble
    run_id = experiment_cfg.get("run_id") or datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    if len(experiment_cfg.models) != 3:
        raise ValueError("Expected exactly three backbone models.")

    model_cfgs = [build_config(model_cfg) for model_cfg in experiment_cfg.models]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones = []
    run_dirs = _resolve_backbone_dirs(experiment_cfg)
    print(f"Loading backbone checkpoints from: {', '.join(map(str, run_dirs))}")
    for model_cfg, run_dir in zip(model_cfgs, run_dirs):
        if not run_dir.is_absolute():
            run_dir = ROOT / model_cfg.paths.model_checkpoint / run_dir
        backbones.append(load_model(best_ckpt(run_dir), device, ConfigurableSpectralModel))

    data_path = ensemble_cfg.get("data_path")
    data_cfgs = []
    for model_cfg, model_entry in zip(model_cfgs, experiment_cfg.models):
        source_path = model_entry.get("data_path", data_path or model_cfg.paths.data_dir)
        model_cfg.paths.data_dir = source_path
        validate_data_dir(Path(source_path))
        data_cfgs.append(model_cfg)
    split_loaders = [build_data(cfg, "spectral") for cfg in data_cfgs]
    train_loader = SynchronizedEnsembleLoader(
        [loaders[0] for loaders in split_loaders], shuffle=True
    )
    val_loader = SynchronizedEnsembleLoader(
        [loaders[1] for loaders in split_loaders], shuffle=False
    )
    test_loader = SynchronizedEnsembleLoader(
        [loaders[2] for loaders in split_loaders], shuffle=False
    )
    num_classes = int(torch.cat((train_loader.y, val_loader.y, test_loader.y)).max().item() + 1)
    class_counts = torch.bincount(train_loader.y, minlength=num_classes)
    model = StackedHead(backbones, num_classes, ensemble_cfg.learning_rate,
                        ensemble_cfg.weight_decay, ensemble_cfg.unfreeze_epoch,
                        loss_cfg=ensemble_cfg.get("loss"), class_counts=class_counts,
                        feature_noise=float(ensemble_cfg.get("feature_noise", 0.2)),
                        backbone_lr_scale=float(
                            ensemble_cfg.get("backbone_lr_scale", 0.1)
                        ))
    loss_cfg = ensemble_cfg.get("loss")
    loss_name = str(loss_cfg.get("name", "baseline") if loss_cfg is not None else "baseline")
    loss_tags = ["loss-experiment", f"loss:{loss_name}"]
    if loss_name == "baseline":
        epsilon = float(loss_cfg.get("label_smoothing", 0.05) if loss_cfg is not None else 0.05)
        loss_tags.append(f"label-smoothing:{epsilon}")
    elif loss_name == "logit_adjusted_ce":
        loss_tags.append(f"tau:{float(loss_cfg.get('tau', 1.0))}")
    elif loss_name == "focal":
        loss_tags.append(f"gamma:{float(loss_cfg.get('gamma', 2.0))}")
    logger = WandbLogger(project=experiment_cfg.wandb.project,
                         name=ensemble_cfg.name,
                         id=str(run_id),
                         resume="never",
                         tags=[
                             experiment_cfg.wandb.experiment,
                             "stacked-head",
                             "three-models",
                             *loss_tags,
                         ],
                         config=OmegaConf.to_container(experiment_cfg, resolve=True))
    wandb_run = logger.experiment
    trainer = pl.Trainer(
        max_epochs=ensemble_cfg.epochs,
        accelerator="auto",
        devices=data_cfgs[0].machine.devices,
        precision=data_cfgs[0].train.get("precision", "32-true"),
        gradient_clip_val=data_cfgs[0].train.gradient_clip_val,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3,
                            dirpath=str(
                                Path(data_cfgs[0].paths.model_checkpoint) / str(run_id)
                            )),
            DelayedEarlyStopping(start_epoch=ensemble_cfg.unfreeze_epoch, monitor="val_loss", mode="min", patience=ensemble_cfg.patience),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
    )
    try:
        trainer.fit(model, train_loader, val_loader)
        test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
        if test_results:
            logger.log_metrics(test_results[0], step=trainer.current_epoch)
        return test_results
    finally:
        wandb_run.finish()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(experiment_cfg: DictConfig):
    run_training(experiment_cfg)


if __name__ == "__main__":
    main()
