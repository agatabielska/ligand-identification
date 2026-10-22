"""The single configurable Lightning model for the retained spectral protocols."""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.models.e3nn.features import sh_coefficients
from src.models.e3nn.spectral_net import SpectralNet
from src.utils.metrics import compute_metrics


GLOBAL_COLUMNS = {
    "none": (),
    "all": tuple(range(12)),
    "geom": tuple(range(3, 12)),
    "count": (0, 1, 2),
}


class ConfigurableSpectralModel(pl.LightningModule):
    """Raw point-cloud Lightning model with all spectral choices in its config."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 3e-3,
        weight_decay: float = 5e-2,
        n_shells: int = 4,
        lmax: int = 6,
        r_max: float = 1.0,
        n_channels: int = 2,
        n_mix: int = 32,
        pow_mode: str = "triu",
        gram: str = "none",
        triples: str = "nonzero",
        bispec_lmax: int = 6,
        bispec_mode: str = "batched",
        norm: str = "batch",
        in_bn: bool = False,
        mix_bn: bool = True,
        use_globals: str = "all",
        glob_dim: int = 64,
        glob_noise: float = 0.0,
        widths: tuple[int, ...] = (512, 256),
        head_dropout: float = 0.2,
        feat_dropout: float = 0.0,
        point_dropout: float = 0.1,
        density_noise: float = 0.0,
        clean_probability: float = 0.125,
        label_smoothing: float = 0.05,
        warmup_frac: float = 0.03,
        chunk_size: int = 128,
        **_,
    ):
        super().__init__()
        if use_globals not in GLOBAL_COLUMNS:
            raise ValueError(f"use_globals must be one of {tuple(GLOBAL_COLUMNS)}, got {use_globals!r}")
        if not widths:
            raise ValueError("widths must contain at least one hidden-layer width")

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.point_dropout = point_dropout
        self.density_noise = density_noise
        self.glob_noise = glob_noise
        self.clean_probability = clean_probability
        self.label_smoothing = label_smoothing
        self.warmup_frac = warmup_frac
        self.chunk_size = chunk_size
        self.use_globals = use_globals
        self.global_columns = GLOBAL_COLUMNS[use_globals]
        self.glob_dim = glob_dim
        self.register_buffer("log_prior", torch.zeros(num_classes))
        self.register_buffer("global_mean", torch.zeros(12))
        self.register_buffer("global_std", torch.ones(12))

        self.net = SpectralNet(
            num_classes=num_classes,
            n_shells=n_shells,
            lmax=lmax,
            r_max=r_max,
            n_channels=n_channels,
            n_mix=n_mix,
            bispec_lmax=bispec_lmax,
            triples=triples,
            pow_mode=pow_mode,
            gram=gram,
            norm=norm,
            in_bn=in_bn,
            mix_bn=mix_bn,
            hidden=widths[0],
            n_hidden=0,
            feat_dropout=feat_dropout,
            bispec_mode=bispec_mode,
        )

        d_in = self.net.d_in
        if self.global_columns:
            self.global_branch = nn.Sequential(
                nn.BatchNorm1d(len(self.global_columns)),
                nn.Linear(len(self.global_columns), glob_dim),
                nn.GELU(),
            )
            d_in += glob_dim
        else:
            self.global_branch = None

        layers: list[nn.Module] = [
            nn.LayerNorm(d_in) if norm == "layer" else nn.BatchNorm1d(d_in)
        ]
        if feat_dropout > 0:
            layers.append(nn.Dropout(feat_dropout))
        previous = d_in
        for index, width in enumerate(widths):
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            if index < len(widths) - 1:
                layers.append(nn.Dropout(head_dropout))
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.head = nn.Sequential(*layers)
        self.val_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )

    def set_class_counts(self, counts: torch.Tensor):
        prior = (counts.float() + 1.0) / (counts.sum() + len(counts))
        self.log_prior.copy_(prior.log().to(self.log_prior.device))

    @torch.no_grad()
    def set_global_stats(self, x: torch.Tensor):
        """Standardize global features using the 2000-point training split."""
        features = torch.cat(
            [self.global_features(batch) for batch in x.split(1024)],
            dim=0,
        )
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite global features found in the 2000-point training tensors")
        self.global_mean.copy_(features.mean(0).to(self.global_mean.device))
        self.global_std.copy_(features.std(0).clamp_min(1e-6).to(self.global_std.device))

    @torch.no_grad()
    def global_features(self, x: torch.Tensor) -> torch.Tensor:
        pts, rho = x[..., :3], x[..., 3]
        valid = (rho > 0).to(x.dtype)
        rho = rho * valid
        n, total = valid.sum(1), rho.sum(1)
        weights = rho / total.clamp_min(1e-8)[:, None]
        center = (pts * weights[..., None]).sum(1, keepdim=True)
        d = pts - center
        radius = d.norm(dim=-1)
        radius_valid = radius.masked_fill(~valid.bool(), 0)
        radius_g = (weights * radius.square()).sum(1).clamp_min(0).sqrt()
        radius_mean = (weights * radius).sum(1)
        covariance = torch.einsum("bn,bni,bnj->bij", weights, d, d)
        covariance = covariance + 1e-8 * torch.eye(3, device=x.device, dtype=x.dtype)
        eig = torch.linalg.eigvalsh(covariance).clamp_min(0).sqrt()
        valid_points = valid.bool().unsqueeze(-1)
        mins = pts.masked_fill(~valid_points, 0).amin(dim=1)
        maxs = pts.masked_fill(~valid_points, 0).amax(dim=1)
        bbox = (maxs - mins).sort(dim=1).values
        return torch.stack(
            [
                torch.log1p(n),
                torch.log1p(total),
                torch.log1p(rho.max(1).values),
                radius_valid.max(1).values,
                radius_g,
                radius_mean,
                eig[:, 0],
                eig[:, 1],
                eig[:, 2],
                bbox[:, 0],
                bbox[:, 1],
                bbox[:, 2],
            ],
            dim=-1,
        )

    def forward(
        self,
        x: torch.Tensor,
        global_x: torch.Tensor | None = None,
        global_values: torch.Tensor | None = None,
    ):
        with torch.autocast(device_type=x.device.type, enabled=False):
            coefficients = sh_coefficients(
                x.float(),
                lmax=self.net.lmax,
                n_shells=self.net.K,
                r_max=self.net.r_max,
                n_channels=self.net.C,
                chunk=self.chunk_size,
            )
            features = self.net.features(coefficients)
            if self.global_branch is not None:
                if global_values is None:
                    source = x if global_x is None else global_x
                    global_values = self.global_features(source)
                    global_values = (global_values - self.global_mean) / self.global_std
                    global_values = global_values[:, self.global_columns]
                globals_ = self.global_branch(global_values)
                features = torch.cat([features, globals_], dim=-1)
        return self.head(features)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.point_dropout <= 0 and self.density_noise <= 0:
            return x
        original = x
        if self.point_dropout > 0:
            keep = (torch.rand_like(x[..., 3]) > self.point_dropout).to(x.dtype)
            x = torch.cat([x[..., :3], (x[..., 3] * keep).unsqueeze(-1)], dim=-1)
        if self.density_noise > 0:
            rho = x[..., 3] * torch.exp(torch.randn_like(x[..., 3]) * self.density_noise)
            x = torch.cat([x[..., :3], rho.unsqueeze(-1)], dim=-1)
        clean = torch.rand(x.shape[0], device=x.device) < self.clean_probability
        return torch.where(clean[:, None, None], original, x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        augmented = self._augment(x)
        global_values = None
        if self.global_branch is not None:
            global_values = self.global_features(x)
            global_values = (global_values - self.global_mean) / self.global_std
            global_values = global_values[:, self.global_columns]
            if self.glob_noise > 0:
                global_values = global_values + self.glob_noise * torch.randn_like(global_values)
        logits = self(augmented, global_values=global_values).float()
        loss = torch.nn.functional.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train_acc", (logits.argmax(1) == y).float().mean(), on_step=False, on_epoch=True,
                 batch_size=x.size(0))
        return loss

    def _eval_step(self, batch, prefix: str, recall):
        x, y = batch
        y = y.view(-1)
        logits = self(x).float()
        metrics = compute_metrics(logits, y, self.num_classes)
        recall.update(logits.argmax(1), y)
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=False, on_epoch=True, batch_size=x.size(0))
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val", self.val_recall)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test", self.test_recall)

    def on_validation_epoch_end(self):
        self.log("val_macro_recall", self.val_recall.compute())
        self.val_recall.reset()

    def on_test_epoch_end(self):
        self.log("test_macro_recall", self.test_recall.compute())
        self.test_recall.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total = int(self.trainer.estimated_stepping_batches)
        warm = max(1, int(total * self.warmup_frac))

        def schedule(step):
            if step < warm:
                return (step + 1) / warm
            progress = min(1.0, (step - warm) / max(1, total - warm))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }
