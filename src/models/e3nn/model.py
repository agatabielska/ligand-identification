import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from e3nn.nn import BatchNorm, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irreps, Linear, spherical_harmonics

from src.utils.metrics import compute_metrics

N_CHANNELS = 3  # density, occupancy, density^2


def natural_irreps(mul: int, l_from: int, l_to: int) -> Irreps:
    """mul copies of each l in [l_from, l_to] with parity (-1)^l (like spherical harmonics)."""
    return Irreps([(mul, (l, (-1) ** l)) for l in range(l_from, l_to + 1)])


class EquivariantBlock(nn.Module):
    """x + Linear(Gate(BN(TP(Lin(x), Lin(x))))): a quadratic (self tensor product) interaction."""

    def __init__(self, irreps_h: Irreps, n_scalars: int, mul_gated: int, lh: int):
        super().__init__()
        irreps_scalars = Irreps(f"{n_scalars}x0e")
        irreps_gated = natural_irreps(mul_gated, 1, lh)
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")
        self.gate = Gate(
            irreps_scalars, [F.silu], irreps_gates, [torch.sigmoid], irreps_gated
        )
        self.lin1 = Linear(irreps_h, irreps_h)
        self.tp = FullyConnectedTensorProduct(irreps_h, irreps_h, self.gate.irreps_in)
        self.bn = BatchNorm(self.gate.irreps_in)
        self.lin2 = Linear(self.gate.irreps_out, irreps_h)

    def forward(self, x):
        h = self.lin1(x)
        h = self.tp(h, h)
        h = self.gate(self.bn(h))
        return x + self.lin2(h)


class E3NNPointCloudModel(pl.LightningModule):
    """Rotation-invariant classifier on a spherical-harmonic expansion of the density.

    1. Density-weighted SH coefficients per radial shell and input channel
       (density, occupancy, density^2) -> irreps "M x l" for l = 0..lmax.
    2. Branch A: per-degree Gram matrices across shells (power spectrum) -> invariants.
    3. Branch B: learned e3nn network on those coefficients (Linear + self tensor
       product + Gate, residual blocks) -> scalars and norms of higher-l features.
    4. MLP head on concat(A, B).
    Input x: (B, N, 4) = [x, y, z, density]; padded points have density 0.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_norm_momentum: float = 0.1,
        n_shells: int = 16,
        lmax: int = 6,
        r_max: float = 1.0,
        hidden: int = 1024,
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
        warmup_frac: float = 0.03,
        full_train_metrics: bool = False,
        use_equivariant: bool = True,
        lh: int = 3,
        n_scalars: int = 128,
        mul_gated: int = 24,
        n_blocks: int = 2,
        point_dropout: float = 0.1,
        jitter: float = 0.0,
        chunk_size: int = 128,
        logit_adjust_tau: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lmax = lmax
        self.label_smoothing = label_smoothing
        self.warmup_frac = warmup_frac
        self.full_train_metrics = full_train_metrics
        self.use_equivariant = use_equivariant
        self.point_dropout = point_dropout
        self.jitter = jitter
        self.chunk_size = chunk_size
        self.logit_adjust_tau = logit_adjust_tau
        self.register_buffer("log_prior", torch.zeros(num_classes))

        self.register_buffer("centers", torch.linspace(0, r_max, n_shells))
        self.width = r_max / n_shells

        d_gram = n_shells * n_shells * (lmax + 1)
        d_in = d_gram

        if use_equivariant:
            m_in = n_shells * N_CHANNELS
            self.irreps_in = natural_irreps(m_in, 0, lmax)
            self.in_bn = BatchNorm(self.irreps_in, momentum=batch_norm_momentum)

            irreps_s = Irreps(f"{n_scalars}x0e")
            self.irreps_h = (irreps_s + natural_irreps(mul_gated, 1, lh)).simplify()
            self.embed = Linear(self.irreps_in, self.irreps_h)
            self.blocks = nn.ModuleList(
                [EquivariantBlock(self.irreps_h, n_scalars, mul_gated, lh) for _ in range(n_blocks)]
            )
            self.final_bn = BatchNorm(self.irreps_h, momentum=batch_norm_momentum)
            d_in += self.irreps_h.num_irreps  # one invariant per multiplicity

        self.head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, num_classes),
        )

        self.val_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )

    def set_class_counts(self, counts: torch.Tensor):
        """Store log class priors (used for logit-adjusted loss when logit_adjust_tau > 0)."""
        prior = (counts.float() + 1.0) / (counts.sum() + len(counts))
        self.log_prior.copy_(prior.log().to(self.log_prior.device))

    # ------------------------------------------------------------ featurizer
    @torch.no_grad()
    def coefficients(self, x: torch.Tensor):
        """Runs the featurizer in chunks so peak GPU memory does not scale with batch size."""
        outs = [self._coefficients_chunk(c) for c in x.split(self.chunk_size)]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def _coefficients_chunk(self, x: torch.Tensor):
        """Parameter-free: SH coefficients per shell/channel, shape (B, K*C, (lmax+1)^2),
        plus the density-channel coefficients (B, K, (lmax+1)^2) for the Gram branch."""
        pts, rho = x[..., :3], x[..., 3]
        valid = (rho > 0).to(rho.dtype)
        rho = rho * valid
        chans = torch.stack([rho, valid, rho * rho], dim=-1)  # (B, N, C)
        chans = chans / chans.sum(1, keepdim=True).clamp(min=1e-8)

        w = chans[..., 0]
        com = (pts * w[..., None]).sum(1, keepdim=True)
        d = pts - com
        r = d.norm(dim=-1)

        Y = spherical_harmonics(
            list(range(self.lmax + 1)), d, normalize=True, normalization="integral"
        )  # (B, N, M)
        g = torch.exp(-(((r[..., None] - self.centers) / self.width) ** 2))  # (B, N, K)

        gw = g[..., :, None] * chans[..., None, :]  # (B, N, K, C)
        gw = gw.reshape(*gw.shape[:2], -1)  # (B, N, K*C), index = k*C + c
        c_all = torch.einsum("bnj,bnm->bjm", gw, Y)  # (B, K*C, M)
        c_rho = c_all.reshape(c_all.shape[0], -1, N_CHANNELS, c_all.shape[-1])[:, :, 0]
        return c_all, c_rho

    def gram_features(self, c_rho: torch.Tensor) -> torch.Tensor:
        feats = []
        for l in range(self.lmax + 1):
            cl = c_rho[..., l**2 : (l + 1) ** 2]
            feats.append((cl @ cl.transpose(1, 2)).flatten(1))
        f = torch.cat(feats, dim=-1)
        return torch.sign(f) * torch.log1p(f.abs())

    def equivariant_features(self, c_all: torch.Tensor) -> torch.Tensor:
        # Flatten to e3nn layout: per degree l, a (mul, 2l+1) block, blocks concatenated.
        blocks = [
            c_all[..., l**2 : (l + 1) ** 2].reshape(c_all.shape[0], -1)
            for l in range(self.lmax + 1)
        ]
        h = torch.cat(blocks, dim=-1)
        h = self.in_bn(h)
        h = self.embed(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_bn(h)

        inv = []
        for (mul, ir), sl in zip(self.irreps_h, self.irreps_h.slices()):
            t = h[:, sl].reshape(h.shape[0], mul, ir.dim)
            if ir.l == 0:
                inv.append(t.reshape(h.shape[0], mul))
            else:
                inv.append((t.pow(2).mean(-1) + 1e-8).sqrt())
        return torch.cat(inv, dim=-1)

    def forward(self, x):
        # Whole featurizer + equivariant stage in fp32 (SH and tensor products are
        # numerically touchy in bf16). TF32 matmuls still apply.
        with torch.autocast(device_type=x.device.type, enabled=False):
            c_all, c_rho = self.coefficients(x.float())
            f = self.gram_features(c_rho)
            if self.use_equivariant:
                f = torch.cat([f, self.equivariant_features(c_all)], dim=-1)
            return self.head(f)

    # --------------------------------------------------------------- training
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.point_dropout > 0:
            keep = (torch.rand_like(x[..., 3]) > self.point_dropout).to(x.dtype)
            x = torch.cat([x[..., :3], (x[..., 3] * keep).unsqueeze(-1)], dim=-1)
        if self.jitter > 0:
            noise = torch.randn_like(x[..., :3]) * self.jitter
            x = torch.cat([x[..., :3] + noise, x[..., 3:]], dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(self._augment(x)).float()
        adj = logits + self.logit_adjust_tau * self.log_prior if self.logit_adjust_tau > 0 else logits
        loss = F.cross_entropy(adj, y, label_smoothing=self.label_smoothing)
        bs = x.size(0)

        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=bs)

        if self.full_train_metrics:
            m = compute_metrics(logits, y, self.num_classes)
            for k in ("top_10_acc", "brier_score", "mean_rank"):
                self.log(f"train_{k}", m[k], on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def _eval_step(self, batch, prefix: str, recall):
        x, y = batch
        y = y.view(-1)
        logits = self(x).float()
        m = compute_metrics(logits, y, self.num_classes)
        recall.update(logits.argmax(dim=1), y)

        bs = x.size(0)
        self.log(f"{prefix}_loss", m["loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}_acc", m["acc"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}_top_10_acc", m["top_10_acc"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}_brier_score", m["brier_score"], on_step=False, on_epoch=True, batch_size=bs)
        self.log(f"{prefix}_mean_rank", m["mean_rank"], on_step=False, on_epoch=True, batch_size=bs)
        return m["loss"]

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

    # ------------------------------------------------------------- optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total = int(self.trainer.estimated_stepping_batches)
        warm = max(1, int(total * self.warmup_frac))

        def lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            p = min(1.0, (step - warm) / max(1, total - warm))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }