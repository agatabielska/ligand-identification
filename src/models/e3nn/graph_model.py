import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import Irreps, Linear, TensorProduct, spherical_harmonics

from src.utils.metrics import compute_metrics


def natural_irreps(mul: int, l_from: int, l_to: int) -> Irreps:
    return Irreps([(mul, (l, (-1) ** l)) for l in range(l_from, l_to + 1)])


class MessagePassingLayer(nn.Module):
    """Channelwise (uvu) tensor-product convolution over a fixed-k neighbor list.

    h_i <- h_i + Gate( Linear( sum_j TP(h_j, Y(r_ij); w(|r_ij|)) / sqrt(k) ) + Linear(h_i) )
    """

    def __init__(self, irreps_in: Irreps, irreps_h: Irreps, irreps_sh: Irreps,
                 num_basis: int, mul: int, lmax: int, radial_hidden: int = 64):
        super().__init__()
        self.residual = irreps_in == irreps_h

        # Build uvu paths: every (input irrep) x (edge SH) -> allowed output irrep in irreps_h
        out_blocks, instr = [], []
        for i, (m_in, ir_in) in enumerate(irreps_in):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in irreps_h:
                        instr.append((i, j, len(out_blocks), "uvu", True))
                        out_blocks.append((m_in, ir_out))
        irreps_tp_out = Irreps(out_blocks)
        irreps_tp_out, perm, _ = irreps_tp_out.sort()
        instr = [(a, b, perm[c], mode, train) for a, b, c, mode, train in instr]

        self.tp = TensorProduct(
            irreps_in, irreps_sh, irreps_tp_out, instr,
            shared_weights=False, internal_weights=False,
        )
        self.radial = FullyConnectedNet([num_basis, radial_hidden, self.tp.weight_numel], F.silu)

        irreps_scalars = Irreps(f"{mul}x0e")
        irreps_gated = natural_irreps(mul, 1, lmax)
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")
        self.gate = Gate(irreps_scalars, [F.silu], irreps_gates, [torch.sigmoid], irreps_gated)
        self.lin_msg = Linear(self.tp.irreps_out, self.gate.irreps_in)
        self.lin_skip = Linear(irreps_in, self.gate.irreps_in)

    def forward(self, h, src_idx, sh, basis, edge_mask, k: int):
        # h: (Bn, D); src_idx: (Bn*k,) global sender index; sh: (Bn*k, dim_sh); basis: (Bn*k, nb)
        w = self.radial(basis)
        msg = self.tp(h[src_idx], sh, w)  # (Bn*k, D_tp)
        msg = (msg * edge_mask[:, None]).view(-1, k, msg.shape[-1]).sum(1) / math.sqrt(k)
        out = self.gate(self.lin_msg(msg) + self.lin_skip(h))
        return h + out if self.residual else out


class E3NNGraphModel(pl.LightningModule):
    """Small SE(3)-equivariant message-passing baseline on the top-density points.

    Input x: (B, N, 4) = [x, y, z, density], sorted by descending density (padding at the end,
    density 0). The first n_points rows are used. Invariant readout (scalars + norms of l>0
    features) -> masked mean/max pooling -> MLP.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_norm_momentum: float = 0.1,  # unused, config compatibility
        n_points: int = 256,
        k: int = 12,
        mul: int = 32,
        lmax: int = 2,
        n_layers: int = 3,
        num_basis: int = 8,
        r_max: float = 1.0,
        hidden: int = 512,
        dropout: float = 0.2,
        label_smoothing: float = 0.0,
        warmup_frac: float = 0.03,
        point_dropout: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.warmup_frac = warmup_frac
        self.point_dropout = point_dropout
        self.n_points, self.k, self.lmax = n_points, k, lmax
        self.num_basis, self.r_max = num_basis, r_max

        self.irreps_sh = Irreps.spherical_harmonics(lmax)
        self.irreps_h = (Irreps(f"{mul}x0e") + natural_irreps(mul, 1, lmax)).simplify()

        self.embed = nn.Sequential(nn.Linear(2, mul), nn.SiLU(), nn.Linear(mul, mul))  # -> mul x 0e
        irreps_first = Irreps(f"{mul}x0e")
        layers = []
        for i in range(n_layers):
            layers.append(
                MessagePassingLayer(
                    irreps_first if i == 0 else self.irreps_h,
                    self.irreps_h, self.irreps_sh, num_basis, mul, lmax,
                )
            )
        self.layers = nn.ModuleList(layers)

        d_inv = self.irreps_h.num_irreps
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_inv),
            nn.Linear(2 * d_inv, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        self.val_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", task="multiclass", zero_division=0
        )

    # ------------------------------------------------------------------ graph
    @torch.no_grad()
    def build_graph(self, x):
        """kNN graph among valid points; returns everything the layers need (no autograd)."""
        n, k = self.n_points, self.k
        pts, rho = x[:, :n, :3], x[:, :n, 3]
        B = pts.shape[0]
        valid = rho > 0
        w = rho * valid
        com = (pts * w[..., None]).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-8)[..., None]
        pts = pts - com

        dist = torch.cdist(pts, pts)  # (B, n, n)
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(valid[:, None, :], dist, inf)  # invalid senders excluded
        eye = torch.eye(n, dtype=torch.bool, device=x.device)[None]
        dist = torch.where(eye, inf, dist)  # no self loops
        d_k, idx = dist.topk(k, dim=-1, largest=False)  # (B, n, k)
        edge_ok = torch.isfinite(d_k) & valid[:, :, None]  # receiver must be valid too

        offs = (torch.arange(B, device=x.device) * n)[:, None, None]
        src_idx = (idx + offs).reshape(-1)  # global sender index
        send_pts = pts.reshape(-1, 3)[src_idx].view(B, n, k, 3)
        vec = (send_pts - pts[:, :, None, :]).reshape(-1, 3)
        d = torch.where(edge_ok, d_k, torch.zeros_like(d_k)).reshape(-1)
        mask = edge_ok.reshape(-1).to(x.dtype)

        sh = spherical_harmonics(self.irreps_sh, vec, normalize=True, normalization="component")
        basis = soft_one_hot_linspace(
            d, 0.0, self.r_max, self.num_basis, basis="smooth_finite", cutoff=True
        ) * (self.num_basis ** 0.5)

        rho_n = rho / rho.max(1, keepdim=True).values.clamp(min=1e-8)
        r_com = pts.norm(dim=-1) / self.r_max
        node_in = torch.stack([rho_n, r_com], dim=-1).reshape(B * n, 2)
        node_mask = valid.reshape(B * n, 1).to(x.dtype)
        return node_in, node_mask, src_idx, sh, basis, mask

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B = x.shape[0]
            node_in, node_mask, src_idx, sh, basis, mask = self.build_graph(x)
            h = self.embed(node_in) * node_mask
            for layer in self.layers:
                h = layer(h, src_idx, sh, basis, mask, self.k) * node_mask

            inv = []
            for (mul, ir), sl in zip(self.irreps_h, self.irreps_h.slices()):
                t = h[:, sl].reshape(h.shape[0], mul, ir.dim)
                if ir.l == 0:
                    inv.append(t.reshape(h.shape[0], mul))
                else:
                    inv.append((t.pow(2).mean(-1) + 1e-8).sqrt())
            inv = torch.cat(inv, dim=-1).view(B, self.n_points, -1)
            m = node_mask.view(B, self.n_points, 1)
            mean = (inv * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = inv.masked_fill(m == 0, float("-inf")).max(1).values
            mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
            return self.head(torch.cat([mean, mx], dim=-1))

    # --------------------------------------------------------------- training
    def _augment(self, x):
        if self.point_dropout > 0:
            keep = (torch.rand_like(x[..., 3]) > self.point_dropout).to(x.dtype)
            x = torch.cat([x[..., :3], (x[..., 3] * keep).unsqueeze(-1)], dim=-1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(self._augment(x)).float()
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        bs = x.size(0)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train_acc", acc, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def _eval_step(self, batch, prefix, recall):
        x, y = batch
        y = y.view(-1)
        logits = self(x).float()
        m = compute_metrics(logits, y, self.num_classes)
        recall.update(logits.argmax(dim=1), y)
        bs = x.size(0)
        for key in ("loss", "acc", "top_10_acc", "brier_score", "mean_rank"):
            self.log(f"{prefix}_{key}", m[key], on_step=False, on_epoch=True, batch_size=bs)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total = int(self.trainer.estimated_stepping_batches)
        warm = max(1, int(total * self.warmup_frac))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            p = min(1.0, (step - warm) / max(1, total - warm))
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}