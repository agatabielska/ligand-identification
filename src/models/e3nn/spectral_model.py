import torch
import torch.nn as nn
import torchmetrics
from e3nn.nn import BatchNorm
from e3nn.o3 import Linear, wigner_3j

from src.models.e3nn.model import E3NNPointCloudModel, N_CHANNELS, natural_irreps
from src.utils.metrics import compute_metrics

# Post-hoc logit adjustment strengths evaluated at val/test time (no retraining needed).
EVAL_TAUS = (0.25, 0.5, 0.75, 1.0)


def _slog(f: torch.Tensor) -> torch.Tensor:
    return torch.sign(f) * torch.log1p(f.abs())


def _tk(t: float) -> str:
    return f"t{int(round(t * 100)):03d}"


class SpectralModel(E3NNPointCloudModel):
    """v1 (Gram + power spectrum + bispectrum -> MLP). With all flags at their defaults this
    is the same model as v1. Every improvement is opt-in so its effect can be measured alone:

      use_globals    add global rotation-invariant scalars (point count, total/max density,
                     radius of gyration, inertia eigenvalues) that the featurizer normalises away
      gram_all       Gram across shells for all 3 input channels (upper triangle) instead of
                     density only
      feat_dropout   dropout on the feature vector before the MLP head
      density_noise  multiplicative log-normal noise on densities during training

    Training always uses plain cross-entropy (logit_adjust_tau is left at 0). Instead, validation
    and test additionally report accuracy / macro recall after post-hoc logit adjustment
    (logits - tau * log_prior) for tau in EVAL_TAUS, so you can read the accuracy vs
    macro-recall trade-off off one run.
    """

    def __init__(
        self,
        num_classes: int,
        n_mix: int = 16,
        bispec_lmax: int = 6,
        use_globals: bool = False,
        gram_all: bool = False,
        feat_dropout: float = 0.0,
        density_noise: float = 0.0,
        **kw,
    ):
        kw["use_equivariant"] = False
        super().__init__(num_classes, **kw)
        lmax = self.lmax
        K = self.centers.numel()
        self.K = K
        self.n_mix, self.bl = n_mix, min(bispec_lmax, lmax)
        self.use_globals = use_globals
        self.use_gram_all = gram_all
        self.density_noise = density_noise

        irreps_in = natural_irreps(K * N_CHANNELS, 0, lmax)
        irreps_mix = natural_irreps(n_mix, 0, lmax)
        self.in_bn = BatchNorm(irreps_in)
        self.mix = Linear(irreps_in, irreps_mix)
        self.mix_bn = BatchNorm(irreps_mix)

        self.triples = [
            (a, b, c)
            for a in range(self.bl + 1)
            for b in range(a, self.bl + 1)
            for c in range(b, min(a + b, self.bl) + 1)
        ]
        for i, (a, b, c) in enumerate(self.triples):
            self.register_buffer(f"w3j_{i}", wigner_3j(a, b, c).float())

        self.register_buffer("iu_mix", torch.triu_indices(n_mix, n_mix))
        self.register_buffer("iu_k", torch.triu_indices(K, K))

        if gram_all:
            d_gram = N_CHANNELS * (lmax + 1) * self.iu_k.shape[1]
        else:
            d_gram = K * K * (lmax + 1)
        d_pow = (lmax + 1) * self.iu_mix.shape[1]
        d_bis = len(self.triples) * n_mix
        d_in = d_gram + d_pow + d_bis

        if use_globals:
            n_glob = 9
            self.glob = nn.Sequential(
                nn.BatchNorm1d(n_glob), nn.Linear(n_glob, 32), nn.GELU()
            )
            d_in += 32

        hidden, p = kw.get("hidden", 1024), kw.get("dropout", 0.2)
        layers = [nn.LayerNorm(d_in)]
        if feat_dropout > 0:
            layers.append(nn.Dropout(feat_dropout))
        layers += [
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, num_classes),
        ]
        self.head = nn.Sequential(*layers)

        # One macro-recall metric per (split, tau) for the post-hoc adjustment sweep.
        self.tau_recall = nn.ModuleDict(
            {
                f"{pfx}_{_tk(t)}": torchmetrics.Recall(
                    num_classes=num_classes,
                    average="macro",
                    task="multiclass",
                    zero_division=0,
                )
                for pfx in ("val", "test")
                for t in EVAL_TAUS
            }
        )

    # ------------------------------------------------------------------ features
    def gram_all_channels(self, c_all: torch.Tensor) -> torch.Tensor:
        """Shell x shell Gram per degree and per input channel. c_all: (B, K*C, M), idx = k*C + c."""
        B = c_all.shape[0]
        c = c_all.reshape(B, self.K, N_CHANNELS, -1)  # (B, K, C, M)
        feats = []
        for l in range(self.lmax + 1):
            cl = c[..., l**2 : (l + 1) ** 2].permute(0, 2, 1, 3)  # (B, C, K, 2l+1)
            g = cl @ cl.transpose(-1, -2)  # (B, C, K, K)
            feats.append(g[..., self.iu_k[0], self.iu_k[1]].flatten(1))
        return _slog(torch.cat(feats, dim=-1))

    def spectral_features(self, c_all: torch.Tensor) -> torch.Tensor:
        B, lmax = c_all.shape[0], self.lmax
        h = torch.cat(
            [c_all[..., l**2 : (l + 1) ** 2].reshape(B, -1) for l in range(lmax + 1)], -1
        )
        h = self.mix_bn(self.mix(self.in_bn(h)))

        ts, off = [], 0
        for l in range(lmax + 1):
            n = self.n_mix * (2 * l + 1)
            ts.append(h[:, off : off + n].reshape(B, self.n_mix, 2 * l + 1))
            off += n

        feats = []
        for t in ts:  # cross-channel power spectrum per degree
            g = t @ t.transpose(1, 2)
            feats.append(g[:, self.iu_mix[0], self.iu_mix[1]])
        for i, (a, b, c) in enumerate(self.triples):  # channelwise bispectrum
            w = getattr(self, f"w3j_{i}")
            feats.append(torch.einsum("mnp,bum,bun,bup->bu", w, ts[a], ts[b], ts[c]))
        return _slog(torch.cat(feats, -1))

    @torch.no_grad()
    def global_features(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation-invariant scalars that the featurizer's normalisation removes: (B, 9)."""
        pts, rho = x[..., :3], x[..., 3]
        valid = (rho > 0).to(x.dtype)
        rho = rho * valid
        n = valid.sum(1)
        s = rho.sum(1)
        w = rho / s.clamp(min=1e-8)[:, None]
        com = (pts * w[..., None]).sum(1, keepdim=True)
        d = pts - com
        r = d.norm(dim=-1)
        rg = (w * r * r).sum(1).clamp(min=0).sqrt()
        rmean = (w * r).sum(1)
        rmax = rho.max(1).values
        cov = torch.einsum("bn,bni,bnj->bij", w, d, d)
        cov = cov + 1e-8 * torch.eye(3, device=x.device, dtype=x.dtype)
        eig = torch.linalg.eigvalsh(cov).clamp(min=0).sqrt()  # (B, 3), ascending
        return torch.stack(
            [
                torch.log1p(n),
                torch.log1p(s),
                torch.log1p(rmax),
                torch.log1p(s / n.clamp(min=1.0)),
                rg,
                rmean,
                eig[:, 0],
                eig[:, 1],
                eig[:, 2],
            ],
            dim=-1,
        )

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            c_all, c_rho = self.coefficients(x)
            if self.use_gram_all:
                parts = [self.gram_all_channels(c_all)]
            else:
                parts = [self.gram_features(c_rho)]
            parts.append(self.spectral_features(c_all))
            if self.use_globals:
                parts.append(self.glob(self.global_features(x)))
            return self.head(torch.cat(parts, dim=-1))

    # ------------------------------------------------------------- augmentation
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = super()._augment(x)  # point dropout + jitter, same as v1
        if self.density_noise > 0:
            rho = x[..., 3] * torch.exp(torch.randn_like(x[..., 3]) * self.density_noise)
            x = torch.cat([x[..., :3], rho.unsqueeze(-1)], dim=-1)  # padded points stay 0
        return x

    # --------------------------------------------------------------- evaluation
    def _eval_step(self, batch, prefix: str, recall):
        x, y = batch
        y = y.view(-1)
        logits = self(x).float()
        m = compute_metrics(logits, y, self.num_classes)
        recall.update(logits.argmax(dim=1), y)

        bs = x.size(0)
        for key in ("loss", "acc", "top_10_acc", "brier_score", "mean_rank"):
            self.log(f"{prefix}_{key}", m[key], on_step=False, on_epoch=True, batch_size=bs)

        # Post-hoc logit adjustment sweep: argmax(logits - tau * log_prior)
        for t in EVAL_TAUS:
            k = _tk(t)
            pred = (logits - t * self.log_prior).argmax(dim=1)
            self.tau_recall[f"{prefix}_{k}"].update(pred, y)
            self.log(
                f"{prefix}_acc_{k}",
                (pred == y).float().mean(),
                on_step=False,
                on_epoch=True,
                batch_size=bs,
            )
        return m["loss"]

    def _log_tau_recalls(self, prefix: str):
        for t in EVAL_TAUS:
            k = _tk(t)
            r = self.tau_recall[f"{prefix}_{k}"]
            self.log(f"{prefix}_macro_recall_{k}", r.compute())
            r.reset()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._log_tau_recalls("val")

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        self._log_tau_recalls("test")