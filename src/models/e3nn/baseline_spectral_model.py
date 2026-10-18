import math
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Linear, wigner_3j

from src.models.e3nn.model import E3NNPointCloudModel, N_CHANNELS, natural_irreps


class BaselineSpectralModel(E3NNPointCloudModel):
    """Learned per-degree channel mix -> power spectrum + bispectrum invariants -> MLP.

    Reuses the parameter-free SH featurizer of E3NNPointCloudModel.
    """

    def __init__(self, num_classes, n_mix: int = 16, bispec_lmax: int = 6, **kw):
        kw["use_equivariant"] = False
        super().__init__(num_classes, **kw)
        lmax, K = self.lmax, len(self.centers)
        self.n_mix, self.bl = n_mix, min(bispec_lmax, lmax)

        irreps_in = natural_irreps(K * N_CHANNELS, 0, lmax)
        irreps_mix = natural_irreps(n_mix, 0, lmax)
        self.in_bn = BatchNorm(irreps_in)
        self.mix = Linear(irreps_in, irreps_mix)
        self.mix_bn = BatchNorm(irreps_mix)

        # l1 <= l2 <= l3, triangle rule; both parities (odd sums are chirality-sensitive)
        self.triples = [
            (a, b, c)
            for a in range(self.bl + 1)
            for b in range(a, self.bl + 1)
            for c in range(b, min(a + b, self.bl) + 1)
        ]
        for i, (a, b, c) in enumerate(self.triples):
            self.register_buffer(f"w3j_{i}", wigner_3j(a, b, c).float())

        iu = torch.triu_indices(n_mix, n_mix)
        self.register_buffer("iu", iu)
        d_gram = K * K * (lmax + 1)
        d_pow = (lmax + 1) * iu.shape[1]
        d_bis = len(self.triples) * n_mix
        d_in = d_gram + d_pow + d_bis

        hidden, p = kw.get("hidden", 1024), kw.get("dropout", 0.2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, num_classes),
        )

    @staticmethod
    def _slog(f):
        return torch.sign(f) * torch.log1p(f.abs())

    def spectral_features(self, c_all):
        B, lmax = c_all.shape[0], self.lmax
        h = torch.cat([c_all[..., l**2:(l + 1) ** 2].reshape(B, -1) for l in range(lmax + 1)], -1)
        h = self.mix_bn(self.mix(self.in_bn(h)))

        ts, off = [], 0
        for l in range(lmax + 1):
            n = self.n_mix * (2 * l + 1)
            ts.append(h[:, off:off + n].reshape(B, self.n_mix, 2 * l + 1))
            off += n

        feats = []
        for t in ts:  # cross-channel power spectrum per degree
            g = t @ t.transpose(1, 2)
            feats.append(g[:, self.iu[0], self.iu[1]])
        for i, (a, b, c) in enumerate(self.triples):  # channelwise bispectrum
            w = getattr(self, f"w3j_{i}")
            feats.append(torch.einsum("mnp,bum,bun,bup->bu", w, ts[a], ts[b], ts[c]))
        return self._slog(torch.cat(feats, -1))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            c_all, c_rho = self.coefficients(x.float())
            f = torch.cat([self.gram_features(c_rho), self.spectral_features(c_all)], -1)
            return self.head(f)