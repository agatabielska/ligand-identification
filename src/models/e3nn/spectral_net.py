"""SpectralNet: SH coefficients (B, K*C, (lmax+1)^2) -> logits. Pure torch + e3nn, no Lightning.

Every simplification candidate is a constructor switch so one class covers the whole ablation grid:

  n_mix            learned equivariant channel mix per degree; 0 = no mix (identity) -> features
                   are then a fixed function of the coefficients (cacheable / probe-able).
                   With n_mix=0 the channels per degree are K*C.
  bispec_lmax      max degree in the bispectrum triples
  triples          which (l1<=l2<=l3) triples to keep:
                     "all"      every triple allowed by the triangle rule (original, 50 at lmax=6)
                     "nonzero"  drops the 15 triples that are identically 0: odd l1+l2+l3 with
                                l1==l2 or l2==l3 (3j antisymmetric under the column swap, but the
                                same channel tensor enters twice)
                     "even"     keeps only even l1+l2+l3 (also drops the pseudoscalar / parity-odd
                                triples such as (2,3,4), which flip sign under reflection)
  drop_l0_triples  remove (0,b,b) triples (nearly redundant with the power spectrum)
  pow_mode         power spectrum: "triu" (all channel pairs, original) | "diag" (per channel only)
                   | "none"
  gram             "none" | "density_full" (original, redundant) | "density" (upper triangle)
                   | "all" (upper triangle, every channel)
  norm             "layer" (original LayerNorm over features) | "batch" (per-feature standardisation)
  in_bn/mix_bn     the two e3nn BatchNorms around the mix
  n_hidden         0 = linear probe, 1 = one hidden layer, 2 = original (hidden -> hidden//2)
  bispec_mode      "loop" (original, one small einsum per triple)
                   | "batched" (one padded contraction)
                   | "sparse" (one gather + one index_add_ over the nonzero 3j entries)
"""
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps, Linear, wigner_3j


def build_degree_irreps(multiplicity: int, first_degree: int, last_degree: int, reflection_equivariant: bool) -> Irreps:
    """Build one irrep block for each angular degree in an inclusive range."""
    l_from, l_to = first_degree, last_degree
    base = -1 if reflection_equivariant else 1
    return Irreps([(multiplicity, (l, base ** l)) for l in range(l_from, l_to + 1)])


def slog(f: torch.Tensor) -> torch.Tensor:
    return torch.sign(f) * torch.log1p(f.abs())


def is_valid_bispectrum_triple(first: int, second: int, third: int, mode: str) -> bool:
    """Return whether a degree triple contributes a non-redundant contraction."""
    odd = (first + second + third) % 2 == 1
    if mode == "all":
        return True
    if mode == "even":
        return not odd
    return not (odd and (first == second or second == third))  # "nonzero"


class SpectralNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_shells: int = 16,
        lmax: int = 6,
        r_max: float = 1.0,
        n_channels: int = 3,
        n_mix: int = 32,
        bispec_lmax: int = 6,
        drop_l0_triples: bool = False,
        triples: str = "all",
        pow_mode: str = "triu",
        gram: str = "density_full",
        norm: str = "layer",
        in_bn: bool = True,
        mix_bn: bool = True,
        bispec_mode: str = "loop",
        reflection_equivariant: bool = True,
    ):
        super().__init__()
        assert gram in ("none", "density_full", "density", "all")
        assert norm in ("layer", "batch") and bispec_mode in ("loop", "batched", "sparse")
        assert triples in ("all", "nonzero", "even") and pow_mode in ("triu", "diag", "none")
        K, C = n_shells, n_channels
        self.K, self.C, self.lmax, self.r_max = K, C, lmax, r_max
        self.bl = min(bispec_lmax, lmax)
        self.gram, self.bispec_mode, self.pow_mode = gram, bispec_mode, pow_mode
        self.m = n_mix if n_mix > 0 else K * C  # channels per degree after the (optional) mix

        irreps_in = build_degree_irreps(K * C, 0, lmax, reflection_equivariant)
        self.in_bn = BatchNorm(irreps_in) if in_bn else nn.Identity()
        if n_mix > 0:
            irreps_mix = build_degree_irreps(n_mix, 0, lmax, reflection_equivariant)
            self.mix = Linear(irreps_in, irreps_mix)
            self.mix_bn = BatchNorm(irreps_mix) if mix_bn else nn.Identity()
        else:
            self.mix, self.mix_bn = nn.Identity(), nn.Identity()

        # triangle rule, l1 <= l2 <= l3, both parities, then the optional filters
        self.triples = [
            (a, b, c)
            for a in range(self.bl + 1)
            for b in range(a, self.bl + 1)
            for c in range(b, min(a + b, self.bl) + 1)
            if not (drop_l0_triples and a == 0)
            and is_valid_bispectrum_triple(a, b, c, triples)
        ]
        D = 2 * self.bl + 1
        w = torch.zeros(len(self.triples), D, D, D)
        for i, (a, b, c) in enumerate(self.triples):
            w[i, : 2 * a + 1, : 2 * b + 1, : 2 * c + 1] = wigner_3j(a, b, c).float()
        self.register_buffer("w3j_pad", w)
        t = torch.tensor(self.triples, dtype=torch.long).reshape(-1, 3)
        self.register_buffer("tA", t[:, 0].clone())
        self.register_buffer("tB", t[:, 1].clone())
        self.register_buffer("tC", t[:, 2].clone())

        if bispec_mode == "sparse":
            ia, ib, ic, wv, tid = [], [], [], [], []
            for i, (a, b, c) in enumerate(self.triples):
                w3 = wigner_3j(a, b, c).float()
                for mm, nn_, pp in (w3.abs() > 1e-9).nonzero().tolist():
                    ia.append(a * a + mm)
                    ib.append(b * b + nn_)
                    ic.append(c * c + pp)
                    wv.append(w3[mm, nn_, pp].item())
                    tid.append(i)
            self.register_buffer("sa", torch.tensor(ia, dtype=torch.long))
            self.register_buffer("sb", torch.tensor(ib, dtype=torch.long))
            self.register_buffer("sc", torch.tensor(ic, dtype=torch.long))
            self.register_buffer("st", torch.tensor(tid, dtype=torch.long))
            self.register_buffer("sw", torch.tensor(wv, dtype=torch.float32))

        self.register_buffer("iu_m", torch.triu_indices(self.m, self.m))
        self.register_buffer("iu_k", torch.triu_indices(K, K))

        L = lmax + 1
        n_tri = self.iu_k.shape[1]
        d_gram = {"none": 0, "density_full": K * K * L, "density": L * n_tri, "all": C * L * n_tri}[gram]
        self.d_pow = {"triu": L * self.iu_m.shape[1], "diag": L * self.m, "none": 0}[pow_mode]
        d_bis = len(self.triples) * self.m
        self.d_in = d_gram + self.d_pow + d_bis

    # ---------------------------------------------------------------- features
    # TODO: "density_full" and "density" act only on channel 0
    def gram_features(self, c):
        if self.gram == "none":
            return []

        B = c.shape[0]
        c = c.reshape(B, self.K, self.C, -1)
        
        if self.gram in ("density_full", "density"):
            channels = [0]
        else: 
            channels = range(self.C)

        parts = []
        for l in range(self.lmax + 1):
            cl = c[..., l * l:(l + 1) ** 2]  # (B, K, C, 2l+1)
            for ch in channels:
                x = cl[:, :, ch]
                g = x @ x.transpose(1, 2)
                
                if self.gram == "density_full":
                    parts.append(g.flatten(1))
                else:
                    parts.append(g[:, self.iu_k[0], self.iu_k[1]])

        return [torch.cat(parts, -1)]

    def _bispec_loop(self, ts):
        out = []
        for i, (a, b, c) in enumerate(self.triples):
            w = self.w3j_pad[i, : 2 * a + 1, : 2 * b + 1, : 2 * c + 1]
            out.append(torch.einsum("mnp,bum,bun,bup->bu", w, ts[a], ts[b], ts[c]))
        return out

    def _bispec_batched(self, ts):
        B, D = ts[0].shape[0], 2 * self.bl + 1
        T = ts[0].new_zeros(B, self.m, self.bl + 1, D)
        for l in range(self.bl + 1):
            T[:, :, l, : 2 * l + 1] = ts[l]
        Ta, Tb, Tc = T[:, :, self.tA], T[:, :, self.tB], T[:, :, self.tC]  # (B, m, P, D)
        z = torch.einsum("pmnq,bupq->bupmn", self.w3j_pad, Tc)
        out = (z * Ta.unsqueeze(-1) * Tb.unsqueeze(-2)).sum((-1, -2))  # (B, m, P)
        return [out.permute(0, 2, 1).flatten(1)]

    def _bispec_sparse(self, ts):
        T = torch.cat(ts[: self.bl + 1], -1)  # (B, m, (bl+1)^2), degree l at [l^2, (l+1)^2)
        prod = T[:, :, self.sa] * T[:, :, self.sb] * T[:, :, self.sc] * self.sw  # (B, m, nnz)
        out = prod.new_zeros(T.shape[0], self.m, len(self.triples)).index_add_(2, self.st, prod)
        return [out.permute(0, 2, 1).flatten(1)]  # triple-major, channel-minor (same as loop)

    def features(self, c):
        """c: (B, K*C, (lmax+1)^2) -> (B, d_in) invariant features (signed-log).

        Column order: [gram | power spectrum | bispectrum (triple-major, channel-minor)].
        """
        B = c.shape[0]
        parts = self.gram_features(c)

        h = torch.cat([c[..., l * l:(l + 1) ** 2].reshape(B, -1) for l in range(self.lmax + 1)], -1)
        h = self.mix_bn(self.mix(self.in_bn(h)))
        ts, off = [], 0
        for l in range(self.lmax + 1):
            n = self.m * (2 * l + 1)
            ts.append(h[:, off:off + n].reshape(B, self.m, 2 * l + 1))
            off += n

        if self.pow_mode == "triu":
            for t in ts:
                parts.append((t @ t.transpose(1, 2))[:, self.iu_m[0], self.iu_m[1]])
        elif self.pow_mode == "diag":
            for t in ts:
                parts.append((t * t).sum(-1))

        if self.triples:
            if self.bispec_mode == "loop":
                parts += self._bispec_loop(ts)
            elif self.bispec_mode == "batched":
                parts += self._bispec_batched(ts)
            else:
                parts += self._bispec_sparse(ts)
        parts = [p if p.dim() == 2 else p.unsqueeze(-1) for p in parts]
        return slog(torch.cat(parts, -1))

    def forward(self, c):
        # features stay fp32 (SH / tensor products are touchy in bf16)
        with torch.autocast(device_type=c.device.type, enabled=False):
            return self.features(c.float())