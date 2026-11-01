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
"""
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps, Linear, wigner_3j, TensorProduct


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
        reflection_equivariant: bool = True,
    ):
        super().__init__()
        assert gram in ("none", "density_full", "density", "all")
        assert norm in ("layer", "batch")
        assert triples in ("all", "nonzero", "even") and pow_mode in ("triu", "diag", "none")
        K, C = n_shells, n_channels
        self.K, self.C, self.lmax, self.r_max = K, C, lmax, r_max
        self.bl = min(bispec_lmax, lmax)
        self.gram, self.pow_mode = gram, pow_mode
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

        if self.triples:
            base = -1 if reflection_equivariant else 1
            irreps_h = build_degree_irreps(self.m, 0, self.bl, reflection_equivariant)
            # one output block per triple: m channels of irrep l_c with parity p_a * p_b
            irreps_tp_out = Irreps([(self.m, (c, base ** (a + b))) for a, b, c in self.triples])
            instructions = [(a, b, i, "uuu", False) for i, (a, b, c) in enumerate(self.triples)]
            self.tp = TensorProduct(
                irreps_h, irreps_h, irreps_tp_out, instructions,
                irrep_normalization="none", path_normalization="none",
            )

            # step 2 (dot with t_c over the 2c+1 axis) as one gather + one index_add_
            gu, gp, seg = [], [], []
            for i, (a, b, c) in enumerate(self.triples):
                for u in range(self.m):
                    for p in range(2 * c + 1):
                        gu.append(u)
                        gp.append(c * c + p)  # position of (c, p) in cat(ts, -1)
                        seg.append(i * self.m + u)
            self.register_buffer("gu", torch.tensor(gu, dtype=torch.long))
            self.register_buffer("gp", torch.tensor(gp, dtype=torch.long))
            self.register_buffer("gseg", torch.tensor(seg, dtype=torch.long))

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
    
    def _bispec(self, ts):
        ts = ts[: self.bl + 1]
        h = torch.cat([t.flatten(1) for t in ts], -1)   # e3nn layout: per degree, (m, 2l+1) flattened
        z = self.tp(h, h)                               # (B, sum_i m*(2c_i+1)); z[i] = sum_mn w3j[m,n,:] t_a[m] t_b[n]
        T = torch.cat(ts, -1)                           # (B, m, (bl+1)^2)
        prod = z * T[:, self.gu, self.gp]               # pair each z component with t_c's matching component
        out = prod.new_zeros(z.shape[0], len(self.triples) * self.m).index_add_(1, self.gseg, prod)
        return [out]                                    # triple-major, channel-minor

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
            parts += self._bispec(ts)
        parts = [p if p.dim() == 2 else p.unsqueeze(-1) for p in parts]
        return slog(torch.cat(parts, -1))

    def forward(self, c):
        # features stay fp32 (SH / tensor products are touchy in bf16)
        with torch.autocast(device_type=c.device.type, enabled=False):
            return self.features(c.float())