"""Parameter-free featurizer: point cloud -> spherical-harmonic coefficients per shell/channel.

Same maths as E3NNPointCloudModel._coefficients_chunk, plus:
  * n_channels in {1, 2, 3}: density | +occupancy | +density^2  (1 = density only)
  * trim_padding: drops all-zero trailing points per chunk (no numerical change, just less work)

Output layout: (B, K*C, (lmax+1)^2), channel index = k*C + c.
Because every channel is normalised independently, a 1-channel result is exactly
c_all[:, 0::3] of a 3-channel result, and lmax' < lmax is c_all[..., :(lmax'+1)**2].
"""
import torch
from e3nn.o3 import spherical_harmonics


def trim_padding(x: torch.Tensor) -> torch.Tensor:
    """x: (B, N, 4). Cut the point axis after the last point with density > 0 in the chunk."""
    n_pts = x.shape[1]
    pos = torch.arange(1, n_pts + 1, device=x.device)
    n = int(((x[..., 3] > 0) * pos).max().item())
    return x[:, : max(n, 1)]


def _chunk(x, lmax, centers, width, n_channels):
    pts, rho = x[..., :3], x[..., 3]
    rho = rho.clamp_min(0)
    valid = (rho > 0).to(rho.dtype)
    chans = torch.stack([rho, valid, rho * rho], dim=-1)[..., :n_channels]  # (B, N, C)
    chans = chans / chans.sum(1, keepdim=True).clamp(min=1e-8)

    com = (pts * chans[..., :1]).sum(1, keepdim=True)  # density-weighted centre of mass
    d = pts - com
    r = d.norm(dim=-1)

    Y = spherical_harmonics(list(range(lmax + 1)), d, normalize=True, normalization="integral")
    g = torch.exp(-(((r[..., None] - centers) / width) ** 2))  # (B, N, K)
    gw = (g[..., :, None] * chans[..., None, :]).flatten(2)  # (B, N, K*C)
    return torch.einsum("bnj,bnm->bjm", gw, Y)  # (B, K*C, M)


@torch.no_grad()
def sh_coefficients(x, lmax=6, n_shells=16, r_max=1.0, n_channels=3, chunk=1024, trim=True):
    x = x.float()
    centers = torch.linspace(0, r_max, n_shells, device=x.device)
    width = r_max / n_shells
    outs = []
    for xc in x.split(chunk):
        if trim:
            xc = trim_padding(xc)
        outs.append(_chunk(xc, lmax, centers, width, n_channels))
    return torch.cat(outs)
