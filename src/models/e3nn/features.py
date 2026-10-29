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


def _extract_sh_features(x, lmax, centers, width, n_channels):
    coords, density = x[..., :3], x[..., 3]
    density = density.clamp_min(0)
    valid = (density > 0).to(density.dtype)
    channels = torch.stack([density, valid, density * density], dim=-1)[..., :n_channels]  # (B, N, C)
    channels = channels / channels.sum(1, keepdim=True).clamp(min=1e-8)

    density_weighted_centre = (coords * channels[..., :1]).sum(1, keepdim=True)
    distances_from_center = coords - density_weighted_centre
    radius = distances_from_center.norm(dim=-1)

    harmonics = spherical_harmonics(list(range(lmax + 1)), distances_from_center, normalize=True, normalization="integral")
    gaus_weights = torch.exp(-(((radius[..., None] - centers) / width) ** 2))  # (B, N, K)
    weighted_channels = (gaus_weights[..., :, None] * channels[..., None, :]).flatten(2)  # (B, N, K*C)
    return torch.einsum("bnj,bnm->bjm", weighted_channels, harmonics)  # (B, K*C, M)


@torch.no_grad()
def sh_coefficients(x, lmax=6, n_shells=16, r_max=1.0, n_channels=3, chunk=1024, trim=True):
    x = x.float()
    centers = torch.linspace(0, r_max, n_shells, device=x.device)
    width = r_max / n_shells
    outs = []
    for xc in x.split(chunk):
        if trim:
            xc = trim_padding(xc)
        outs.append(_extract_sh_features(xc, lmax, centers, width, n_channels))
    return torch.cat(outs)
