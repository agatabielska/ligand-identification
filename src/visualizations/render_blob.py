"""3D rendering of blobs as shaded spheres or as a smoothed density surface.

Plotly's Scatter3d markers are screen-space sprites (flat squares/circles that don't
shade, occlude or scale with zoom). Here every point becomes real geometry instead:

* ``spheres`` – one lit icosphere per voxel/sample (Mesh3d), coloured by density.
* ``volume``  – points are splatted onto the voxel grid, Gaussian-smoothed and drawn as
  nested semi-transparent isosurfaces (Isosurface). Works for raw dense blobs and for
  the sparse 2000-point files produced by ``src.scripts.preprocess``.

CLI:
    python -m src.visualizations.render_blob <file.npz> [--mode spheres|volume|both] [--out fig.html]
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def load_blob(path) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Return (coords (N,3) float, values (N,), grid shape) for a dense or sparse .npz."""
    with np.load(path) as f:
        if "indices" in f:  # sparse output of preprocess.py
            return f["indices"].astype(float), f["values"].astype(float), tuple(f["shape"])
        dense = f[f.files[0]]
    return dense_to_points(dense) + (dense.shape,)


def dense_to_points(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argwhere(dense > 0)
    return idx.astype(float), dense[tuple(idx.T)].astype(float)


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdivisions):
        cache, new_f = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_f += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = new_f
    return np.array(verts), np.array(f)


def spheres_trace(coords, values, radius=0.6, subdivisions=None, colorscale="Viridis") -> go.Mesh3d:
    """All points as one instanced Mesh3d of lit spheres (radius in voxels)."""
    if subdivisions is None:  # keep the mesh around <=1M vertices
        subdivisions = 1 if len(coords) <= 5000 else 0
    sv, sf = _icosphere(subdivisions)
    n, nv = len(coords), len(sv)
    # float32/int32 halve the browser payload vs float64/int64 (plotly sends arrays as raw typed bytes)
    verts = (coords[:, None, :] + radius * sv[None]).reshape(-1, 3).astype(np.float32)
    faces = (sf[None] + nv * np.arange(n)[:, None, None]).reshape(-1, 3).astype(np.int32)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=np.repeat(values, nv).astype(np.float32), colorscale=colorscale,
        colorbar=dict(title="Density"), flatshading=False,
        lighting=dict(ambient=0.35, diffuse=0.8, specular=0.3, roughness=0.5, fresnel=0.2),
        lightposition=dict(x=100, y=200, z=150), hoverinfo="skip", name="spheres",
    )


def auto_sigma(coords) -> float:
    """Smoothing width ~ typical point spacing: ~0.75 voxel for full blobs, larger for sparse samples."""
    if len(coords) < 2:
        return 1.0
    d, _ = cKDTree(coords).query(coords, k=2)
    return max(0.75, 0.75 * float(np.median(d[:, 1])))


def splat_to_grid(coords, values, sigma, weight="values"):
    """Gaussian-splat points onto a cropped voxel grid. Returns (grid, origin)."""
    pad = int(np.ceil(3 * sigma)) + 1
    origin = np.floor(coords.min(0)).astype(int) - pad
    idx = np.rint(coords).astype(int) - origin
    grid = np.zeros(idx.max(0) + pad + 1)
    np.add.at(grid, tuple(idx.T), values if weight == "values" else 1.0)
    return gaussian_filter(grid, sigma) if sigma > 0 else grid, origin


def volume_trace(coords, values, sigma=None, weight="values", levels=(0.15, 0.6),
                 surface_count=3, opacity=0.25, colorscale="Viridis") -> go.Isosurface:
    """Nested isosurfaces of the smoothed density; ``levels`` are fractions of the max."""
    sigma = auto_sigma(coords) if sigma is None else sigma
    grid, origin = splat_to_grid(coords, values, sigma, weight)
    gx, gy, gz = np.meshgrid(*[np.arange(s, dtype=np.float32) + o for s, o in zip(grid.shape, origin)], indexing="ij")
    vmax = grid.max() or 1.0
    return go.Isosurface(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=(grid.ravel() / vmax).astype(np.float32),
        isomin=levels[0], isomax=levels[1], surface_count=surface_count, opacity=opacity,
        colorscale=colorscale, caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="Rel. density"), name=f"volume (σ={sigma:.2f} vx)", hoverinfo="skip",
    )


def _scene(shape):
    if shape is None:
        return dict(aspectmode="data")
    return dict(aspectmode="data", **{f"{a}axis": dict(title=f"{a.upper()} (voxel)", range=[0, s - 1])
                                      for a, s in zip("xyz", shape)})


def blob_figure(coords, values, shape=None, mode="spheres", title="", **kw) -> go.Figure:
    """mode: 'spheres', 'volume' or 'both' (side by side, cameras not linked)."""
    if len(coords) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data to display", showarrow=False)
        return fig
    sph_kw = {k: kw[k] for k in ("radius", "subdivisions") if k in kw}
    vol_kw = {k: kw[k] for k in ("sigma", "weight", "levels", "surface_count", "opacity") if k in kw}
    if mode == "both":
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
                            subplot_titles=("Spheres", "Smoothed density"))
        fig.add_trace(spheres_trace(coords, values, **sph_kw), 1, 1)
        vt = volume_trace(coords, values, **vol_kw)
        vt.colorbar.x = 1.02
        fig.data[0].colorbar.x = 0.45
        fig.add_trace(vt, 1, 2)
        fig.update_layout(scene=_scene(shape), scene2=_scene(shape))
    else:
        trace = spheres_trace(coords, values, **sph_kw) if mode == "spheres" else volume_trace(coords, values, **vol_kw)
        fig = go.Figure(trace)
        fig.update_layout(scene=_scene(shape))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", type=Path, help="raw dense blob (arr_0) or sparse preprocessed file")
    p.add_argument("--mode", choices=["spheres", "volume", "both"], default="both")
    p.add_argument("--radius", type=float, default=0.6, help="sphere radius in voxels")
    p.add_argument("--sigma", type=float, default=None, help="Gaussian smoothing in voxels (default: auto)")
    p.add_argument("--weight", choices=["values", "count"], default="values",
                   help="splat density values, or just point counts (use 'count' for probabilistic samples, "
                        "which are already drawn ∝ density)")
    p.add_argument("--levels", type=float, nargs=2, default=(0.15, 0.6), help="iso range as fractions of max")
    p.add_argument("--out", type=Path, default=None, help="write HTML here instead of opening a browser")
    a = p.parse_args()

    coords, values, shape = load_blob(a.npz)
    fig = blob_figure(coords, values, shape, a.mode, title=f"{a.npz.name}  ({len(coords)} points)",
                      radius=a.radius, sigma=a.sigma, weight=a.weight, levels=tuple(a.levels))
    if a.out:
        fig.write_html(a.out, include_plotlyjs="cdn")
        print(f"wrote {a.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
