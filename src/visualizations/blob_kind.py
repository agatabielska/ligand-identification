"""Recognise which kind of blob an .npz holds and return it as a dense grid.

Rules, checked on 300 files of each dataset (18 Oct 2026):
  raw            one dense array (``arr_0``); negative values present -> unthresholded
  uniform        sparse {indices, values, shape}, every index at s//2 + k*s for a stride s > 1
                 (96 % of uniform files, none of the raw or probabilistic ones)
  probabilistic  sparse, exactly MAX_POINTS points (98 % of probabilistic files, no uniform ones)
  small          sparse, fewer points, no lattice: the source blob had <= MAX_POINTS voxels and was
                 saved unchanged, so the stack can't be told (2-4 % of files)

Self-check: python -m src.visualizations.blob_kind
"""
from pathlib import Path

import numpy as np

MAX_POINTS = 2000  # max_blob_size in src/scripts/preprocess.py


def lattice_stride(idx: np.ndarray, max_stride: int = 32) -> int:
    """Stride s > 1 if all indices lie on UniformSelectionTransform's output lattice (s//2 + k*s), else 1."""
    if len(idx) < 8:  # too few points to tell a lattice from chance
        return 1
    for s in range(max_stride, 1, -1):  # largest first: a stride-6 lattice also passes the stride-2 test
        if np.all(idx % s == s // 2):
            return s
    return 1


def recognise(file) -> tuple[np.ndarray, dict]:
    """Load a raw or preprocessed blob .npz (path or file-like) as a dense float64 grid, plus what it is.

    info: kind, description, density_sampled (points were drawn with probability ∝ density), pdb_id, ligand.
    Raises ValueError for any other file.
    """
    with np.load(file) as f:
        if {"indices", "values", "shape"} <= set(f.files):
            idx = f["indices"]
            dense = np.zeros(tuple(f["shape"]))
            dense[tuple(idx.T)] = f["values"]
        elif len(f.files) == 1:
            idx, dense = None, f[f.files[0]].astype(np.float64)
        else:
            raise ValueError(f"Not a blob file: expected one array or indices/values/shape, found {f.files}")
    if dense.ndim != 3:
        raise ValueError(f"Expected a 3D grid, got a {dense.ndim}D array")

    if idx is None:
        n, n_neg = int(np.count_nonzero(dense > 0)), int(np.count_nonzero(dense < 0))
        if n_neg:
            kind, desc = "raw unthresholded", (
                f"Raw unthresholded blob: {n} positive and {n_neg} negative voxels. The plots show positive density "
                "only, and ProbabilisticSelectionTransform raises on negative values.")
        else:
            kind, desc = "raw thresholded", f"Raw thresholded blob with {n} voxels > 0, as in data/cryoem_blobs."
    else:
        n, s = len(idx), lattice_stride(idx)
        if s > 1:
            kind, desc = "uniform", f"Preprocessed file, uniform stack: {n} points on a stride-{s} max-pooling lattice."
        elif n == MAX_POINTS:
            kind, desc = "probabilistic", (
                f"Preprocessed file, probabilistic stack (or normalization stack: the file can't tell them apart): "
                f"{n} points sampled with probability ∝ density.")
        else:
            kind, desc = "small", (
                f"Preprocessed file with {n} points: the source blob had ≤ {MAX_POINTS} voxels and was saved "
                "unchanged, so the transform stack can't be told.")
    if n <= MAX_POINTS:
        desc += f" At ≤ {MAX_POINTS} voxels the selection transforms (default max_blob_size) leave it unchanged."

    # shortcut: label read from the file name; a code outside the trained classes is stored as 'rare', which needs
    # the dataset's class list to tell
    parts = Path(getattr(file, "name", file)).stem.split("_")
    pdb_id, ligand = (parts[0], parts[-1]) if len(parts) >= 4 else (None, None)
    return dense, dict(kind=kind, description=desc, density_sampled=kind == "probabilistic",
                       pdb_id=pdb_id, ligand=ligand)


if __name__ == "__main__":
    import io

    def npz(**arrays):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        buf.seek(0)
        buf.name = "7ABC_A_401_NAG.npz"
        return buf

    rng = np.random.default_rng(0)
    raw = np.zeros((30, 30, 30))
    raw[5:25, 5:25, 5:25] = rng.random((20, 20, 20)) + 0.1  # 8000 voxels
    pts = np.argwhere(raw > 0)

    def sparse(idx):
        return npz(indices=idx, values=raw[tuple(idx.T)], shape=np.array(raw.shape))

    cases = {
        "raw thresholded": npz(arr_0=raw),
        "raw unthresholded": npz(arr_0=raw - 0.5),
        "uniform": sparse(pts[np.all(pts % 3 == 1, axis=1)]),
        "probabilistic": sparse(pts[rng.choice(len(pts), MAX_POINTS, replace=False)]),
        "small": sparse(pts[rng.choice(len(pts), 500, replace=False)]),
    }
    for expected, f in cases.items():
        dense, info = recognise(f)
        assert info["kind"] == expected, (expected, info)
        assert dense.shape == raw.shape and (info["pdb_id"], info["ligand"]) == ("7ABC", "NAG"), info
    f = cases["uniform"]
    f.seek(0)
    assert np.count_nonzero(recognise(f)[0]) == 216  # densified sparse keeps every point
    try:
        recognise(npz(arr_0=np.zeros((3, 3))))
        raise AssertionError("2D array accepted")
    except ValueError:
        pass
    print("blob_kind self-check passed")
