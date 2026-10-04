"""Two-stage near-duplicate screening. Stage A: cheap retrieval on 32x32 representations. Stage B: full-res
verification metrics on candidate pairs only. No threshold here is a duplicate definition."""
import numpy as np
from scipy.fft import dctn
from skimage.transform import resize
from skimage.metrics import structural_similarity

THUMB = 32


def thumbnail(a):
    return resize(np.asarray(a, dtype=np.float64), (THUMB, THUMB), anti_aliasing=True, order=1)


def phash64(t):
    d = dctn(t, norm="ortho")[:8, :8].ravel()
    return (d > np.median(d[1:])).astype(np.uint8)  # median excludes DC term


def zscore_vec(t):
    v = t.ravel() - t.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def verify_pair(a, b):
    """Full-resolution metrics. If shapes differ, the larger image is area-resized to the smaller one's shape
    (resized_for_comparison=True); frac_pixels_equal is then not meaningful and is NaN."""
    resized = a.shape != b.shape
    a = a.astype(np.float64); b = b.astype(np.float64)
    if resized:
        tgt = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
        a = resize(a, tgt, anti_aliasing=True, order=1) if a.shape != tgt else a
        b = resize(b, tgt, anti_aliasing=True, order=1) if b.shape != tgt else b
    rng = max(a.max(), b.max()) - min(a.min(), b.min())
    rng = rng if rng > 0 else 1.0
    return {"ncc_full": ncc(a, b),
            "ssim_full": float(structural_similarity(a, b, data_range=rng)),
            "mad_norm": float(np.abs(a - b).mean() / rng),
            "frac_pixels_equal": np.nan if resized else float((a == b).mean()),
            "resized_for_comparison": bool(resized)}


DIHEDRAL = {"flipud": lambda t: t[::-1, :], "fliplr": lambda t: t[:, ::-1], "rot90": lambda t: np.rot90(t, 1),
            "rot180": lambda t: np.rot90(t, 2), "rot270": lambda t: np.rot90(t, 3),
            "transpose": lambda t: t.T, "antitranspose": lambda t: np.rot90(t, 2).T}


def dihedral_best(Z):
    """Z: (n, THUMB*THUMB) unit-norm zscored thumbnails. Returns (best_ncc[n,n], best_name_index[n,n]) over the
    7 non-identity dihedral transforms of the second image."""
    n = len(Z)
    best = np.full((n, n), -2.0, np.float32); arg = np.zeros((n, n), np.int8)
    for k, f in enumerate(DIHEDRAL.values()):
        Zt = np.stack([f(z.reshape(THUMB, THUMB)).ravel() for z in Z])
        M = Z @ Zt.T
        m = M > best
        best[m] = M[m]; arg[m] = k
    return best, arg
