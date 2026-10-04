"""Frozen preprocessing for the sanity experiment (identical for both arms and every image).

raw int16 -> record dtype/min/max -> reject constant image -> float32 -> per-image min-max to [0,1]
-> bilinear resize (antialias=True) to 224x224. 3-channel replication + ImageNet normalisation happen on the fly
in the training script. Augmentation (flip, rotation) is applied after this step, also in training.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data_audit.matio import read_mat

SIZE = 224


def preprocess_image(arr):
    """Returns (tensor [224,224] float32 in [0,1], info dict). Raises on constant images."""
    info = {"orig_dtype": str(arr.dtype), "orig_min": float(arr.min()), "orig_max": float(arr.max()),
            "orig_shape": "x".join(map(str, arr.shape))}
    if arr.max() == arr.min():
        raise ValueError(f"constant-valued image (value {arr.max()}); refusing to normalise silently")
    x = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))
    x = (x - x.min()) / (x.max() - x.min())
    x = F.interpolate(x[None, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)[0, 0]
    return x.clamp_(0, 1), info


def load_all(manifest_csv, cache=ROOT / "data/interim/pre224.npy", log=ROOT / "results/sanity/preprocessing_log.csv"):
    m = pd.read_csv(manifest_csv)
    ids = m.image_id.astype(int).values
    if Path(cache).exists() and Path(log).exists():
        return ids, np.load(cache, mmap_mode="r")
    X = np.zeros((len(ids), SIZE, SIZE), np.float32); rows = []
    for k, (i, p) in enumerate(zip(ids, m.file_path)):
        r = read_mat(ROOT / p)
        assert r.readable, f"{p}: {r.error}"
        t, info = preprocess_image(r.image)
        X[k] = t.numpy(); rows.append({"image_id": int(i), **info})
    Path(cache).parent.mkdir(parents=True, exist_ok=True); np.save(cache, X)
    pd.DataFrame(rows).to_csv(log, index=False)
    return ids, X
