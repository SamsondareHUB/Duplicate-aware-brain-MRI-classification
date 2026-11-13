"""Extract official tumorMask arrays, verify alignment with the image, and resize (nearest) to the 224x224 frozen grid.
Writes git-ignored data/interim/masks224.npy and a committed integrity table experiments/extension/mask_integrity.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, h5py, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
man = pd.read_csv(ROOT / "reports/manifests/image_manifest.csv")
M = np.zeros((len(man), 224, 224), np.uint8); rows = []
for k, (i, p) in enumerate(zip(man.image_id, man.file_path)):
    with h5py.File(ROOT / p, "r") as f:
        g = f["cjdata"]; img = np.array(g["image"]).T; m = np.array(g["tumorMask"]).T
    ok = m.shape == img.shape
    m = (m > 0).astype(np.float32)
    t = F.interpolate(torch.from_numpy(m)[None, None], size=(224, 224), mode="nearest")[0, 0].numpy().astype(np.uint8)
    M[k] = t
    rows.append({"image_id": int(i), "image_shape": "x".join(map(str, img.shape)), "mask_shape_matches_image": bool(ok),
                 "mask_frac_orig": float(m.mean()), "mask_frac_224": float(t.mean()), "mask_empty_224": bool(t.sum() == 0),
                 "mask_full_224": bool(t.mean() > 0.9), "mask_dtype_values": "binary" if set(np.unique(m)) <= {0.0, 1.0} else "other"})
np.save(ROOT / "data/interim/masks224.npy", M)
d = pd.DataFrame(rows); d.to_csv(ROOT / "experiments/extension/mask_integrity.csv", index=False)
print(d.describe(include="all").T.to_string())
