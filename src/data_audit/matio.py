"""Reading Cheng et al. .mat files (MATLAB v7.3 / HDF5) with explicit failure reporting."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


@dataclass
class Record:
    path: str
    readable: bool
    error: str = ""
    image: "np.ndarray | None" = None
    label: "int | None" = None
    patient_id: "str | None" = None
    extra: dict = field(default_factory=dict)  # tumorBorder length, mask shape, field names


def _decode_pid(arr):
    a = np.asarray(arr).ravel()
    return "".join(chr(int(x)) for x in a)


def read_mat(path):
    """Return a Record. Never raises: every failure becomes readable=False with a reason."""
    path = str(path)
    try:
        import h5py
        with h5py.File(path, "r") as f:
            if "cjdata" not in f:
                return Record(path, False, f"missing 'cjdata' group; top-level keys={list(f.keys())}")
            g = f["cjdata"]
            missing = [k for k in ("image", "label", "PID") if k not in g]
            if missing:
                return Record(path, False, f"missing fields {missing}")
            image = np.array(g["image"]).T  # h5py reverses MATLAB axis order; .T restores MATLAB (row, col)
            if image.ndim != 2 or image.size == 0:
                return Record(path, False, f"unexpected image shape {image.shape}")
            label = int(np.asarray(g["label"]).ravel()[0])
            pid = _decode_pid(g["PID"])
            extra = {"fields": sorted(g.keys()),
                     "mask_shape": tuple(np.array(g["tumorMask"]).T.shape) if "tumorMask" in g else None}
            return Record(path, True, "", image, label, pid, extra)
    except Exception as e:  # malformed / truncated / not HDF5
        return Record(path, False, f"{type(e).__name__}: {e}")


def read_cvind(path):
    """Return the raw cvind vector as float array (no mapping to images is applied here)."""
    import h5py
    with h5py.File(str(path), "r") as f:
        return np.array(f["cvind"]).ravel()
