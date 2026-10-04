"""Deterministic image-content hashes.

raw_array_hash      sha256( b"raw|<dtype-str>|<rows>x<cols>|" + C-order little-endian bytes of the array exactly
                    as stored (MATLAB orientation, original dtype, no value change). Same bytes AND dtype AND shape.
canonical_image_hash sha256( b"canon|<rows>x<cols>|" + C-order little-endian int32 bytes ). Values preserved
                    losslessly (int16->int32); dtype differences are erased, values/shape are not.
display_hash        sha256 of the README-style min-max rescaled uint8 image (255*(x-min)/(max-min), truncated).
                    LOSSY: can merge images differing only in intensity scale/offset. Reported separately, never
                    used as the primary exact-duplicate key.
Metadata (patient id, label, fold, filename, tumor mask/border) is EXCLUDED from all hashes.
"""
import hashlib
import numpy as np


def raw_array_hash(a):
    a = np.asarray(a)
    le = a.astype(a.dtype.newbyteorder("<"), copy=False)
    head = f"raw|{a.dtype.str.lstrip('<>|=')}|{a.shape[0]}x{a.shape[1]}|".encode()
    return hashlib.sha256(head + np.ascontiguousarray(le).tobytes()).hexdigest()


def canonical_image_hash(a):
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.integer):
        if not np.all(np.mod(a, 1) == 0):
            raise ValueError("non-integer pixel values; canonical int32 form would be lossy")
    c = np.ascontiguousarray(a.astype("<i4"))
    return hashlib.sha256(f"canon|{a.shape[0]}x{a.shape[1]}|".encode() + c.tobytes()).hexdigest()


def display_hash(a):
    x = np.asarray(a).astype(np.float64)
    lo, hi = x.min(), x.max()
    u = np.zeros(x.shape, np.uint8) if hi == lo else (255.0 / (hi - lo) * (x - lo)).astype(np.uint8)
    return hashlib.sha256(f"disp|{x.shape[0]}x{x.shape[1]}|".encode() + u.tobytes()).hexdigest()
