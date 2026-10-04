import sys
from pathlib import Path
import numpy as np
import h5py
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_audit.hashing import raw_array_hash, canonical_image_hash, display_hash
from data_audit.duplicates import group_by_key, pair_category, union_find_clusters
from data_audit.folds import patient_fold_violations, verify_cvind_mapping
from data_audit.matio import read_mat


def _img(seed=0, shape=(8, 8)):
    return np.random.default_rng(seed).integers(0, 1000, shape).astype(np.int16)


def _write_mat(path, image, label=1, pid="123456"):
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g["image"] = image.T
        g["label"] = np.array([[float(label)]])
        g["PID"] = np.array([[ord(c)] for c in pid], dtype=np.uint16)


# ---- hashing
def test_hash_deterministic_and_content_sensitive():
    a = _img(1)
    assert raw_array_hash(a) == raw_array_hash(a.copy())
    assert canonical_image_hash(a) == canonical_image_hash(a.copy())
    b = a.copy(); b[0, 0] += 1
    assert raw_array_hash(a) != raw_array_hash(b)
    assert canonical_image_hash(a) != canonical_image_hash(b)


def test_hash_ignores_memory_layout_but_not_shape_or_orientation():
    a = _img(2)
    assert raw_array_hash(np.asfortranarray(a)) == raw_array_hash(a)
    assert raw_array_hash(a.reshape(4, 16)) != raw_array_hash(a)
    assert raw_array_hash(a.T) != raw_array_hash(a)


def test_raw_hash_dtype_sensitive_canonical_not():
    a = _img(3)
    assert raw_array_hash(a) != raw_array_hash(a.astype(np.int32))
    assert canonical_image_hash(a) == canonical_image_hash(a.astype(np.int32))


def test_display_hash_is_lossy_by_design():
    a = _img(4)
    assert display_hash(a) == display_hash(a * 2)  # intensity-scaling merges here, so never the primary key
    assert raw_array_hash(a) != raw_array_hash(a * 2)


# ---- duplicate grouping
def test_group_by_key_and_pair_category():
    g = group_by_key(["b", "a", "c", "d"], ["k1", "k1", "k2", "k3"])
    assert g == [["a", "b"]]
    assert pair_category("p1", "p1", 1, 1) == "within-patient/within-fold"
    assert pair_category("p1", "p1", 1, 2) == "within-patient/cross-fold"
    assert pair_category("p1", "p2", 1, 1) == "cross-patient/within-fold"
    assert pair_category("p1", "p2", 1, 2) == "cross-patient/cross-fold"


def test_union_find_clusters():
    r = union_find_clusters(5, [(0, 1), (1, 2), (3, 4)])
    assert r[0] == r[1] == r[2] and r[3] == r[4] and r[0] != r[3]


# ---- patient / fold integrity and cvind mapping
def test_patient_fold_violations():
    assert patient_fold_violations(["a", "a", "b"], [1, 1, 2]) == {}
    assert patient_fold_violations(["a", "a", "b"], [1, 2, 2]) == {"a": [1, 2]}


def test_cvind_mapping_supported_when_patient_disjoint_and_unresolved_otherwise():
    patients = np.repeat(np.arange(40), 5)             # 40 patients x 5 slices, file order
    cv = np.repeat(np.arange(40) % 5 + 1, 5)           # patient-disjoint folds in file order
    assert verify_cvind_mapping(patients, cv, n_perm=50)["status"] == "SUPPORTED"
    shuffled = np.random.default_rng(0).permutation(cv)  # wrong mapping
    assert verify_cvind_mapping(patients, shuffled, n_perm=50)["status"] == "UNRESOLVED"
    assert verify_cvind_mapping(patients, cv[:-1])["status"] == "UNRESOLVED"  # length mismatch


# ---- malformed .mat handling
def test_read_mat_ok(tmp_path):
    p = tmp_path / "1.mat"; a = _img(5)
    _write_mat(p, a, label=3, pid="MR0042")
    r = read_mat(p)
    assert r.readable and r.label == 3 and r.patient_id == "MR0042"
    assert np.array_equal(r.image, a)


def test_read_mat_malformed_never_raises(tmp_path):
    garbage = tmp_path / "bad.mat"; garbage.write_bytes(b"not an hdf5 file")
    empty = tmp_path / "empty.mat"; empty.write_bytes(b"")
    nogroup = tmp_path / "nogroup.mat"
    with h5py.File(nogroup, "w") as f:
        f["other"] = np.zeros(3)
    missing = tmp_path / "missing.mat"
    with h5py.File(missing, "w") as f:
        f.create_group("cjdata")["image"] = np.zeros((4, 4), np.int16)
    for p in (garbage, empty, nogroup, missing, tmp_path / "does_not_exist.mat"):
        r = read_mat(p)
        assert not r.readable and r.error
