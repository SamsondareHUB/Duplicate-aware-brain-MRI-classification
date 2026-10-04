import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leakage.splits import (make_image_level_folds, fold_class_matrix, expected_same_fold, exposure_table,
                            patient_overlap_stats)


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    patients = np.repeat(np.arange(n // 10), 10)             # 30 patients x 10 slices
    pat_cls = rng.integers(1, 4, n // 10)
    lab = np.repeat(pat_cls, 10)
    pat_fold = np.arange(n // 10) % 5 + 1
    fa = np.repeat(pat_fold, 10)                              # patient-disjoint folds
    return ids, patients.astype(str), lab, fa


def test_margins_and_permutation():
    ids, pid, lab, fa = _toy()
    fb = make_image_level_folds(ids, lab, fa, 11)
    assert (fold_class_matrix(lab, fa).values == fold_class_matrix(lab, fb).values).all()
    assert len(fb) == len(ids) and set(fb) == {1, 2, 3, 4, 5}


def test_deterministic_and_seed_sensitive_and_order_invariant():
    ids, pid, lab, fa = _toy()
    f1 = make_image_level_folds(ids, lab, fa, 11)
    assert (f1 == make_image_level_folds(ids, lab, fa, 11)).all()
    assert (f1 != make_image_level_folds(ids, lab, fa, 12)).any()
    p = np.random.default_rng(1).permutation(len(ids))        # input row order must not matter
    f_perm = make_image_level_folds(ids[p], lab[p], fa[p], 11)
    assert (f_perm == f1[p]).all()


def test_expected_same_fold_matches_simulation():
    ids, pid, lab, fa = _toy()
    exp, frac = expected_same_fold(lab, fa)
    sim = np.mean([(make_image_level_folds(ids, lab, fa, 100 + k) == fa).sum() for k in range(400)])
    assert abs(sim - exp) < 0.05 * exp and abs(frac - exp / len(ids)) < 1e-12


def test_expected_same_fold_closed_form_small():
    lab = np.array([1, 1, 1, 1]); fa = np.array([1, 1, 2, 2])   # one class, quotas 2/2 over n=4 -> 2*(4/4)... sum t^2/n
    assert expected_same_fold(lab, fa)[0] == (2 ** 2 + 2 ** 2) / 4


def test_exposure_patient_disjoint_is_zero_and_single_slice_handled():
    ids, pid, lab, fa = _toy()
    e = exposure_table(ids, pid, fa)
    assert (e.n_other_train == 0).all() and not e.has_same_patient_train_slice.any()
    e2 = exposure_table([1, 2, 3, 4], ["a", "a", "a", "b"], [1, 2, 2, 1])
    # a: slices in folds 1,2,2 ; image1 (fold1): other train = 2 of 2 ; image2 (fold2): other train = 1 of 2
    assert list(e2.n_other_train) == [2, 1, 1, 0] and list(e2.n_other_total) == [2, 2, 2, 0]
    assert list(e2.frac_other_train[:3]) == [1.0, 0.5, 0.5] and np.isnan(e2.frac_other_train[3])
    assert list(e2.single_slice_patient) == [False, False, False, True]
    assert list(e2.has_same_patient_train_slice) == [True, True, True, False]


def test_patient_overlap_stats():
    s = patient_overlap_stats(["a", "a", "b"], [1, 2, 2])
    assert s["patients_in_more_than_one_test_fold"] == 1
    assert s["per_fold"][2]["test_patients_also_in_train"] == 1
