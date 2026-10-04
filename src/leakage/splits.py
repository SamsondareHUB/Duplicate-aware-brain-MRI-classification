"""Arm A (original patient-disjoint folds) and arm B (matched image-level stratified folds) + exposure table.

Arm B assignment uses ONLY (image_id, label, original fold). patient_id is never passed to make_image_level_folds.
"""
import numpy as np
import pandas as pd

CLASSES = (1, 2, 3)  # 1 meningioma, 2 glioma, 3 pituitary
FOLDS = (1, 2, 3, 4, 5)


def fold_class_matrix(labels, folds):
    m = pd.crosstab(pd.Series(folds, name="fold"), pd.Series(labels, name="label"))
    return m.reindex(index=FOLDS, columns=CLASSES, fill_value=0)


def make_image_level_folds(image_ids, labels, folds_a, seed):
    """Stratified random partition whose fold x class counts equal arm A's exactly. Returns fold_B array aligned
    with the inputs. Deterministic: per-class numpy Generator seeded by SeedSequence([seed, class])."""
    image_ids = np.asarray(image_ids, dtype=int); labels = np.asarray(labels, dtype=int)
    target = fold_class_matrix(labels, np.asarray(folds_a))
    fold_b = np.zeros(len(image_ids), dtype=int)
    for c in CLASSES:
        idx = np.where(labels == c)[0]
        idx = idx[np.argsort(image_ids[idx], kind="stable")]  # deterministic base order
        perm = np.random.default_rng(np.random.SeedSequence([seed, c])).permutation(idx)
        start = 0
        for f in FOLDS:
            n = int(target.loc[f, c])
            fold_b[perm[start:start + n]] = f
            start += n
        assert start == len(idx)
    return fold_b


def expected_same_fold(labels, folds_a):
    """Exact expected number/fraction of images whose arm-B fold equals their arm-A fold under the
    matched-margin random assignment: sum_c sum_f target[f][c]^2 / n_c."""
    t = fold_class_matrix(labels, folds_a)
    n_c = t.sum(axis=0)
    exp = float(((t ** 2) / n_c).sum().sum())
    return exp, exp / len(labels)


def exposure_table(image_ids, patient_ids, test_fold):
    """Per test image: slices of the same patient that are in training (= in a different test fold)."""
    d = pd.DataFrame({"image_id": image_ids, "patient_id": patient_ids, "fold": test_fold})
    n_p = d.groupby("patient_id").image_id.transform("size")
    in_fold = d.groupby(["patient_id", "fold"]).image_id.transform("size")  # this patient's slices in the same test fold
    d["n_other_total"] = n_p - 1
    d["n_other_train"] = n_p - in_fold
    d["frac_other_train"] = np.where(d.n_other_total > 0, d.n_other_train / d.n_other_total.where(d.n_other_total > 0, 1), np.nan)
    d["single_slice_patient"] = d.n_other_total == 0
    d["has_same_patient_train_slice"] = d.n_other_train > 0
    return d


def patient_overlap_stats(patient_ids, test_fold):
    d = pd.DataFrame({"p": patient_ids, "f": test_fold})
    folds_per_patient = d.groupby("p").f.nunique()
    per_fold = {}
    for f in FOLDS:
        test_p = set(d.p[d.f == f]); train_p = set(d.p[d.f != f])
        per_fold[int(f)] = {"test_images": int((d.f == f).sum()), "test_patients": len(test_p),
                            "test_patients_also_in_train": len(test_p & train_p),
                            "train_patients": len(train_p)}
    return {"patients_in_more_than_one_test_fold": int((folds_per_patient > 1).sum()),
            "n_patients": int(len(folds_per_patient)), "per_fold": per_fold}
