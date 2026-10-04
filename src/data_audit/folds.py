"""Patient/fold integrity and cvind->image mapping verification (no splits are created here)."""
from collections import defaultdict
import numpy as np


def patient_fold_violations(patients, folds):
    """patients/folds: parallel sequences. Returns {patient: sorted folds} for patients in >1 fold."""
    pf = defaultdict(set)
    for p, f in zip(patients, folds):
        pf[p].add(f)
    return {p: sorted(v) for p, v in pf.items() if len(v) > 1}


def verify_cvind_mapping(patients_in_file_order, cvind, n_perm=1000, seed=0):
    """Empirically test the hypothesis cvind[i] <-> i-th image in file-number order.

    Evidence: the number of patients split across folds under the hypothesised mapping, compared with the same
    count when cvind is randomly permuted (null = mapping carries no patient information). A mapping that is
    patient-disjoint while permutations are not is strong (not logical) proof."""
    patients = np.asarray(patients_in_file_order)
    cv = np.asarray(cvind)
    if len(patients) != len(cv):
        return {"status": "UNRESOLVED", "reason": f"length mismatch {len(patients)} vs {len(cv)}"}
    obs = len(patient_fold_violations(patients, cv))
    rng = np.random.default_rng(seed)
    null = [len(patient_fold_violations(patients, rng.permutation(cv))) for _ in range(n_perm)]
    ok = obs == 0 and min(null) > 0
    return {"status": "SUPPORTED" if ok else "UNRESOLVED", "observed_violations": obs,
            "null_min": int(min(null)), "null_mean": float(np.mean(null)), "n_perm": n_perm,
            "n_patients": int(len(set(patients)))}
