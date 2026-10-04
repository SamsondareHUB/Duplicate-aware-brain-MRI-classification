import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from leakage import metrics as M
from leakage.bootstrap import stratified_multiplicities, patient_stats, metrics_from_weights, run_bootstrap
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    logits = rng.normal(size=(n, 3)) + 2.0 * np.eye(3)[y]
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    return y, p


def test_classification_metrics_match_sklearn():
    y, p = _data(); pred = p.argmax(1); cm = M.confusion(y, pred)
    assert np.array_equal(cm, confusion_matrix(y, pred, labels=[0, 1, 2]))
    assert abs(M.balanced_accuracy(cm) - balanced_accuracy_score(y, pred)) < 1e-12
    assert abs(M.macro_f1(cm) - f1_score(y, pred, average="macro")) < 1e-12
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    a, b, c = M.prf_from_conf(cm)
    assert np.allclose(a, pr) and np.allclose(b, rc) and np.allclose(c, f1)


def test_brier_known_values():
    p = np.array([[1, 0, 0], [0, 1, 0]], float)
    assert M.brier(p, [0, 1]) == 0.0
    assert M.brier(p, [1, 0]) == 2.0                      # maximally wrong: (1)^2 + (1)^2 per image
    u = np.full((1, 3), 1 / 3)
    assert abs(M.brier(u, [0]) - (4 / 9 + 1 / 9 + 1 / 9)) < 1e-12


def test_ece_known_values_and_bin_edges():
    # perfectly calibrated-looking extremes: confident and always right -> ECE = 1 - conf
    p = np.tile([0.9, 0.05, 0.05], (10, 1))
    assert abs(M.ece(p, np.zeros(10, int)) - 0.1) < 1e-12
    assert abs(M.ece(p, np.ones(10, int)) - 0.9) < 1e-12   # always wrong, conf 0.9 -> |0-0.9|
    assert M.NBINS == 15
    assert list(M.bin_index([1 / 3, 0.4, 0.4000001, 0.6, 1.0])) == [4, 5, 6, 8, 14]  # (b/15,(b+1)/15]; 0.4 -> bin 5


def test_ece_weighted_equals_replication():
    y, p = _data(200, 1); w = np.random.default_rng(2).integers(0, 4, 200)
    rep = np.repeat(np.arange(200), w)
    assert abs(M.ece(p, y, w) - M.ece(p[rep], y[rep])) < 1e-12
    assert abs(M.brier(p, y, w) - M.brier(p[rep], y[rep])) < 1e-12


def test_bootstrap_stats_equal_direct_metrics_on_resampled_images():
    y, p = _data(300, 3); pat = np.repeat(np.arange(30), 10); pc = np.array([y[i * 10] for i in range(30)])
    y = np.repeat(pc, 10)                                    # one class per patient
    st = patient_stats(pat, 30, y, p)
    w = np.random.default_rng(4).integers(0, 3, 30).astype(float)
    got = metrics_from_weights(st, w)
    idx = np.repeat(np.arange(300), np.repeat(w, 10).astype(int))
    cm = M.confusion(y[idx], p[idx].argmax(1))
    assert abs(got["balanced_accuracy"] - M.balanced_accuracy(cm)) < 1e-12
    assert abs(got["macro_f1"] - M.macro_f1(cm)) < 1e-12
    assert abs(got["ece"] - M.ece(p[idx], y[idx])) < 1e-12
    assert abs(got["brier"] - M.brier(p[idx], y[idx])) < 1e-12


def test_bootstrap_class_stratified_pairing_and_seed():
    pc = np.array([0] * 10 + [1] * 15 + [2] * 5)
    W1 = stratified_multiplicities(pc, 50, 2026); W2 = stratified_multiplicities(pc, 50, 2026)
    assert (W1 == W2).all() and (stratified_multiplicities(pc, 50, 7) != W1).any()
    for c in (0, 1, 2):                                       # class composition preserved in every replicate
        assert (W1[:, pc == c].sum(1) == (pc == c).sum()).all()


def test_bootstrap_uses_identical_multiplicities_for_both_arms():
    y, pa = _data(300, 5); _, pb = _data(300, 6); pat = np.repeat(np.arange(30), 10)
    pc = np.array([y[i * 10] for i in range(30)]); y = np.repeat(pc, 10)
    sa, sb = patient_stats(pat, 30, y, pa), patient_stats(pat, 30, y, pb)
    rows, W = run_bootstrap(sa, sb, pc, n_rep=20, seed=2026)
    for r in range(20):
        a = metrics_from_weights(sa, W[r].astype(float)); b = metrics_from_weights(sb, W[r].astype(float))
        assert abs(rows[r]["delta_balanced_accuracy"] - (b["balanced_accuracy"] - a["balanced_accuracy"])) < 1e-12
    rows2, _ = run_bootstrap(sa, sb, pc, n_rep=20, seed=2026)
    assert rows == rows2


def test_oof_coverage_checker():
    from leakage.oof import check_oof
    a = pd.DataFrame({"image_id": [1, 2, 3], "true_class": [0, 1, 2]})
    check_oof(a, a.copy(), expected_n=3)
    import pytest
    with pytest.raises(AssertionError): check_oof(a.iloc[:2], a.iloc[:2], expected_n=3)
    with pytest.raises(AssertionError): check_oof(pd.concat([a, a.iloc[:1]]), pd.concat([a, a.iloc[:1]]), expected_n=3)
    b = a.copy(); b.loc[0, "true_class"] = 2
    with pytest.raises(AssertionError): check_oof(a, b, expected_n=3)
