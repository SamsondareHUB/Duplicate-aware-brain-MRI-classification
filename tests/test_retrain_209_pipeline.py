"""Synthetic-fixture tests for the PROPOSED 209 pipeline. No training, no forward/backward pass, no GPU, no raw MRI."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import run_retrain_209 as R
import analyze_209 as AN
import serial_209 as S

DEV = ("synthetic-fixture",); NEP = 2


def fake_train(model, split, fold, calls=None):
    if calls is not None: calls.append((model, fold))
    te = split[split.fold == fold]
    rng = np.random.default_rng([int(fold), sum(map(ord, model)), len(split)])
    lg = rng.normal(0, 1.2, (len(te), 3)); lg[np.arange(len(te)), te.label.values - 1] += 2.5
    return {"logits": lg, "epochs": [{"epoch": i + 1, "train_loss": 0.5 / (i + 1), "train_acc": 0.9, "lr_end": 0.0} for i in range(NEP)],
            "runtime_s": 0.0, "device": DEV[0]}


def run(model, arm, out, **kw):
    return R.run_folds(model, arm, fake_train, out=out, n_epochs=NEP, allowed_devices=DEV, commit="fixture", cooldown_s=0, log=lambda *a: None, **kw)


def state(out, model, arm, fold):
    sp = pd.read_csv(R.SPL / R.FILES[arm], dtype={"patient_id": str, "stem": str}).sort_values("image_id").reset_index(drop=True)
    return R.fold_state(out, model, arm, fold, sp, R.sha(R.SPL / R.FILES[arm]), NEP, DEV)


def test_matrices():
    d, r = R.matrix("distinct"), R.matrix("repeated"); n = R.matrix("no_response")
    assert len(d) == 20 and set(d.arm) == {"A209"} and len(r) == 40 and set(r.arm) == {"A209", "B209"} and r.drop(columns="scenario").equals(n.drop(columns="scenario"))
    for m in (d, r):
        assert set(m.model) == set(R.MODELS) and set(m.fold) == {1, 2, 3, 4, 5} and set(m.train_seed) == {0} and m.run_id.is_unique
    assert abs(d.est_minutes.sum() / 60 - 1.63) < 0.02


def test_split_validate_ok():
    a, b = R.validate()
    assert len(a) == 3064 and a.stem.nunique() == 209


def test_schema_compat_with_pr1_pr2():
    pr2 = pd.read_csv(ROOT / "results/multimodel/oof_predictions_densenet121_A.csv", nrows=1)
    assert list(pr2.columns) == R.PR2_COMMON
    pr1 = pd.read_csv(ROOT / "results/sanity/oof_predictions_A.csv", nrows=1)
    assert set(pr1.columns) - {"label"} <= set(R.SCHEMA) | set()  # every PR#1 column (incl. ResNet50-only logits/exposure) is carried
    assert set(R.PR1_RESNET_ONLY) <= set(R.SCHEMA) and "stem" in R.SCHEMA and len(R.SCHEMA) == len(set(R.SCHEMA))


def test_full_synthetic_matrix_and_resume(tmp_path):
    calls = []
    for m in R.MODELS:
        for arm in ("A209", "B209"):
            assert R.run_folds(m, arm, lambda *a, c=calls: fake_train(*a, calls=c), out=tmp_path, n_epochs=NEP, allowed_devices=DEV, commit="fixture", cooldown_s=0, log=lambda *a: None, max_new_runs=2) == 2
    assert len(calls) == 16
    calls.clear()
    for m in R.MODELS:
        for arm in ("A209", "B209"):
            assert R.run_folds(m, arm, lambda *a, c=calls: fake_train(*a, calls=c), out=tmp_path, n_epochs=NEP, allowed_devices=DEV, commit="fixture", cooldown_s=0, log=lambda *a: None) == 3
    assert len(calls) == 24 and {f for _, f in calls} == {3, 4, 5}  # completed folds 1-2 verified and skipped, not retrained
    for m in R.MODELS:
        for arm in ("A209", "B209"):
            d = AN.load_arm(m, arm, tmp_path, n_epochs=NEP, allowed_devices=DEV)
            assert list(d.columns) == R.SCHEMA and len(d) == 3064 and d.image_id.is_unique and d.protocol.nunique() == 1


@pytest.mark.parametrize("breakage", ["truncate_pred", "bad_prob", "wrong_hash", "drop_json", "drop_pred", "tmp_left", "bad_json", "wrong_split_sha", "wrong_fold_rows"])
def test_corrupt_or_partial_fold_refused_not_skipped(tmp_path, breakage):
    run("mobilenet_v2", "A209", tmp_path, max_new_runs=1)
    p, j = R.paths(tmp_path, "mobilenet_v2", "A209", 1)
    if breakage == "truncate_pred": p.write_text(p.read_text()[:2000])
    elif breakage == "bad_prob":
        d = pd.read_csv(p, dtype={"patient_id": str, "stem": str}); d.loc[0, "prob_0"] = 0.5; d.to_csv(p, index=False)
    elif breakage == "wrong_hash": p.write_text(p.read_text() + "\n")
    elif breakage == "drop_json": j.unlink()
    elif breakage == "drop_pred": p.unlink()
    elif breakage == "tmp_left": p.with_suffix(".csv.tmp").write_text("x")
    elif breakage == "bad_json": j.write_text("{not json")
    elif breakage == "wrong_split_sha": x = json.loads(j.read_text()); x["split_sha256"] = "0" * 64; j.write_text(json.dumps(x))
    elif breakage == "wrong_fold_rows":
        d = pd.read_csv(p, dtype={"patient_id": str, "stem": str}).iloc[:-1]; d.to_csv(p, index=False); x = json.loads(j.read_text()); x["predictions_sha256"] = R.sha(p); j.write_text(json.dumps(x))
    calls = []
    with pytest.raises(R.RunIntegrityError):
        R.run_folds("mobilenet_v2", "A209", lambda *a: fake_train(*a, calls=calls), out=tmp_path, n_epochs=NEP, allowed_devices=DEV, commit="f", cooldown_s=0, log=lambda *a: None)
    assert calls == []  # nothing trained or overwritten
    moved = R.quarantine(tmp_path, "mobilenet_v2", "A209", 1)  # explicit recovery path
    assert moved and (tmp_path / "quarantine").exists()
    run("mobilenet_v2", "A209", tmp_path, max_new_runs=1)
    assert state(tmp_path, "mobilenet_v2", "A209", 1) == "complete"


def test_wrong_device_rejected(tmp_path):
    run("resnet50", "A209", tmp_path, max_new_runs=1)
    sp = pd.read_csv(R.SPL / R.FILES["A209"], dtype={"patient_id": str, "stem": str}).sort_values("image_id").reset_index(drop=True)
    with pytest.raises(R.RunIntegrityError):
        R.fold_state(tmp_path, "resnet50", "A209", 1, sp, R.sha(R.SPL / R.FILES["A209"]), NEP, ("mps",))  # a synthetic run can never pass as a real MPS run


def test_training_refused_without_approval_or_for_wrong_scenario():
    py = sys.executable; sc = str(ROOT / "scripts/run_retrain_209.py")
    for args in (["--train"], ["--train", "--model", "resnet50", "--arm", "A209", "--scenario", "distinct"],
                 ["--train", "--approved-by", "x", "--model", "resnet50", "--arm", "B209", "--scenario", "distinct"]):
        r = subprocess.run([py, sc, *args], capture_output=True, text=True)
        assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)
    assert not (R.OUT / "runs").exists()  # nothing was created in the real output dir


def test_monitor_stops_only_own_child_on_unsafe_host(tmp_path):
    unsafe_snap = lambda: {"available_gib": 1.0}
    t0 = time.time()
    rc, reason = S.monitored_child([sys.executable, "-c", "import time;time.sleep(60)"], tmp_path / "l.log", snapshot=unsafe_snap,
                                   check=lambda s: "available RAM below 5 GiB", interval=0.2)
    assert reason == "available RAM below 5 GiB" and rc != 0 and time.time() - t0 < 20
    os.kill(os.getpid(), 0)  # this process untouched
    rc, reason = S.monitored_child([sys.executable, "-c", "print('ok')"], tmp_path / "l.log", snapshot=lambda: {}, check=lambda s: None, interval=0.2)
    assert rc == 0 and reason is None


def test_preflight_wait_times_out_and_blocks_launch():
    snap = lambda: {"available_gib": 2.0, "pressure_free_pct": 80}
    t = [0.0]
    with pytest.raises(RuntimeError):
        S.wait_for_headroom(60, snapshot=snap, check=lambda s: None, sleep=lambda x: t.__setitem__(0, t[0] + x), now=lambda: t[0])  # <8 GiB
    S.wait_for_headroom(60, snapshot=lambda: {"available_gib": 16.0, "pressure_free_pct": 80}, check=lambda s: None, sleep=lambda x: None, now=lambda: 0.0)


@pytest.fixture(scope="module")
def synthetic_analysis(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs")
    for m in R.MODELS:
        for arm in ("A209", "B209"): run(m, arm, out)
    data = {(m, a): AN.load_arm(m, a, out, n_epochs=NEP, allowed_devices=DEV) for m in R.MODELS for a in ("A209", "B209")}
    o = pd.read_csv(ROOT / "data/splits/patient_disjoint_original.csv").sort_values("image_id")
    hist = {}
    for m in R.MODELS:
        h = data[m, "A209"].copy(); rng = np.random.default_rng(len(m)); lg = np.log(h[["prob_0", "prob_1", "prob_2"]].to_numpy()) + rng.normal(0, .3, (3064, 3))
        pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
        for k in range(3): h[f"prob_{k}"] = pr[:, k]
        hist[m] = h
    ao = tmp_path_factory.mktemp("an")
    res = AN.run_analysis(data, R.MODELS, ("A209", "B209"), ao, n_rep=150, seed=2026, hist=hist, unchanged_mask=(o.fold.values == data["resnet50", "A209"].test_fold.values))
    return data, ao, res


def test_analysis_outputs_stats_and_pairing(synthetic_analysis):
    data, ao, (point, reps, man) = synthetic_analysis
    for f in ("metrics_summary_209.csv", "per_class_metrics_209.csv", "confusion_matrices_209.csv", "reliability_bins_209.csv", "protocol_differences_209.csv",
              "pairwise_differences_209.csv", "ranking_summary_209.csv", "top_ranked_frequencies_209.csv", "ranking_protocol_tau_209.csv",
              "robustness_original233_vs_A209.csv", "analysis_manifest_209.json", "figures/reliability_diagrams_209.png"):
        assert (ao / f).exists(), f
    assert man["n_clusters"] == 209 and man["bootstrap_seed"] == 2026 and man["n_images"] == 3064
    d = data["densenet121", "A209"]; y = d.true_class.to_numpy(); pred = d.pred_class.to_numpy()
    ba = np.mean([(pred[y == c] == c).mean() for c in range(3)]); assert abs(ba - point["densenet121", "A209"]["balanced_accuracy"]) < 1e-12  # independent BA
    brier = np.mean(((d[["prob_0", "prob_1", "prob_2"]].to_numpy() - np.eye(3)[y]) ** 2).sum(1)); assert abs(brier - point["densenet121", "A209"]["brier"]) < 1e-12
    pdiff = pd.read_csv(ao / "protocol_differences_209.csv"); row = pdiff[(pdiff.model == "resnet50") & (pdiff.metric == "balanced_accuracy")].iloc[0]
    assert "MATCHED" in row.pairing and row.ci_lo <= row.delta_image_minus_patient + 0.1  # CI exists
    rb = pd.read_csv(ao / "robustness_original233_vs_A209.csv"); assert rb.pairing.str.contains("ROBUSTNESS ONLY").all() and not rb.pairing.str.contains("MATCHED").any()
    assert any(s.startswith("images_with_unchanged_test_fold_n") for s in rb.subset.unique())
    assert (ao / "protocol_differences_209.csv").read_text().count("historical") == 0  # A209 never paired with historical B


def test_a209_b209_identical_images_and_margins(synthetic_analysis):
    data, *_ = synthetic_analysis
    for m in R.MODELS:
        a, b = data[m, "A209"], data[m, "B209"]
        assert np.array_equal(a.image_id, b.image_id) and np.array_equal(a.true_class, b.true_class) and np.array_equal(a.stem, b.stem)
        assert (pd.crosstab(a.test_fold, a.true_class).values == pd.crosstab(b.test_fold, b.true_class).values).all()
        assert (a.groupby("stem").test_fold.nunique() == 1).all() and (b.groupby("stem").test_fold.nunique() > 1).any()


def test_bootstrap_is_deterministic_and_stem_clustered(synthetic_analysis, tmp_path):
    data, ao, (point, reps, man) = synthetic_analysis
    again = AN.run_analysis(data, R.MODELS, ("A209", "B209"), tmp_path, n_rep=150, seed=2026)[1]
    assert [r["balanced_accuracy"] for r in again["resnet50", "A209"]] == [r["balanced_accuracy"] for r in reps["resnet50", "A209"]]
    from leakage.bootstrap import stratified_multiplicities
    W = stratified_multiplicities(np.zeros(209, int), 10, 2026); assert W.shape == (10, 209) and (W.sum(1) == 209).all()
