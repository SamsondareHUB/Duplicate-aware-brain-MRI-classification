"""Independent headline recomputation from saved prediction and draw files."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/multimodel"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
ARMS = ("A", "B")


def independent_metrics(d, weights=None):
    y = d.true_class.to_numpy(dtype=int)
    p = d[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=float)
    pred = p.argmax(1)
    conf = p.max(1)
    w = np.ones(len(d)) if weights is None else np.asarray(weights)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2], sample_weight=w)
    recall = np.diag(cm) / cm.sum(axis=1)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall)>0)
    # Right-closed equal-width confidence bins, with confidence 0 in the first.
    bin_id = np.maximum(0, np.minimum(14, np.ceil(conf*15).astype(int)-1))
    ece = sum(abs(w[bin_id==b] @ ((pred[bin_id==b]==y[bin_id==b]).astype(float)-conf[bin_id==b])) for b in range(15)) / w.sum()
    one_hot = np.eye(3)[y]
    brier = np.average(((p-one_hot)**2).sum(axis=1), weights=w)
    return {"balanced_accuracy": recall.mean(), "macro_f1": f1.mean(),
            "accuracy": np.trace(cm)/cm.sum(), "ece": ece, "brier": brier}


def main():
    summary = pd.read_csv(OUT / "metrics_summary.csv")
    draws = np.load(OUT / "bootstrap_patient_draws.npz")
    patient_id = draws["patient_id"].astype(str)
    W = draws["multiplicities"]
    boot = pd.read_csv(OUT / "bootstrap_replicates.csv.gz")
    assert W.shape == (5000, 233)
    assert len(boot) == 5000*8
    for model, arm in itertools.product(MODELS, ARMS):
        d = pd.read_csv(OUT / f"oof_predictions_{model}_{arm}.csv", dtype={"patient_id": str})
        assert len(d) == 3064 and d.image_id.is_unique
        p = d[["prob_0", "prob_1", "prob_2"]].to_numpy()
        assert np.allclose(p.sum(axis=1), 1, atol=1e-6)
        assert np.array_equal(d.pred_class, p.argmax(axis=1))
        point = independent_metrics(d)
        published = summary[(summary.model==model)&(summary.arm==arm)].set_index("metric")
        for k, value in point.items():
            assert abs(value-published.loc[k, "estimate"]) < 1e-10, (model, arm, k)
        codes = np.searchsorted(patient_id, d.patient_id.to_numpy())
        assert np.array_equal(patient_id[codes], d.patient_id.to_numpy())
        for r in (0, 2026, 4999):
            measured = independent_metrics(d, W[r, codes])
            published_rep = boot[(boot.model==model)&(boot.arm==arm)&(boot.replicate==r)].iloc[0]
            for k, value in measured.items():
                assert abs(value-published_rep[k]) < 1e-10, (model, arm, r, k)
    print("Independent verification passed: 8 OOF datasets, 40 point metrics, 24 paired bootstrap spot checks")


if __name__ == "__main__":
    main()
