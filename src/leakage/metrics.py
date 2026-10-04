"""Metrics with explicit, frozen definitions. All functions accept optional per-image weights (bootstrap multiplicities).

ECE      top-label ECE, 15 equal-width bins on top-label confidence: bin b = (b/15, (b+1)/15] (first bin includes 0);
         ECE = sum_b (n_b/N) * |acc_b - conf_b|.
Brier    multiclass: mean over images of sum over classes (p_k - 1[y=k])^2.   (range 0..2)
BA       macro-average of per-class recall.  macro-F1: unweighted mean of per-class F1 (0 when P+R=0).
"""
import numpy as np

K = 3
NBINS = 15


def confusion(y, pred, w=None):
    w = np.ones(len(y)) if w is None else w
    return np.bincount(np.asarray(y) * K + np.asarray(pred), weights=w, minlength=K * K).reshape(K, K)


def prf_from_conf(cm):
    tp = np.diag(cm); rec_d = cm.sum(1); prec_d = cm.sum(0)
    rec = np.divide(tp, rec_d, out=np.zeros(K), where=rec_d > 0)
    prec = np.divide(tp, prec_d, out=np.zeros(K), where=prec_d > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return prec, rec, f1


def balanced_accuracy(cm): return float(prf_from_conf(cm)[1].mean())
def macro_f1(cm): return float(prf_from_conf(cm)[2].mean())
def accuracy(cm): return float(np.trace(cm) / cm.sum())


def brier_per_image(probs, y):
    onehot = np.eye(K)[np.asarray(y)]
    return ((np.asarray(probs) - onehot) ** 2).sum(1)


def brier(probs, y, w=None):
    b = brier_per_image(probs, y)
    return float(np.average(b, weights=w))


def bin_index(conf):
    return np.clip(np.ceil(np.asarray(conf) * NBINS).astype(int) - 1, 0, NBINS - 1)


def ece_bins(probs, y, w=None):
    """Returns per-bin (weight, sum_conf, sum_correct) arrays of length 15."""
    probs = np.asarray(probs); conf = probs.max(1); correct = (probs.argmax(1) == np.asarray(y)).astype(float)
    w = np.ones(len(conf)) if w is None else w
    b = bin_index(conf)
    return (np.bincount(b, weights=w, minlength=NBINS), np.bincount(b, weights=w * conf, minlength=NBINS),
            np.bincount(b, weights=w * correct, minlength=NBINS))


def ece_from_bins(n, sc, sa):
    N = n.sum()
    nz = n > 0
    return float((np.abs(sa[nz] - sc[nz])).sum() / N)  # sum_b n_b/N * |acc_b - conf_b| = sum_b |sa_b - sc_b| / N


def ece(probs, y, w=None):
    return ece_from_bins(*ece_bins(probs, y, w))
