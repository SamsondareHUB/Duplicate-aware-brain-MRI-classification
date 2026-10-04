"""Simple, publication-style figures from results/sanity/*.csv. Arms: A = blue (#2a78d6), B = orange (#eb6834)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/sanity"
FIG = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports/figures"; FIG.mkdir(parents=True, exist_ok=True)
CA, CB, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dc"
CLS = ["meningioma", "glioma", "pituitary"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
BS = pd.read_csv(RES / "bootstrap_summary.csv").set_index("metric")
LBL = {"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1", "accuracy": "Accuracy", "ece": "ECE (15 bins)", "brier": "Brier"}
ARMS = "A = patient-disjoint (original folds)   B = image-level (seed 11)"


def fig1():
    ms = ["balanced_accuracy", "macro_f1", "accuracy"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, m in enumerate(ms):
        r = BS.loc[m]
        for off, col, arm, lab in ((-0.12, CA, "A", "A"), (0.12, CB, "B", "B")):
            ax.errorbar(r[arm], i + off, xerr=[[r[arm] - r[f"{arm}_ci_lo"]], [r[f"{arm}_ci_hi"] - r[arm]]], fmt="o", color=col, ms=6, capsize=3, lw=1.6)
            ax.text(r[f"{arm}_ci_hi"] + 0.0015, i + off, f"{r[arm]:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([LBL[m] for m in ms]); ax.invert_yaxis(); ax.set_xlabel("Pooled out-of-fold value (95% patient-cluster bootstrap CI)")
    ax.plot([], [], "o", color=CA, label="A  patient-disjoint"); ax.plot([], [], "o", color=CB, label="B  image-level"); ax.legend(loc="lower left")
    ax.grid(axis="y", visible=False); fig.tight_layout(); fig.savefig(FIG / "fig1_primary_metrics.png"); plt.close(fig)


def fig2():
    order = ["balanced_accuracy", "macro_f1", "accuracy", "ece", "brier"] + [f"{k}_c{c}" for k in ("recall", "f1") for c in range(3)]
    names = [LBL[m] if m in LBL else f"{m.split('_c')[0].capitalize()} - {CLS[int(m[-1])]}" for m in order]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, m in enumerate(order):
        r = BS.loc[m]; ax.errorbar(r.delta_B_minus_A, i, xerr=[[r.delta_B_minus_A - r.delta_ci_lo], [r.delta_ci_hi - r.delta_B_minus_A]], fmt="o", color=CB, ms=5.5, capsize=3, lw=1.6)
    ax.axvline(0, color=INK, lw=0.9); ax.set_yticks(range(len(order))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Paired difference  B (image-level) - A (patient-disjoint), 95% CI"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "fig2_paired_differences.png"); plt.close(fig)


def fig3():
    R = pd.read_csv(RES / "bootstrap_replicates.csv.gz")
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    for ax, m in zip(axs, ["balanced_accuracy", "macro_f1", "ece", "brier"]):
        ax.hist(R[f"delta_{m}"], bins=40, color=CB, alpha=0.85, edgecolor="#fcfcfb", linewidth=0.5)
        r = BS.loc[m]; ax.axvline(0, color=INK, lw=0.9); ax.axvspan(r.delta_ci_lo, r.delta_ci_hi, color=CB, alpha=0.12)
        ax.set_title(LBL[m], fontsize=10); ax.set_xlabel("B - A"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("bootstrap replicates"); fig.tight_layout(); fig.savefig(FIG / "fig3_bootstrap_deltas.png"); plt.close(fig)


def fig4():
    cm = pd.read_csv(RES / "confusion_matrices.csv")
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, arm, title in zip(axs, ["A_patient_disjoint", "B_imagelevel_seed11"], ["A  patient-disjoint", "B  image-level"]):
        d = cm[cm.arm == arm].pivot(index="true", columns="pred", values="count").loc[CLS, CLS].values; pct = d / d.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{d[i, j]}\n({pct[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=9, color="white" if pct[i, j] > 0.55 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(CLS, rotation=20); ax.set_yticks(range(3)); ax.set_yticklabels(CLS)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
        for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig4_confusion_matrices.png"); plt.close(fig)


def fig5():
    rb = pd.read_csv(RES / "reliability_bins.csv"); cal = pd.read_csv(RES / "calibration_summary.csv").set_index("arm")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--"); 
    for arm, col, lab in (("A_patient_disjoint", CA, "A  patient-disjoint"), ("B_imagelevel_seed11", CB, "B  image-level")):
        d = rb[(rb.arm == arm) & (rb.n > 0)]
        ax.plot(d.mean_conf, d.accuracy, "o-", color=col, lw=1.8, ms=5, label=f"{lab} (ECE {cal.loc[arm, 'ece_15bin']:.4f})")
        ax2.step((d.lo + d.hi) / 2, d.n, where="mid", color=col, lw=1.6)
    ax.set_ylabel("Accuracy in bin"); ax.legend(loc="upper left"); ax.set_ylim(0, 1.02); ax2.set_yscale("log"); ax2.set_ylabel("images / bin"); ax2.set_xlabel("Top-label confidence (15 equal-width bins)")
    fig.tight_layout(); fig.savefig(FIG / "fig5_reliability.png"); plt.close(fig)


def fig6():
    ex = pd.read_csv(RES / "exposure_analysis.csv"); b = ex[(ex.section == "binned_B_with_A_descriptive") & (ex.population == "all")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
    for r_, var, order in ((0, "n_other_train", ["0", "1-9", "10-13", "14-17", "18+"]), (1, "frac_other_train", ["0", "(0,0.65]", "(0.65,0.8]", "(0.8,0.9]", "(0.9,1.0]"])):
        d = b[b.variable == var].set_index("level").reindex([o for o in order if o in set(b[b.variable == var].level)])
        for ax, (m, lab) in zip(axs[r_], (("correct", "Accuracy"), ("conf", "Mean confidence"), ("brier", "Brier (per image)"))):
            x = np.arange(len(d))
            ax.errorbar(x, d[f"{m}_mean"], yerr=[d[f"{m}_mean"] - d[f"{m}_ci_lo"], d[f"{m}_ci_hi"] - d[f"{m}_mean"]], fmt="o-", color=CB, capsize=3, lw=1.6, ms=5, label="B (CI: patient bootstrap)")
            ax.plot(x, d[f"A_{m}_mean"], "s--", color=CA, ms=4.5, lw=1.2, label="A, same images (descriptive)")
            ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={int(n)})" for l, n in zip(d.index, d.n_images)], fontsize=8); ax.set_ylabel(lab)
            ax.set_xlabel(var.replace("_", " ") + " bin")
    axs[0, 0].legend(fontsize=8, loc="lower right"); fig.suptitle("Exploratory, associational: arm-B performance by same-patient training exposure", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig6_exposure_response.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6): f()
    print("figures ->", FIG)
