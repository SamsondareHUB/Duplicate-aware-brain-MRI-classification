"""Render the multi-model scientific report from generated result tables."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/multimodel"
MODELS = ("resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121")
ARMS = ("A", "B")
LABEL = {"A": "Patient-disjoint", "B": "Image-level seed 11"}
DISPLAY = {"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1", "accuracy": "Accuracy",
           "ece": "ECE (15 bins)", "brier": "Multiclass Brier"}


def fmt(x): return f"{float(x):.4f}"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def main():
    metrics = pd.read_csv(OUT / "metrics_summary.csv")
    classes = pd.read_csv(OUT / "per_class_metrics.csv")
    protocol = pd.read_csv(OUT / "protocol_differences.csv")
    pairwise = pd.read_csv(OUT / "pairwise_differences.csv")
    ranking = pd.read_csv(OUT / "ranking_summary.csv")
    tops = pd.read_csv(OUT / "top_ranked_frequencies.csv")
    manifests = [json.loads(p.read_text()) for p in (OUT / "runs").glob("*_run.json")]
    assert len(manifests) == 30 and all(len(r["epochs"]) == 10 and r["device"] == "mps" for r in manifests)
    envs = {m: json.loads((OUT / f"environment_{m}.json").read_text()) for m in MODELS if m != "resnet50"}
    assert all(e["device"] == "mps" for e in envs.values())
    lines = ["# Multi-model patient-overlap evaluation and ranking reliability", "",
             "**Status:** 30 of 30 new local Apple MPS runs completed. ResNet50 is reused from PR #1; it was not retrained.", "",
             "## Design and protocol identity", "",
             "This analysis uses the original 3,064-image, 233-patient, three-class official Figshare dataset and the frozen PR #1 splits. "
             "For prediction filenames and tables, **A means patient-disjoint** (`patient_disjoint_original.csv`) and "
             "**B means matched image-level seed 11** (`imagelevel_seed11.csv`). `EXPERIMENT_SPEC_MULTI_MODEL.md` reverses "
             "these letters; the split filenames, split hashes, and PR #1 code determine the mapping used here. "
             "All protocol differences below are B minus A (image-level minus patient-disjoint).", "",
             "The new models are MobileNetV2, EfficientNet-B0, and DenseNet121, each trained for five A and five B folds. "
             "The unchanged recipe uses per-image min-max normalization, antialiased bilinear resize to 224×224, "
             "three-channel replication, ImageNet normalization and pretrained weights, batch size 32, 10 epochs, "
             "AdamW (learning rate 1e-4, weight decay 1e-4), per-iteration cosine schedule, unweighted cross-entropy, "
             "horizontal flip probability 0.5, rotation ±10°, training seed 0, no early stopping, and final-epoch evaluation. "
             "All runs used physical batch size 32; no microbatching, gradient accumulation, mixed precision, "
             "or activation checkpointing was used.", "",
             "The original split SHA-256 values are `61bec5aaf9a5f4fce4adf89b186ffd6479eb0f6095ee20b8f60ef44274a3e49e` "
             "(A) and `e8b0442c107396fe66a989b8d448aa0e524c1c554823a711db3cc84beeee74f2` (B). "
             "No architecture-specific hyperparameters were tuned.", "",
             "## Pooled out-of-fold metrics", "",
             "Every model and protocol has exactly 3,064 unique out-of-fold predictions. Parentheses are class-stratified "
             "patient-cluster bootstrap 95% percentile confidence intervals (5,000 draws, seed 2026).", ""]
    metric_rows = []
    for m in MODELS:
        for arm in ARMS:
            d = metrics[(metrics.model == m) & (metrics.arm == arm)].set_index("metric")
            metric_rows.append([m, LABEL[arm]] + [f"{fmt(d.loc[k,'estimate'])} ({fmt(d.loc[k,'ci_lo'])}–{fmt(d.loc[k,'ci_hi'])})" for k in DISPLAY])
    lines += [table(["Model", "Protocol"] + list(DISPLAY.values()), metric_rows), ""]
    lines += ["## Paired protocol differences", "",
              "The same patient multiplicities are used for A and B within each model and jointly across all models. "
              "Positive performance differences favor image-level evaluation; negative ECE and Brier differences mean "
              "better apparent calibration under image-level evaluation.", ""]
    delta_rows = []
    for m in MODELS:
        d = protocol[protocol.model == m].set_index("metric")
        delta_rows.append([m] + [f"{fmt(d.loc[k,'delta_image_minus_patient'])} ({fmt(d.loc[k,'ci_lo'])}–{fmt(d.loc[k,'ci_hi'])})" for k in DISPLAY])
    lines += [table(["Model"] + list(DISPLAY.values()), delta_rows), ""]
    lines += ["## Class-specific performance", "",
              "Support is 708 meningioma, 1,426 glioma, and 930 pituitary images in each protocol. "
              "Full confidence intervals and confusion matrices are in `results/multimodel/per_class_metrics.csv` and "
              "`results/multimodel/confusion_matrices.csv`.", ""]
    class_rows = []
    for m in MODELS:
        for arm in ARMS:
            for c in ("meningioma", "glioma", "pituitary"):
                d = classes[(classes.model == m) & (classes.arm == arm) & (classes["class"] == c)].set_index("metric")
                class_rows.append([m, LABEL[arm], c] + [fmt(d.loc[k, "estimate"]) for k in ("precision", "recall", "f1")])
    lines += [table(["Model", "Protocol", "Class", "Precision", "Recall", "F1"], class_rows), ""]
    lines += ["## Model ranking reliability", "",
              "Ranks use pooled balanced accuracy or macro-F1. Kendall τ-b compares the four score orderings across "
              "protocols and handles ties. Bootstrap top-ranked frequencies count how often each model has the highest "
              "metric under the shared patient resamples; tied maxima split credit equally. These frequencies describe "
              "the bootstrap procedure, **not posterior probabilities**.", ""]
    rank_rows = []
    for k in ("balanced_accuracy", "macro_f1"):
        for arm in ARMS:
            d = ranking[(ranking.metric == k) & (ranking.arm == arm)].sort_values("rank")
            top = tops[(tops.metric == k) & (tops.arm == arm)].set_index("model")
            for r in d.itertuples():
                rank_rows.append([DISPLAY[k], LABEL[arm], r.rank, r.model, fmt(r.estimate),
                                  fmt(top.loc[r.model, "bootstrap_top_ranked_frequency"]), fmt(r.kendall_tau_between_protocols)])
    lines += [table(["Metric", "Protocol", "Rank", "Model", "Estimate", "Top frequency", "Cross-protocol τ-b"], rank_rows), ""]
    for k in ("balanced_accuracy", "macro_f1"):
        best_a = ranking[(ranking.metric == k) & (ranking.arm == "A")].sort_values("rank").iloc[0].model
        best_b = ranking[(ranking.metric == k) & (ranking.arm == "B")].sort_values("rank").iloc[0].model
        lines += [f"For {DISPLAY[k]}, the top listed architecture is **{best_a}** under A and **{best_b}** under B; "
                  f"the point ranking {'changes' if best_a != best_b else 'does not change'}. "
                  "Pairwise intervals below indicate how well the apparent ordering is resolved.", ""]
    pair_rows = []
    for r in pairwise.itertuples():
        pair_rows.append([DISPLAY[r.metric], LABEL[r.arm], f"{r.model_left} − {r.model_right}",
                          f"{fmt(r.difference_left_minus_right)} ({fmt(r.ci_lo)}–{fmt(r.ci_hi)})"])
    lines += ["### Pairwise model differences", "", table(["Metric", "Protocol", "Contrast", "Difference (95% CI)"], pair_rows), ""]
    lines += ["## Calibration and comparison figures", "",
              "Top-label ECE uses 15 fixed equal-width bins; multiclass Brier is the mean summed squared three-class "
              "probability error (range 0–2). The probability vectors in the OOF files regenerate both measures. "
              "Bin counts and values are in `results/multimodel/reliability_bins.csv`.", "",
              "![Reliability diagrams](../results/multimodel/figures/reliability_diagrams.png)", "",
              "![Model comparison](../results/multimodel/figures/model_comparison.png)", "",
              "## Environment and reproducibility", "",
              f"New runs: {len(manifests)}; all accepted runs completed 10 epochs; summed fold training/evaluation runtime: "
              f"{sum(r['runtime_s'] for r in manifests)/60:.1f} minutes. All new runs used the Mac's MPS backend. "
              "Safety-stopped attempts produced no accepted predictions and are excluded from the analysis. "
              "For the final two DenseNet121 folds, the guarded preflight required at least 8 GiB available RAM, "
              "at least 35% free macOS memory pressure, MPS availability, and no thermal warning; swap was diagnostic. "
              "During training, the monitor stopped on less than 5 GiB available RAM, less than 25% free memory pressure, "
              "thermal warning, or an unresponsive system check. These resource gates did not change the training recipe. "
              "The recorded software versions, input commit, seeds, and frozen configuration are in "
              "`results/multimodel/environment_*.json`; fold runtimes and epoch logs are in `results/multimodel/runs/`. "
              "ImageNet weight downloads and raw MRIs are excluded from Git.", "",
              "From a fresh checkout of this branch on a local Mac with MPS and the packages in `requirements-audit.txt`:", "",
              "```sh", "python3 scripts/download_figshare.py --extract", "python3 scripts/serial_model.py mobilenet_v2",
              "python3 scripts/serial_model.py efficientnet_b0", "python3 scripts/serial_model.py densenet121",
              "python3 scripts/analyze_multimodel.py", "python3 scripts/verify_multimodel.py",
              "python3 scripts/build_multimodel_report.py", "```", "",
              "The trainer resumes verified complete folds and refuses incomplete or invalid fold files for inspection. "
              "Each new model uses ten final-epoch runs. The original ResNet50 predictions in `results/sanity/` are used "
              "without retraining. `bootstrap_patient_draws.npz` stores the exact common patient multiplicities; "
              "`bootstrap_replicates.csv.gz` stores all five headline metrics for each model and protocol. "
              "`scripts/verify_multimodel.py` independently recomputes every headline point metric and selected draws "
              "directly from predictions.", "",
              "## Limitations", "",
              "1. One image-level split seed and one training seed were used. Patient bootstrap intervals do not include "
              "split or training randomness; MPS is not guaranteed bitwise deterministic.",
              "2. The A/B comparison is a protocol contrast. Fold composition changes along with patient overlap, so "
              "the results do not identify a per-image causal exposure effect.",
              "3. The original patient-disjoint mapping is strongly supported by its zero cross-fold patient overlap "
              "but is not author-confirmed; dataset provenance and site confounding remain unresolved.",
              "4. ECE estimates near zero have resampling bias, and top-ranked frequencies are descriptive bootstrap "
              "selection frequencies, not posterior probabilities.",
              "5. All models use the frozen ten-epoch recipe; this controls the comparison but does not establish that "
              "any architecture is optimally tuned.", ""]
    (ROOT / "reports/MULTIMODEL_RESULTS.md").write_text("\n".join(lines))
    print("Wrote reports/MULTIMODEL_RESULTS.md")


if __name__ == "__main__":
    main()
