# Controlled patient-sibling exposure — results

**Status: COMPLETE.** 72/72 runs verified (2 models × 2 folds {2,4} × 9 conditions {CLEAN, CTRL×4, SIB×4} × 2 training seeds {0,1}).
Frozen recipe unchanged (PR #1/#2). Design: `PROTOCOL_FROZEN.md`, `exposure/design_summary.json`. The SIB arms **deliberately create train/test patient overlap**; this is an intervention to measure how much same-patient exposure helps, not a leakage-free evaluation.

## Whole fixed-test balanced accuracy (mean of 2 seeds; descriptive)
| model | fold | CLEAN | CTRL 25/50/75/100 | SIB 25/50/75/100 |
|---|---|---|---|---|
| ResNet50 | 2 | 0.930 | 0.910/0.904/0.936/0.915 | 0.948/0.955/0.972/0.975 |
| ResNet50 | 4 | 0.947 | 0.961/0.951/0.955/0.947 | 0.963/0.955/0.947/0.948 |
| DenseNet121 | 2 | 0.916 | 0.936/0.928/0.924/0.927 | 0.945/0.956/0.967/0.971 |
| DenseNet121 | 4 | 0.937 | 0.945/0.949/0.947/0.945 | 0.953/0.961/0.947/0.960 |

Full metrics (BA, macro-F1, accuracy, per-class recall/F1, ECE, Brier) per condition/seed: `exposure_condition_metrics.csv`. Contrasts with 5,000-draw class-stratified patient-cluster bootstrap CIs (BA, macro-F1, accuracy, ECE, Brier, mean true-class probability): `exposure_contrasts.csv`. Dose-response/DiD: `exposure_dose_response_balanced_accuracy.csv`; Holm family: `exposure_holm_family_balanced_accuracy.csv`.

## Findings
**Swap (CTRL − CLEAN, adding other-patient images).** Mean ΔBA +0.003 over 16 model×fold×level cells; CI excludes 0 in 1/16. Swapping in matched other-patient images does not systematically change accuracy → null control behaves as intended.

**Sibling effect (SIB − CTRL, exposed patients E_L).** ΔBA positive in 16/16 cells, 95% CI excluding 0 in 7/16. Representative: ResNet50 fold 2 L=100 +0.059 (0.026, 0.104); DenseNet121 fold 2 L=100 +0.045 (0.007, 0.093); DenseNet121 fold 4 L=100 +0.019 (−0.002, 0.044); ResNet50 fold 4 L=100 +0.002 (−0.013, 0.015).
**The effect is NOT consistent across folds**: fold 2 shows +0.045 to +0.092 across levels for both models; fold 4 shows 0.00–0.03, mostly CI including 0 (ResNet50 fold 4 ≈ 0 at L=75–100). Calibration/probability metrics move in the same direction (Brier on the whole fixed test falls monotonically with SIB level in fold 2: ResNet50 0.0835→0.0371, DenseNet121 0.0740→0.0367).

**Spillover (SIB − CTRL, non-exposed patients).** Mean ΔBA −0.006; CI excludes 0 in 3/16 cells (1 positive, 2 negative), small N_L at high L (3–6 patients at L=100, uninformative). No consistent spillover.

**Difference-in-differences (E_L minus N_L).** Positive in 16/16 cells, CI excluding 0 in 8/16; DiD at L=100 rests on 3–6 unexposed patients and should be ignored.

**Dose-response.** The slope of the sibling contrast over L=25..100 is ≈0 and non-significant in every cell (−0.027 to +0.002 per 100 pp; all CIs include 0); pooled slope −0.010 (−0.073, 0.031), p=0.72. Pooled slope of the DiD (L=25–75) +0.042 (−0.063, 0.122), p=0.40. **No reproducible dose-response was demonstrated**: a sibling effect appears already at 25–50% exposure in fold 2 and does not clearly grow. (Whole-test BA in fold 2 does rise with SIB level, but that is mostly because the exposed fraction of the test set rises.)

## Holm-corrected frozen family (4 cells at L=100 on Q + pooled slope; two-sided bootstrap p)
| hypothesis | estimate (95% CI) | p raw | p Holm |
|---|---|---|---|
| ResNet50 fold 2 | +0.059 (0.026, 0.104) | <0.001 | **<0.001** |
| ResNet50 fold 4 | +0.002 (−0.013, 0.015) | 0.84 | 1.00 |
| DenseNet121 fold 2 | +0.045 (0.007, 0.093) | 0.018 | 0.072 |
| DenseNet121 fold 4 | +0.019 (−0.002, 0.044) | 0.087 | 0.26 |
| pooled dose-response slope | −0.010 (−0.073, 0.031) | 0.72 | 1.00 |

Only ResNet50 fold 2 survives Holm correction. Everything else (swap, spillover, DiD, per-level contrasts, other metrics) is exploratory and uncorrected.

## Limitations
* Only 2 folds (the pre-registered 2, of 5) and 2 training seeds; bootstrap intervals cover patient sampling for fixed trained models, not training randomness (seeds are not resampled).
* Fold 2 vs fold 4 heterogeneity is unexplained; cannot attribute to a mechanism from these data (see `TUMOR_REGION_RESULTS.md`).
* Small subpopulations (E_25 ≈ 10 patients; N_100 = 3–6) give wide/unstable intervals; BA on subsets with few patients per class is coarse.
* Same-patient exposure raises accuracy because sibling slices share anatomy/acquisition/lesion with the test slice; this does not identify *which* shared feature is used.

## Conclusion supported by the evidence
Same-patient (sibling) exposure improves exposed-patient accuracy over a matched other-patient control in every cell (direction 16/16), and not for unexposed patients, but magnitude is heterogeneous across folds (sizeable in fold 2, negligible-to-small in fold 4), no dose-response was demonstrated, and only one of four primary cells survives Holm correction. This is suggestive, not established, evidence of a patient-specific exposure benefit.

## Reproduce
```
python3 experiments/extension/make_exposure.py          # design (hash 5d4662eb…)
python3 experiments/extension/ext_driver.py             # trains/skips verified runs
python3 experiments/extension/analyze_exposure.py
python3 experiments/extension/analyze_dose.py
```
Note: in `exposure_dose_response_*.csv`, `nE` for `DiD_E_minus_N` rows is the number of non-exposed patients (N_L).
