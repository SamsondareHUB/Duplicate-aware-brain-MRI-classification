# Patient-Overlap Sanity Experiment (ResNet50, one split seed, one training seed)

**Question.** Holding model, training recipe and training-set size/class mix fixed, how much does evaluating with
patient overlap (image-level folds) change reported performance and calibration, relative to the original
patient-disjoint folds?

**Scope.** A go/no-go sanity experiment: one architecture, image-level split seed 11, training seed 0, 10 runs.
It is *not* a final effect estimate and says nothing yet about model rankings.

**Terminology.** Exact-duplicate leakage was confirmed absent and cross-patient near-identical duplication was not
supported (see `DATASET_AUDIT.md`). The construct tested here is **patient-level overlap / same-patient neighbouring-slice
dependence**. Same-patient slices are *not* called near-duplicates.

Labels used below: **CONFIRMED RESULT** (measured, pre-specified analysis, for this configuration), **EXPLORATORY
RESULT** (associational / not pre-specified as inferential), **LIMITATION**.

## Design (frozen before training)
- **Arm A:** original `cvind` patient-disjoint five folds (0 patients in more than one test fold).
- **Arm B:** image-level stratified five folds, seed 11, with the exact fold sizes (542/679/572/628/643) and exact
  fold x class counts of arm A; every image is a test image exactly once; `patient_id` was not used in the assignment.
  Split SHA-256 values and all validation checks: `data/splits/split_validation.json`. Experiment-input commit:
  `e78a2b735352b6456a91a807897adc4623f3298e`.
- **Model/training (identical for both arms, nothing tuned):** torchvision ResNet50, `IMAGENET1K_V1`; 224x224 input
  (per-image min-max to [0,1] -> antialiased bilinear resize -> 3-channel replicate -> ImageNet normalisation, same for
  all 3,064 images including the fifteen 256x256 ones; no constant-valued image exists, which the code would reject);
  AdamW lr 1e-4, weight decay 1e-4, cosine schedule to 0, batch 32, **10 epochs**, cross-entropy without class weights,
  horizontal flip p=0.5 and rotation +/-10 degrees, training seed 0, no early stopping, no validation-based selection,
  final epoch evaluated, raw softmax used for calibration.
- **Compute:** Apple MPS for all 10 runs (same backend for both arms), 62.8 min total, torch 2.11.0, torchvision 0.26.0,
  numpy 2.3.2, pandas 2.3.3, Python 3.13.6, macOS 15.6.1 arm64. MPS does not guarantee bitwise determinism.
  `results/sanity/environment.json`, `run_manifest.json`.
- **Metrics (pooled out-of-fold, 3,064 predictions per arm).** Balanced accuracy and macro-F1 (primary); accuracy,
  per-class precision/recall/F1, confusion matrices (secondary); top-label ECE with 15 equal-width bins; multiclass Brier
  = mean over images of the sum over classes of squared probability error (range 0-2).
- **Uncertainty.** Class-stratified patient-cluster bootstrap (resample patients with replacement within each tumour class;
  every patient has exactly one class), 5,000 replicates, seed 2026, identical patient multiplicities for A and B in every
  replicate (paired). Percentile 95% CIs. Delta = B - A.
- **Decision rule (fixed before results; `scripts/analyze_sanity.py`):** GO if a paired 95% CI excludes 0 with
  |delta| >= 0.01 for balanced accuracy, macro-F1, ECE or Brier, or |delta| >= 0.02 for a per-class recall/F1; or if the
  point |delta| of balanced accuracy or macro-F1 is >= 0.02 regardless of the CI.

## Run status
10 of 10 runs completed, 0 failed, no non-trivial warnings (one harmless read-only-array NumPy warning). Hard checks
passed: 3,064 out-of-fold predictions per arm, unique image ids, identical image sets and labels across arms.

## CONFIRMED RESULT: protocol contrast (pooled out-of-fold; B = image-level, A = patient-disjoint)
| Metric | A patient-disjoint | B image-level | B - A (95% paired patient-bootstrap CI) |
|---|---|---|---|
| Balanced accuracy | 0.9386 (0.9114-0.9608) | 0.9816 (0.9745-0.9880) | **+0.0430 (+0.0237, +0.0673)** |
| Macro-F1 | 0.9394 (0.9128-0.9612) | 0.9830 (0.9765-0.9888) | **+0.0436 (+0.0243, +0.0673)** |
| Accuracy | 0.9465 (0.9238-0.9657) | 0.9853 (0.9800-0.9903) | +0.0388 (+0.0219, +0.0590) |
| ECE (15 bins) | 0.0321 (0.0176-0.0522) | 0.0034 (0.0026-0.0084) | **-0.0287 (-0.0456, -0.0134)** |
| Brier (multiclass) | 0.0879 (0.0556-0.1265) | 0.0224 (0.0152-0.0303) | **-0.0655 (-0.1007, -0.0376)** |

Per class (B - A):
| Class | Recall | F1 |
|---|---|---|
| meningioma (708 images) | +0.0749 (+0.0290, +0.1351); A 0.881 -> B 0.956 | +0.0746 (+0.0407, +0.1173); A 0.895 -> B 0.969 |
| glioma (1,426) | +0.0316 (+0.0116, +0.0565); A 0.961 -> B 0.993 | +0.0291 (+0.0157, +0.0457); A 0.962 -> B 0.991 |
| pituitary (930) | +0.0226 (-0.0010, +0.0557); A 0.973 -> B 0.996 | +0.0270 (+0.0073, +0.0531); A 0.962 -> B 0.989 |

Calibration (`calibration_summary.csv`, `reliability_bins.csv`): arm A mean confidence 0.978 vs accuracy 0.947
(over-confident by 0.032); arm B 0.988 vs 0.985 (0.003). Mean NLL 0.208 (A) vs 0.045 (B). So the image-level protocol
makes the same architecture look both more accurate and much better calibrated. Confusion: meningioma is the main
casualty under patient-disjoint evaluation (42 predicted glioma, 42 pituitary of 708; B: 15 and 16).

Fold-level descriptive accuracies (`fold_metrics.csv`): A 0.924-0.969, B 0.980-0.989.

**Interpretation, limited to what was tested.** For this model and recipe, evaluating with patient overlap yields
clearly higher apparent accuracy and clearly better apparent calibration than patient-disjoint evaluation, with paired
patient-cluster intervals excluding 0 for every primary and calibration metric. The pre-specified rule returns
**GO_TO_MULTI_MODEL** (hits: balanced accuracy, macro-F1, ECE, Brier, meningioma and glioma recall/F1, pituitary F1;
magnitude flag also met for balanced accuracy and macro-F1). The data do not by themselves show *why* (the arms differ in
which images are in training for each test image, i.e. the whole fold composition); patient overlap is the designed
difference.

## Patient-leakage exposure in arm B (measured; `data/splits/split_validation.json`, `exposure_B_seed11.csv`)
- 3,051 of 3,064 images (99.58%) have at least one same-patient slice in training; the 13 without are 7 single-slice-patient
  images and 6 others. Arm A: 0 images (0%).
- Mean `frac_other_train` among multi-slice patients 0.795 (SD 0.119; 5th-95th percentile 0.61-1.0), i.e. compressed near 0.8
  by the construction of the split. `n_other_train`: median 13 (IQR 9-17, max 34).
- 224 of 233 patients have slices in more than one test fold in arm B (0 in arm A); in each fold 182-201 test patients also
  appear in training.

## EXPLORATORY RESULT: exposure-response within arm B (associational, not causal; `exposure_analysis.csv`, `fig6`)
- `frac_other_train` shows no clear association with correctness, confidence or per-image Brier (Spearman rho -0.002, 0.026,
  -0.026; all CIs include 0). Within the narrow observed range there is no evidence of a dose-response.
- `n_other_train` is positively associated with correctness (rho 0.082, CI 0.042-0.120), confidence (0.171) and lower Brier
  (-0.172), but it is almost the same variable as patient slice count (Spearman with `n_other_total` = 0.95). In an
  exploratory cluster-robust OLS including class and log patient slice count, `frac_other_train` is not significant for any
  outcome while log patient size is (e.g. confidence +0.0175, p=0.0001), so this association is not separable from patient
  size in this design.
- The 13 images with no same-patient training slice (9 patients) had arm-B accuracy 0.846 (bootstrap CI 0.667-1.0) versus
  0.986 for the others. That is directionally consistent with an overlap effect but is based on 13 images and is not
  interpretable inferentially. Arm A's accuracy on the same 13 images (0.923) is shown only descriptively; arm A is a
  protocol-level reference, **not** a zero-exposure counterfactual for these images.
- Context: mean slices per patient differs by class (meningioma 8.6, glioma 16.0, pituitary 15.0) and all MR-prefixed IDs
  are glioma (75% of glioma patients), so patient size and provenance are correlated with class.

## LIMITATIONS
1. **One architecture, one image-level split seed, one training seed.** The bootstrap quantifies patient-sampling
   uncertainty only; it does not include split-seed or training-seed variance. MPS runs are not bitwise reproducible.
2. **The contrast is a protocol contrast.** Arm A and B differ in the full fold composition; per-image exposure effects are
   not identified. No causal dose-response claim is made.
3. **Exposure range is compressed** near 0.8 by construction and entangled with patient size.
4. **Ceiling effects / epoch budget.** Both arms reach about 99.9% training accuracy at the a-priori 10-epoch budget;
   absolute levels may differ with other recipes. Hyper-parameters were not tuned per arm (by design).
5. **ECE bootstrap CI for arm B** (0.0034, CI 0.0026-0.0084) sits close to its lower bound because ECE is positively biased
   under resampling; read it as order-of-magnitude only.
6. **Source confounding unresolved:** MR-prefixed patient IDs are all glioma; no source-aware analysis is possible here.
   Patient 108931 (fifteen 256x256 images) was kept in all analyses; no exclusion sensitivity was run.
7. **No model-ranking evidence yet;** a large protocol effect makes ranking distortion plausible but does not show it.

## DECISION
`GO_TO_MULTI_MODEL` (rule applied mechanically from `results/sanity/decision.json`). No additional architectures or split
seeds were run in this sprint.

## Recommended next step
Expand the identical protocol to the planned architectures (MobileNetV2, EfficientNet-B0, DenseNet121) and to image-level
split seeds 12-15 and at least one more training seed, then test the stability of the model ranking (e.g. Kendall's tau
across protocols, bootstrap top-1 agreement). Keep the exposure analysis as a secondary descriptive output.

## Reproduce
```
python scripts/download_figshare.py --extract
python scripts/audit_dataset.py
python scripts/make_splits.py
EXPERIMENT_INPUT_COMMIT=e78a2b735352b6456a91a807897adc4623f3298e python scripts/run_sanity.py
python scripts/analyze_sanity.py && python scripts/build_run_manifest.py && python scripts/make_figures.py
python -m pytest -q tests
```
Figures: `reports/figures/fig1_primary_metrics.png` ... `fig6_exposure_response.png`. Model weights are not committed
(regenerate from the frozen protocol). Raw data and the preprocessed cache (`data/interim/`) are git-ignored.
