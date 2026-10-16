# Results — verification notes (support for Sammie; does NOT replace `paper/RESULTS.md`)

All numbers independently recomputed from the committed OOF prediction files (`python scripts/reproduce_from_predictions.py`, 200/200 checks pass; max |diff| < 1e-9 vs tables). Tables: `results/multimodel/{metrics_summary,protocol_differences,pairwise_differences,ranking_summary,top_ranked_frequencies}.csv`. A = patient-disjoint, B = image-level seed 11; Δ = B − A. CI = 95% patient-cluster bootstrap.

## Placeholders still open in `paper/RESULTS.md` (12) and replacement wording
1. `[0]` exact duplicate groups → **0** (hash-based; 3 hash variants).
2. `[briefly state the near-duplicate finding…]` → "A two-stage screen (thumbnail NCC ≥ 0.90 or perceptual-hash distance ≤ 10, then full-resolution NCC/SSIM) retrieved 33,432 candidate pairs; none reached NCC 0.95 or SSIM 0.90, and the highest cross-patient values (NCC 0.932, SSIM 0.855) did not separate from within-patient values. No near-duplicate threshold was frozen and the screen has limited recall; we therefore report that cross-patient near-duplication was not found, not that it is absent."
3. `[insert metric names… from PR #1]` ResNet50 ECE 0.0321 (A) vs 0.0034 (B), Δ −0.0287 [−0.0456, −0.0134]; Brier 0.0879 vs 0.0224, Δ −0.0655 [−0.1007, −0.0376].
4. `[summarize … four architectures]` see Table 1/2 below.
5. `Table [X]` → Table 1 (pooled metrics) and Table 2 (protocol differences).
6. `[model]` (image-level highest) → **ResNet50** (BA 0.9816; macro-F1 0.9830); `[model]` (patient-disjoint highest) → **DenseNet121** (BA 0.9456; macro-F1 0.9467).
7. `[report estimate, interval…]` → ResNet50−DenseNet121 BA under patient-disjoint evaluation −0.0069 (−0.0160, 0.0015); macro-F1 −0.0073 (−0.0161, 0.0012). Under image-level: BA +0.0043 (−0.0015, 0.0104); macro-F1 +0.0031 (−0.0019, 0.0086). **Both intervals include 0 in both protocols** (this applies to image-level too — the draft only mentions the patient-disjoint contrast).
8. `[name the calibration metric…]`, `[describe the direction…]` → "15-bin top-label ECE and multiclass Brier; both were lower (better) under image-level evaluation for all four models: ECE Δ −0.0287 (ResNet50), −0.0290 (MobileNetV2), −0.0244 (EfficientNet-B0), −0.0189 (DenseNet121), all intervals excluding 0; Brier Δ −0.0655, −0.0660, −0.0614, −0.0498."
9. `[insert findings from the error analysis]`, `[insert relevant per-class…]` → meningioma is the weakest class under patient-disjoint evaluation: recall 0.8814 / 0.8460 / 0.8460 / 0.8927 (ResNet50 / MobileNetV2 / EfficientNet-B0 / DenseNet121) vs 0.9562 / 0.9322 / 0.9223 / 0.9463 under image-level (see `per_class_metrics.csv`, `confusion_matrices.csv`).

## Table 1 — pooled out-of-fold (n = 3,064 per cell)
| Model | Protocol | Balanced acc. (95% CI) | Macro-F1 | Accuracy | ECE-15 | Brier |
|---|---|---|---|---|---|---|
| ResNet50 | A | 0.9386 (0.9114–0.9608) | 0.9394 | 0.9465 | 0.0321 | 0.0879 |
| ResNet50 | B | 0.9816 (0.9745–0.9880) | 0.9830 | 0.9853 | 0.0034 | 0.0224 |
| MobileNetV2 | A | 0.9246 (0.8962–0.9475) | 0.9266 | 0.9360 | 0.0383 | 0.1047 |
| MobileNetV2 | B | 0.9708 (0.9618–0.9788) | 0.9733 | 0.9765 | 0.0093 | 0.0387 |
| EfficientNet-B0 | A | 0.9252 (0.8969–0.9490) | 0.9280 | 0.9373 | 0.0320 | 0.0993 |
| EfficientNet-B0 | B | 0.9672 (0.9561–0.9766) | 0.9697 | 0.9736 | 0.0076 | 0.0379 |
| DenseNet121 | A | 0.9456 (0.9201–0.9662) | 0.9467 | 0.9533 | 0.0228 | 0.0755 |
| DenseNet121 | B | 0.9774 (0.9681–0.9852) | 0.9798 | 0.9824 | 0.0040 | 0.0257 |

## Table 2 — protocol difference B − A, balanced accuracy (95% CI)
ResNet50 +0.0430 (0.0237, 0.0673); MobileNetV2 +0.0462 (0.0280, 0.0685); EfficientNet-B0 +0.0420 (0.0238, 0.0631); DenseNet121 +0.0318 (0.0143, 0.0528).

## Supported wording
* "Image-level evaluation yielded higher balanced accuracy than patient-disjoint evaluation for all four architectures (+0.032 to +0.046; every interval excluded 0)."
* "The nominal best architecture differed between protocols (DenseNet121 vs ResNet50), but the pairwise intervals between these two included zero under both protocols; we therefore do not conclude a change in the best architecture."
* "Bootstrap top-ranked frequencies (e.g. 0.94 for DenseNet121 under patient-disjoint, 0.92 for ResNet50 under image-level, balanced accuracy) describe resampling of patients for fixed trained models and are not posterior probabilities."
* Kendall τ-b between protocol orderings = 0.333 for both balanced accuracy and macro-F1.

## Wording to remove or soften
* "Exact and near-duplicate leakage were not identified as the main source of inflation" → the design does not decompose sources; say "exact duplicates were absent and no cross-patient near-duplicates were found by the screen; the evaluated contrast is patient overlap".
* "the results indicate confidence calibration depended on the protocol" is fine; avoid "models were better calibrated" without "apparent".
* Do not write "patient overlap causes", "leakage reverses the ranking", "ResNet50 is best under image-level evaluation" (unqualified), or anything implying clinical readiness.
* The trailing sentence in "Dataset and leakage audit" should add that `cvind` was inferred and that patient-ID suffix siblings are unresolved.

## Optional additional results already computed (post hoc; label as exploratory)
Patient-weighted balanced accuracy orderings are unchanged; interaction (ResNet50−DenseNet121) B−A = +0.0112 [0.0005, 0.0220] (borderline; 6/18 interaction contrasts exclude 0, no multiplicity correction); per-fold ResNet50−DenseNet121 signs are inconsistent (A: + − − + −; B: + − 0 + +). Source: `results/audit/ranking_extras*.json/csv`.
