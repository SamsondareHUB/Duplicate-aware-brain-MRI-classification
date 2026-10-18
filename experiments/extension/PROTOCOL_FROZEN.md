# Extension experiments: frozen protocol (written BEFORE any new training outcome existed)

Branch `experiment/extension-seeds-exposure-v1`. Exploratory; does not modify PR #1/#2/#3/#4 results, historical splits, or the manuscript.
Identity: 233 PIDs = 233 individuals (dataset author); the original 233-ID evaluation remains primary; no 209-group retraining.

## Frozen seeds / folds / levels
| Item | Value |
|---|---|
| New split seeds (A: StratifiedGroupKFold over 233 PIDs; B: matched image-level, `make_image_level_folds`) | **101, 202, 303** (historical image-level seed was 11) |
| New training seeds on historical splits | **1, 2** (historical = 0) |
| Exposure fixed-test folds (original split A) | **2, 4** |
| Exposure training seeds | **0, 1** |
| Exposure levels | 0 / 25 / 50 / 75 / 100 % of eligible test patients (all feasible; counts in `exposure/design_summary.json`) |
| Models | ResNet50, DenseNet121 (ImageNet V1) |
| Recipe | identical to PR #1/#2 (`ext_runner.py` imports `CFG`, `augment`, `to_input`; batch 32, 10 epochs, AdamW 1e-4/1e-4, per-iteration cosine, CE, final epoch, no early stopping) |
Queue (172 runs, priority order in `ext_specs.build_queue`): exposure seed 0 (36) -> split 101 (20) -> exposure seed 1 (36) -> split 202, 303 (40) -> training seeds 1, 2 (40).

## Hypotheses and primary metrics (frozen)
Primary metric: balanced accuracy; secondary macro-F1, accuracy, per-class P/R/F1, ECE (15 bins), multiclass Brier.
* **H-split (confirmatory-descriptive):** across split seeds the image-level (B) advantage over patient-disjoint (A) is positive for both models. Ranking (ResNet50 vs DenseNet121) is reported per split; no significance claim from 3 seeds.
* **H-tseed (descriptive):** training-seed variation in the A and B model difference is reported; ranking consistency is counted, not tested.
* **H-expo (primary exposure contrast):** on the eval slices of exposed patients, SIB_L - CTRL_L balanced accuracy (and mean true-class probability, Brier) is > 0; the exposure-specific effect is the difference-in-differences against the unexposed eligible patients' slices (same models, same fixed test images). Dose-response: slope of that contrast over L in {25,50,75,100}. Family of primary contrasts: 2 models x 2 folds x {BA at L=100} + the pooled dose-response slope; Holm correction is applied to the p-values of this family; everything else is exploratory.
* Tumor-mask diagnostics are exploratory; masks are used only in analysis-time input variants, never in training.

## Design guards (verified by `tests/test_extension_design.py`)
Fixed test images never in any training set; donors removed from the test set in all arms; CLEAN/CTRL have zero train/test patient overlap; CTRL and SIB have identical size and class counts; levels are nested; control patients are class-matched with ID format (numeric vs `MR`) matched where possible (all matched in both folds). The SIB arms deliberately create train/test patient overlap: **an experimental intervention, never a leakage-free evaluation.**
Known design limits: the clean baseline trains on ~1,750 images (reservoir patients removed), fewer than the historical ~2,450; donor slices of a test patient may be near-adjacent to its eval slices (intended); control and exposed patients differ in identity/appearance.

Split hashes: `splits/split_validation_ext.json`; exposure design hash: `exposure/design_summary.json` (`design_sha256`).
