# 209-group retraining plan — PROPOSED, NOT EXECUTED

**Status: nothing in this document has been trained or run. No checkpoint, prediction or metric from a 209-group model exists.** All manifests under `data/splits/proposed_209/` are labeled **PROPOSED / NOT YET USED FOR TRAINING**. Historical splits and PR #1/#2 outputs are unchanged (SHA-256 re-checked: A `61bec5aa…3e49e`, B `e8b0442c…74f2`). Branch: PR #3 branch (open, not merged).

Label convention (repo, PR #1): file label **A = patient-disjoint**, **B = image-level seed 11**. New arms are named **A209** (stem-disjoint) and **B209** (matched image-level) to avoid confusion.

## 1. Current evidence and the unresolved identity question
- The suffix meaning of IDs like `MR024780B/D/E` is **unresolved** (Figshare README, API text and Cheng 2015/2016 do not say; see `PATIENT_ID_VALIDATION.md`). The 209-group reading is a **conservative hypothesis, not biological ground truth**.
- By raw PID the original folds are disjoint (0/233 span). By hypothetical stem, 16/209 stems span folds (734 images, all glioma, 23.96%).
- Completed sensitivity results (existing predictions only, no retraining): exposed-image accuracy shows no excess; dropping the 734 exposed images and stem-clustered bootstrap leave orderings and CIs essentially unchanged (≤0.0005). These are **not** a corrected estimate.
- Author question is drafted (report §), **not sent**.

## 2. 233-ID vs 209-group mapping
Rule (`normalize_pid`, single source in `scripts/validate_patient_ids.py`): `^MR\d{6}[A-Z]$` loses its one trailing capital letter; numeric IDs, bare `MR\d{6}` and everything else unchanged.

| | raw PID (233) | stem (209) |
|---|---|---|
| units | 233 | 209 (−24) |
| merged groups | – | 18 groups merge 42 raw PIDs (14×2, 2×3, 2×4); 811 images, all glioma |
| spanning folds (original folds) | 0 | 16 stems (734 images) |
| each unit single-class | yes | yes (verified) |

Full table: `data/splits/proposed_209/PROPOSED_pid_to_stem_mapping.csv` (columns: stem, n_images, label, fold_original, fold_minmove, merged_with_other_pid, moved_by_minmove). The 233-ID interpretation stays untouched as the reference (`data/splits/patient_disjoint_original.csv`).

## 3. Leakage-free split design
**Leakage-free 209-group splits can be constructed** (verified, `PROPOSED_split_validation_209.json`; 0 stems and 0 raw PIDs span folds, every image once, 5 folds, all stems single-class).

**Primary proposal `A209_minmove`** (deterministic, no RNG): all 193 non-spanning stems keep their original fold. The 16 spanning stems (18 raw PIDs; 332 images = 10.8% of data, all glioma) are each moved whole to a fold they already occupy, largest first, choosing the fold that minimises squared deviation of fold×class counts from the original matrix (tie → lowest fold). Moves are listed in `PROPOSED_minmove_moved_pids.csv`.

| fold | original size (men/gli/pit) | A209_minmove size (men/gli/pit) |
|---|---|---|
| 1 | 542 (112/245/185) | 514 (112/217/185) |
| 2 | 679 (167/338/174) | 708 (167/367/174) |
| 3 | 572 (139/231/202) | 552 (139/211/202) |
| 4 | 628 (124/325/179) | 625 (124/322/179) |
| 5 | 643 (166/287/190) | 665 (166/309/190) |

Meningioma and pituitary counts are identical to the original; only glioma shifts (max ±30 per fold).

**Alternative `A209_sgkf_alternative`** (comparison only): `StratifiedGroupKFold(5, shuffle, random_state=11)` over stems → near-perfectly balanced folds (≈613 each) but the fold assignment is essentially unrelated to the original (only 14% of images share a fold number, labels being arbitrary), so it is a different experiment, not a minimal correction. Not proposed as primary.

**Matched image-level arm `B209_imagelevel_seed11`**: the *same* `make_image_level_folds(seed=11)` function as PR #1, fed the A209 fold×class matrix (so fold×class counts equal A209 exactly; verified, deterministic).

Manifest SHA-256 (proposed; see JSON): A209_minmove `6b8f98cc…6313`, B209 `8fa98fd8…1c51`.

Regenerate: `python scripts/make_splits_209.py` (writes only to `data/splits/proposed_209/`).

## 4. Run matrix (proposed; none run)
Completed under PR #1/#2: 4 models × arms A,B × 5 folds, train seed 0 only (40 runs; `results/audit/run_inventory.csv`). There are **no additional seeds**, so the new matrix also uses train seed 0.

| Model | Arm | Folds | Seed | Split file | Output |
|---|---|---|---|---|---|
| resnet50, mobilenet_v2, efficientnet_b0, densenet121 | A209 (primary) | 1–5 | 0 | `PROPOSED_A209_minmove.csv` | `results/retrain_209/runs/{model}_A209_fold{k}_{predictions.csv,run.json}` |
| same four | B209 (matched control) | 1–5 | 0 | `PROPOSED_B209_imagelevel_seed11.csv` | same pattern with `B209` |

- **Total: 40 runs** (20 primary A209 + 20 B209). Minimum if only the stem-disjoint arm is wanted: **20**. Every fold's training set changes (moved images touch all five folds), so no existing run can be reused for A209. Machine-readable: `data/splits/proposed_209/run_matrix_209.csv`.
- Config identical to PR #1: `CFG` in `scripts/run_sanity.py` (224², batch 32, 10 epochs, AdamW lr 1e-4 wd 1e-4, per-iteration cosine to 0, plain CE, hflip + ±10° rotation, ImageNet-V1 weights, final epoch, no early stopping, train seed 0 reset per fold, eval batch 64). No hyperparameter tuning.
- Checkpoints: the existing pipeline saves **predictions + run JSON, not model weights**; the same applies here.

## 5. Preserved vs changed
| Preserved | Changed / cannot stay identical |
|---|---|
| Images, preprocessing cache, 4 architectures, CFG, train seed 0, metrics, 5-fold OOF design, bootstrap (5,000 reps, seed 2026, class-stratified cluster) | Fold membership of 332 glioma images (10.8%) → every fold's train and test set differs from PR #1/#2; therefore A209 is **not** a re-run of A on identical test folds |
| B-arm construction function, seed 11 | B209 fold assignment differs from B (it follows the A209 margins); old B ≠ B209 |
| Output schema (+ `stem` column) | Bootstrap cluster = stem (209 units) instead of raw PID (233) for 209 analyses |
| MPS non-bitwise determinism caveat | Not bitwise-reproducing old runs; run-to-run noise with the same seed is unmeasured |

Also unchanged and unresolved: the MR-vs-numeric ID format confound (MR glioma ≈99% vs numeric-ID glioma 76–83%).

## 6. Evaluation and statistical plan
Re-use `analyze_multimodel.py` logic (metrics: accuracy, BA, macro-F1, per-class P/R/F1, confusion matrices, ECE, Brier, reliability bins; 5,000-rep paired bootstrap, seed 2026; ranking tables, Kendall τ, top-rank frequencies) via a copy parameterised by split family and stem clustering — **to be written after approval** (not yet implemented; no run output exists to test it on). Plus split-integrity verification (`validate()` in `run_retrain_209.py`, `tests/test_splits_209.py`).

Comparisons:
1. **Primary after retraining:** A209 vs B209 (same 3,064 images, identical fold×class margins, stem-clustered paired bootstrap) = protocol effect under the conservative identity.
2. **Orig-233 vs 209 (A vs A209):** same images, different train/test partitions. Report as a *robustness comparison*, not a matched paired experiment, with a secondary subset analysis on the 2,732 images whose test fold did not change. Differences include run noise (single seed) and cannot be attributed to leakage alone.
3. Ranking comparison across four models in A, A209, B, B209; ResNet50−DenseNet121 contrast already includes 0 in A and B, so expect non-resolution unless A209 differs; no claim pre-registered.
4. Clearly separate in the paper: *completed* (prediction-only sensitivity, `results/audit/patient_id/`) vs *requires retraining* (anything A209/B209).

## 7. Compute / time (Apple-silicon MPS, from PR #1/#2 run JSONs; no paid compute, no new downloads)
| Model | s / fold-run | 5 folds |
|---|---|---|
| ResNet50 | ~377 | 31 min |
| MobileNetV2 | ~169 | 14 min |
| EfficientNet-B0 | ~241 | 20 min |
| DenseNet121 | ~390 | 33 min |
| **One arm (20 runs)** | | **≈1.6 h** |
| **Both arms (40 runs)** | | **≈3.3 h** training + runner's 20-min-max cooling/RAM checks between folds (the PR #2 runner inserts them; wall clock may be ~1.5–2×, i.e. up to ~6 h) |

Resources: one MPS GPU job at a time (sequential), a few GiB unified memory (planning figures, not measured), no extra disk beyond ~tens of MB of CSVs. Run only with the Mac otherwise idle; do not shut down/restart it.

## 8. Author-response scenarios
- **A. Suffixes = different patients:** keep the 233-ID evaluation (PR #1/#2) as primary; A209/B209 is an optional conservative robustness analysis (A209 only, 20 runs, or none if the existing sensitivity suffices).
- **B. Suffixes = same patient (repeated scans):** A209/B209 becomes the primary patient-disjoint evaluation; the 233-ID results are reported as the original-fold reference showing partial leakage. Run all 40 (or 20 if B comparison is judged adequate by the old B + sensitivity). Methods/Results must be rewritten around 209 units.
- **C. No author response — PENDING TEAM DECISION. I do not choose.** Sammie's two proposed approaches are not in the repository files I can read, and I do not want to guess their wording; below are the two options that the scenario logically allows — **please confirm which match Sammie's messages**:
  - **C1 (233-primary + 209 as sensitivity):** conservative with respect to claims already made; costs 0–20 runs; paper states the identity ambiguity and shows 209 robustness. Risk: a reviewer may say the primary analysis may leak if suffix = same person.
  - **C2 (209-primary / worst-case-safe):** treat the ambiguity as a potential leak and report the 209 stem-disjoint analysis as primary; costs 20–40 runs and rewrites Methods/Results; may slightly understate patient count (209 vs 233) if suffixes are in fact different patients. Safest against leakage criticism.
  - Either way both numbers must appear and the "233 patients" claim must carry the disclosure.
- **No retraining starts under any scenario until Sammie/team approves in writing.**

## 9. Exact commands (RUN ONLY AFTER APPROVAL)
```bash
# 0. safe, already runnable: verify splits and matrix (no torch import, no training)
python scripts/make_splits_209.py && python scripts/run_retrain_209.py --dry-run && python -m pytest -q tests
# 1. one-fold smoke test of the new runner (needs approval; ~3 min, writes results/retrain_209/ only)
python scripts/run_retrain_209.py --train --approved-by "<name, date>" --model mobilenet_v2 --arm A209 --max-new-runs 1
# 2. full primary arm, one model at a time (resumable; skips existing outputs)
for m in mobilenet_v2 efficientnet_b0 resnet50 densenet121; do
  python scripts/run_retrain_209.py --train --approved-by "<name, date>" --model $m --arm A209; done
# 3. optional matched control
for m in mobilenet_v2 efficientnet_b0 resnet50 densenet121; do
  python scripts/run_retrain_209.py --train --approved-by "<name, date>" --model $m --arm B209; done
```
Known caveat: the training branch of `run_retrain_209.py` mirrors `run_multimodel.py` but **has not been executed**; step 1 is its first real test, and results from step 1 should be discarded or kept only if it passes review. The stem-clustered analysis script for the new outputs is also still to be written (§6).

## 10. Decisions needed from Sammie
1. Scenario C choice (C1 vs C2) — and confirm the author email may be sent (not yet sent).
2. Approve A209 `minmove` as the primary corrected split (vs StratifiedGroupKFold alternative).
3. Approve 20 runs (A209) or 40 (A209+B209); approve the single-seed limitation or ask for extra seeds (+20 runs per added seed per arm).
4. Approve the hypothetical normalisation rule as the 209 definition.
