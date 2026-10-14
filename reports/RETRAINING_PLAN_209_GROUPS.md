# 209-group retraining plan — PROPOSED, NOT EXECUTED (decision policy finalised by Sammie)

**Status: nothing in this document has been trained or run. No checkpoint, prediction or metric from a 209-group model exists.** All manifests under `data/splits/proposed_209/` are labeled **PROPOSED / NOT YET USED FOR TRAINING**. Historical splits and PR #1/#2 outputs are unchanged (SHA-256 re-checked: A `61bec5aa…3e49e`, B `e8b0442c…74f2`). Branch: PR #3 branch (open, not merged). The split-selection algorithm (`minmove`, §3) still awaits Sammie's final approval; the manifests are unchanged and regenerate byte-identically (SHA-256 `6b8f98cc…6313` A209, `8fa98fd8…1c51` B209).

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

## 4. Final author-response policy (confirmed by Sammie) and run matrices
| Author outcome | Primary analysis | Training after explicit approval | Matrix file |
|---|---|---|---|
| Suffix variants = **different patients** | original 233-ID (PR #1/#2) | **20 runs**: A209 only (4 models × 5 folds), as a patient-disjoint sensitivity analysis | `run_matrix_209_distinct_20.csv` |
| Suffix variants = **repeated scans of the same person** | **209-group** | **40 runs**: A209 + matched B209 (4 × 5 × 2) | `run_matrix_209_repeated_or_noresponse_40.csv` |
| **No response** after a *reasonable, team-approved* waiting period | **conservative 209-group**; 233-ID becomes sensitivity | same **40 runs** | same 40-run file |

- The waiting deadline is **not chosen here**; it is a team decision. Nothing trains under any scenario without explicit approval; the runner enforces this (`--approved-by` + `--scenario`; scenario `distinct` refuses B209).
- Both matrices: models resnet50, mobilenet_v2, efficientnet_b0, densenet121; folds 1–5; **original train seed 0** (the only seed used in PR #1/#2). This is **not a multi-seed replication**; extra seeds need separate team approval (+20 runs per added seed per arm).
- Verified by tests: 20 and 40 rows, unique run IDs, seed 0 everywhere, estimated 1.63 h / 3.27 h of training.
- Outputs: `results/retrain_209/runs/{model}_{arm}_fold{k}_{predictions.csv,run.json}` (+ `environment_{model}_{arm}.json`). Like PR #1/#2 the pipeline stores predictions and run records, not weights.

## 5. Protocol parity with PR #1 / PR #2 (code-level only)
`tests/test_runner_parity.py` (42 checks) verifies that each training/evaluation statement exists **verbatim in both** `run_multimodel.py` (PR #2) and `run_retrain_209.py`, and the ResNet50 build lines verbatim in `run_sanity.py` (PR #1):

| Element | Status |
|---|---|
| Preprocessing (`load_all`, cached `pre224.npy`, min-max → 224², 3-ch ImageNet mean/std in `to_input`) | **imported** from PR #1 modules, not re-implemented; runner refuses to regenerate the cache silently |
| Augmentation (hflip p=0.5, rotation ±10° bilinear fill 0, generator seed 0+1) | `augment` imported; generator lines identical |
| Architectures + ImageNet-V1 weights | MobileNetV2/EfficientNet-B0/DenseNet121 via imported `make_model`/`WEIGHTS`; ResNet50 lines identical to `run_sanity.py`; head shapes checked (weights not downloaded in test) |
| Batch 32, 10 epochs, AdamW lr 1e-4 wd 1e-4, per-iteration cosine (T_max = total steps, eta_min 0), plain CE, final epoch, no early stopping | `CFG` imported from `run_sanity.py`; frozen values asserted; statements identical |
| Train seed 0 reset per fold (torch, numpy, data and aug generators) | identical |
| Evaluation (eval batch 64, `model.eval()`, `no_grad`, float64 softmax from logits) | identical |

Known, documented differences (none intended to change numerics, but **not provable without a GPU run**): (i) ResNet50 under the new runner follows the PR #2 loop (per-epoch `torch.mps.empty_cache()`, tensor `del`s) whereas PR #1's ResNet50 script did not; (ii) fold membership (332 images moved) so train/test sets differ; (iii) MPS is not bitwise deterministic, so even identical inputs would not reproduce old numbers exactly; (iv) the PR #2 per-step memory print was dropped (logging only).

**Not verified (needs real GPU, not done):** that the real training function runs end-to-end, MPS memory behaviour, actual runtimes, and that `serial_209.py` launches the child correctly on this Mac.

## 6. Prediction schema
Per-fold prediction CSV columns (`SCHEMA` in `run_retrain_209.py`), checked by tests against the real PR #1/#2 files:
- **PR #2 common mandatory fields (exactly the 15 columns of the PR #2 files):** model, arm, image_id, patient_id, protocol, test_fold, true_class, true_class_name, prob_0–2, pred_class, pred_class_name, confidence, experiment_input_commit.
- **PR #1 ResNet50-specific columns, now written for all four models:** logit_0–2 and the exposure columns (n_other_total, n_other_train, frac_other_train, single_slice_patient, has_same_patient_train_slice; computed by raw patient ID as in PR #1). PR #2 files for the other three models do not contain these.
- **209 additions:** `stem`, `split_sha256`.
- Arm/protocol labels are new (`A209`/`B209`, `patient_disjoint_209`/`image_level_209_seed11`) so they cannot be confused with historical `A`/`B`.
- Run JSON = PR #2 fields + `predictions_sha256`, `scenario`, `approved_by`, `cfg`.

## 7. Resume safety and Mac protections
- A fold is skipped **only if both** its prediction CSV and run JSON exist and independently pass checks (schema, image set = split fold, arm/fold/model/protocol, patient/stem/label match to the split, probabilities, argmax/confidence/class names, finite logits; JSON: model/arm/fold/device `mps`/split SHA/epoch count/seed 0/n_train/n_test/hash of the CSV). One file alone, leftover `.tmp`, corrupt or mismatching files raise `RunIntegrityError`; nothing is skipped or overwritten. `--quarantine-partial MODEL ARM FOLD` moves the files aside (never deletes) so the fold can be rerun.
- Publishing a fold ignores SIGINT/SIGTERM during the two atomic file replacements (CSV first, then JSON containing the CSV hash).
- `scripts/serial_209.py` ports the PR #2 protocol: single lock, serial execution, one fold per child process under `caffeinate` + `nice`, preflight (RAM ≥ 8 GiB, memory-pressure free ≥ 35%, disk, thermal, responsiveness; bounded wait, no automatic retry), 60 s cooling between folds, 15 s active monitoring that sends **SIGINT only to its own child session** on unsafe state, completed folds retained. It never signals other processes, never restarts/shuts down the Mac, and does **not** push to git (PR #2's per-fold auto-push was dropped; publish manually after review).
- Verified with synthetic fixtures: resume skips completed folds without retraining, nine corruption/partial modes refused, unsafe-host stop kills only the test child, preflight timeout blocks launch, synthetic runs can never pass as real `mps` runs. **Not verified on real hardware.**

## 8. Analysis pipeline (`scripts/analyze_209.py`, validated on synthetic fixtures only)
- Frozen metric/bootstrap code from `src/leakage` is reused unchanged: accuracy, balanced accuracy, macro-F1, per-class P/R/F1 with CIs, confusion matrices, 15-bin ECE, Brier, reliability bins and diagrams, pairwise model differences, rankings, Kendall τ, top-rank frequencies; **5,000 class-stratified paired cluster replicates, seed 2026**, one draw matrix shared by all models and arms.
- **Documented change:** bootstrap cluster = **209 stems** (not 233 raw IDs). Historical predictions that are re-analysed in the robustness table are re-clustered by stem too.
- **A209 vs B209 = matched:** same 3,064 images, identical fold×class counts, same stem draws (`protocol_differences_209.csv`, labeled MATCHED).
- **Original 233-ID (A) vs A209 = robustness only** (`robustness_original233_vs_A209.csv`, labeled ROBUSTNESS ONLY): changed fold assignment, single seed, differences include run noise; secondary subset on the 2,732 images whose test fold is unchanged (train sets still differ). **A209 is never compared with historical B**, and the pipeline has no code path that does.
- Already completed (no retraining): prediction-only stem-clustered sensitivity (`results/audit/patient_id/`). Everything above on A209/B209 requires retraining and does not exist.
- Run: `python scripts/analyze_209.py --scenario distinct|repeated|no_response` (works only once all required folds exist and pass integrity checks).

## 9. Compute / time
| Model | s per fold-run | 5 folds |
|---|---|---|
| ResNet50 | ~377 | 31 min |
| MobileNetV2 | ~169 | 14 min |
| EfficientNet-B0 | ~241 | 20 min |
| DenseNet121 | ~390 | 33 min |
| **20 runs (A209 only)** | | **≈1.6 h** training |
| **40 runs (A209 + B209)** | | **≈3.3 h** training |

Add the 60 s cooling pauses and preflight waits (≈ +5–10%); PR #2's 20-minute guard could stretch wall clock under memory pressure. Estimates come from past MPS runs and are unverified for this runner. No paid compute; one local MPS job at a time; the Mac should otherwise be idle.

## 10. Commands (RUN ONLY AFTER EXPLICIT APPROVAL; none executed)
```bash
# Safe checks (no training)
python scripts/make_splits_209.py && python scripts/run_retrain_209.py --dry-run && python -m pytest -q tests
# Scenario "distinct" (20 runs)  -- or replace --scenario/arms for the 40-run scenarios
for m in mobilenet_v2 efficientnet_b0 resnet50 densenet121; do
  python scripts/serial_209.py --model $m --arm A209 --scenario distinct --approved-by "<name, date>"; done
# Scenarios "repeated" or "no_response" (40 runs): A209 then B209
for arm in A209 B209; do for m in mobilenet_v2 efficientnet_b0 resnet50 densenet121; do
  python scripts/serial_209.py --model $m --arm $arm --scenario repeated --approved-by "<name, date>"; done; done   # or --scenario no_response
python scripts/analyze_209.py --scenario repeated        # after all 40 runs verify
```
Recommended first step if approved: a one-fold run (`python scripts/run_retrain_209.py --train --approved-by "…" --scenario … --model mobilenet_v2 --arm A209 --max-new-runs 1`) to validate the real training path before committing hours; its output is a real result of the proposed experiment and should be kept only if it passes review.

## 11. Remaining blockers / decisions for Sammie
1. Approve (or change) the split-selection algorithm `A209_minmove` (§3) — the code is frozen to it until then.
2. Approve sending the drafted author question (not sent) and fix the waiting period for "no response" (not chosen by us).
3. Explicit written approval to start training for the applicable scenario (20 or 40 runs); recommended first as one-fold real-path validation.
4. Approve the single-seed limitation (no extra seeds without approval).
5. Confirm bootstrap cluster change (233 raw IDs → 209 stems) for A209/B209 analyses.
