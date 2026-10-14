# Final independent scientific audit

Reviewed repository: `SamsondareHUB/Duplicate-aware-brain-MRI-classification`, `main` @ `14e7f75e3d516fc89d5b758442a611e86707b366` (PR #1, #2 merged; METHODS/RESULTS skeleton commits on top).
Audit branch: `melody/final-scientific-assurance-v1`. Date: 2026-10-14. Reviewer: Melody (assisted by Claude).
Dispositions: PASS · FIX · NEEDS_NEW_EVIDENCE · REFORMULATE · UNSUPPORTED. Status labels: CONFIRMED · STRONGLY SUPPORTED · THRESHOLD-DEPENDENT · UNRESOLVED.

**What was and was not re-executed.** Re-executed from committed artifacts only: everything under §3–§8 (`python scripts/reproduce_from_predictions.py`: **200/200 checks PASS**; `pytest`: **25 passed**; `scripts/verify_multimodel.py` passes).
**NOT REEXECUTED** (needs raw MRI/GPU, deliberately not downloaded): Figshare MD5 check, pixel hashing, near-duplicate screening, preprocessing, all 40 training jobs, `cvind.mat` permutation test. For those, only internal consistency of committed manifests was checked.

## 1. Repository and commit
See header. Existing outputs and Sammie's `paper/METHODS.md` / `paper/RESULTS.md` are untouched; new files only (plus README rewrite).

## 2. Dataset integrity (from committed manifests)
| Item | Status | Basis |
|---|---|---|
| Source = official Figshare 1512427 v8, CC BY 4.0, 6 files MD5 match | CONFIRMED (as recorded in `provenance.json`; MD5s NOT REEXECUTED) | `reports/manifests/provenance.json` `all_md5_match: true` |
| 3,064 images, 233 patients, classes 708/1426/930 | CONFIRMED | recomputed from `image_manifest.csv` and split files |
| Image shapes 512² (3049) + 256² (15, one patient 108931) | CONFIRMED (manifest) | `audit_summary.json` |
| Each patient has exactly one class | CONFIRMED | recomputed |
| Original folds = `cvind` file-order mapping; 0/233 patients span folds | STRONGLY SUPPORTED (not author-confirmed) | split A == manifest folds; permutation null (mean 223.4) not re-run |
| Image-level arm B shares patients across folds | CONFIRMED | `reproduce_from_predictions.py` counts patients spanning ≥2 folds in B (`results/audit/reproduction_checks.json`) |
| `MR…` IDs ↔ glioma (67 IDs, 1,221 images; all glioma) | UNRESOLVED provenance/source-confounding | recomputed; association only |
| **NEW** (superseded in detail by `reports/PATIENT_ID_VALIDATION.md`, CASE C – unresolved; sensitivity analyses there show no change to conclusions): patient-ID **stems** — 18 stems carry 2–4 suffixed IDs (e.g. `MR024780B/D/E/G`; 42 IDs, 811 images); **16 of those 18 stems have IDs in different "patient-disjoint" folds (734 images)** | **UNRESOLVED**. If suffixed IDs are the same individual (different studies), the original folds are not strictly subject-disjoint for ~24% of images and the dataset's "233 patients" would overcount individuals (209 stems). The dataset documentation does not say. | `scripts/pid_suffix_sensitivity.py`, `results/audit/pid_suffix_sensitivity.json` |
| Sensitivity of A-accuracy to sibling exposure | exploratory; ceiling-limited | MR-glioma accuracy is 98.5–99.2% (exposed) vs 99.2–99.8% (not); stem-cluster CIs of the difference include 0 for all 4 models. Not evidence of leakage, but also cannot rule it out (floor/ceiling) |

## 3. Duplicate-audit verification
* Exact duplicates: 0 duplicated `raw_array_hash`/`canonical_image_hash` among 3,064 manifest rows (recomputed) — CONFIRMED absent *conditional on manifest hashes being correct (hash computation NOT REEXECUTED)*.
* Near-duplicates: Stage-A retrieval (thumb NCC ≥ 0.90 or pHash ≤ 10) yields 33,432 candidates; verified file shapes and the 19 cross-patient NCC≥0.90 pairs (max NCC 0.932, SSIM 0.855; 12 same-class, 12 cross-fold) match the report. Cross-patient near-identical duplication: **NOT SUPPORTED by the screen — THRESHOLD-DEPENDENT / recall-limited** (pairs below the retrieval cut, shifts, crops, rescales not examined; threshold not frozen). The top-similarity cross-patient pairs (e.g. files 1878/2229, 1877/2228) deserve visual inspection (not done).
* Same-patient neighbouring slices: CONFIRMED dependent observations, not duplicates.
* Wording risk: "Cross-patient duplication not found" must always carry "within the screened candidate set; threshold not frozen".

## 4. Split integrity
SHA-256 of both split files match `split_validation.json` and every run record (40/40). A: 3,064 rows, folds 1–5, sizes 542/679/572/628/643, 0 patients across folds. B: same fold sizes and fold×class table as A (verified), 233 patients, many span folds. **Protocol identity: A = patient-disjoint, B = image-level (seed 11)**; `EXPERIMENT_SPEC_MULTI_MODEL.md` reverses the letters → FIX (add a header note; do not renumber results).

## 5. Run inventory (40 runs = 10 ResNet50 + 30 new)
| Model | Protocol | Fold records | Train seed | Epochs | Weights | Device | Input commit |
|---|---|---|---|---|---|---|---|
| resnet50 | A,B | 5+5 | 0 | 10 | IMAGENET1K_V1 | mps | e78a2b7 |
| mobilenet_v2, efficientnet_b0, densenet121 | A,B | 5+5 each | 0 | 10 | IMAGENET1K_V1 | mps | 919d6f0 |
Each of 8 model×protocol sets: exactly 3,064 unique OOF rows; image IDs, patient IDs, fold and label agree with the split file; probabilities finite, in [0,1], row sums deviate ≤1e-6; `pred_class == argmax`; per-fold prediction files equal the consolidated OOF files (to 1e-12; CSV round-trip ~1e-23). Summed new-run runtime 133.3 min (matches report); ResNet50 62.8 min. Caveat: ResNet50 and the three new models were trained by different scripts at different commits (`run_sanity.py` vs `run_multimodel.py`/`serial_model.py`); recipes are documented identical but code-equivalence was not byte-diffed here (preprocess/splits modules unchanged between the two commits: verified via `git diff`). Per-run logits are stored only for ResNet50.

## 6. Metric reproduction
Independent implementation (scikit-learn for accuracy/BA/macro-F1/PRF/confusion; own ECE-15 and Brier) vs committed tables: max |diff| < 1e-9 for all 40 headline point metrics, per-class metrics, confusion matrices, protocol differences; all 40 cells of the headline table in `reports/MULTIMODEL_RESULTS.md` match at 4 dp. **No discrepancies found.** Disposition: PASS.

## 7. Calibration verification
ECE (15 equal-width bins, top-label) and multiclass Brier (range 0–2) recomputed from probabilities: PASS. Interpretation FIX: ECE of ≈0.003–0.009 for 97–98%-accurate models is bounded by error rate and has resampling bias near zero (report's own limitation 4); "better calibration under B" mostly reflects higher accuracy plus the same-patient neighbours seen in training. Do not describe it as the model "being calibrated". No temperature scaling/recalibration was done.

## 8. Bootstrap and ranking verification
* Stored draws: 5,000×233 integer matrix; patients are the unit; per-class multiplicity totals equal per-class patient counts in every replicate (class stratification correct); regenerates exactly from seed 2026; same draws used for all 8 sets, so A/B and cross-model contrasts are paired.
* Metrics recomputed independently on the stored draws: percentile CIs, protocol-difference CIs, all 24 pairwise contrasts, top-ranked frequencies (ties split equally; no exact ties occur), Kendall τ-b: all match (<1e-9). A fresh 20,000-rep bootstrap with a new seed reproduces published CIs within 0.003 (`results/audit/fresh_bootstrap_ci_check.csv`). PASS.
* **Ranking reliability (claim boundary).** ResNet50−DenseNet121 balanced accuracy: A −0.0069 [−0.0160, 0.0015]; B +0.0043 [−0.0015, 0.0104]; macro-F1 A −0.0073 [−0.0161, 0.0012]; B +0.0031 [−0.0019, 0.0086]. **All four intervals include 0** (also with the fresh bootstrap). → no statistically established best architecture in either protocol, and no established reversal.
* Top-ranked frequencies (A: DenseNet121 0.940 BA / 0.953 F1; B: ResNet50 0.923 / 0.884) are bootstrap selection frequencies, **not posterior probabilities**, and exclude split/training randomness.
* Kendall τ-b between protocols = 0.333 (point). Bootstrap τ: mean 0.53, 95% interval [0.33, 1.0], τ>0 in 98.6% of replicates, identical full ordering in 6.9% (BA) — i.e. the *top-two swap* is the only unstable feature; the bottom pair (EfficientNet-B0/MobileNetV2) also swaps (A: MobileNetV2 last; B: EfficientNet-B0 last) but their difference is ≈0 under both protocols.
* Metric sensitivity: top model is DenseNet121 (A) / ResNet50 (B) for BA, macro-F1, accuracy, Brier and NLL alike; patient-weighted BA gives the same orderings (ResNet50−DenseNet121 patient-weighted: A −0.011 [−0.025, 0.002]; B +0.007 [−0.004, 0.018]). The point-estimate ranking is therefore not metric-fragile, but equally unresolved.
* Exploratory interaction (post hoc): (ResNet50−DenseNet121)_B − (…)_A = +0.0112 [0.0005, 0.0220] BA; +0.0103 [0.0001, 0.0210] F1. 6 of 18 interaction contrasts exclude 0 (all involve DenseNet121 vs ResNet50/MobileNetV2), no multiplicity control, one seed → hypothesis-generating only.
* Per-fold sign of ResNet50−DenseNet121 BA (folds 1–5): A + − − + −; B + − 0 + +, i.e. inconsistent across folds.
* Cluster vs naive image bootstrap for ResNet50−DenseNet121: cluster CI is ≈11% (B) to ≈21% (A) wider; neither excludes 0 (clustering changes precision modestly here).
* Pairwise contrasts resolved (CI excludes 0): 4/6 (A) and 3/6 (B) for BA; 4/6, 4/6 for macro-F1.
All figures: `results/audit/ranking_extras.json`.

## 9. Documentation / reproducibility status
README rewritten (was 2 lines). Prediction-only reproduction command added. Remaining gaps: no pinned `requirements` with versions (versions recorded only in `environment_*.json`); torchvision weight hashes not recorded; dataset papers to cite should be copied from the Figshare README (not verified here); `EXPERIMENT_LOG.md` lacks the multi-model entry; `PROJECT_PLAN.md` still says the three extra models "have not been run yet" (stale); no `tests/` coverage for training scripts (acceptable). Raw-data-dependent checks: NOT REEXECUTED.

## 10. Statistical limitations
Single dataset, single image-level split seed (11), single training seed (0), non-deterministic MPS; bootstrap resamples patients for a fixed trained-model set (no training/split variance); percentile intervals, no multiplicity control (24+ contrasts); OOF pooling across five different models per fold (CV estimates a procedure-average, cf. Bates–Hastie–Tibshirani; CI coverage not guaranteed); A and B differ in training-set composition, fold composition and patient overlap simultaneously → protocol contrast, not causal effect; patient-ID suffix issue (§2) may mean A is not strictly subject-disjoint; MR-ID/glioma confound unresolved.

## 11. Report / code discrepancies
| # | Finding | Disposition |
|---|---|---|
| 1 | Spec letters A/B reversed vs PR #1 and all outputs | FIX (header note) |
| 2 | `paper/RESULTS.md` contains 12 unfilled bracket placeholders; `paper/METHODS.md` is headings only | FIX (see verification notes) |
| 3 | RESULTS.md: "Exact and near-duplicate leakage were not identified as the main source of inflation" — the experiment does not decompose inflation sources, and near-duplicate screening is recall-limited | FIX |
| 4 | `EXPERIMENT_LOG.md` only EXP-001; `PROJECT_PLAN.md` stale on multi-model status | FIX |
| 5 | Patient-ID stem structure not mentioned anywhere | NEEDS_NEW_EVIDENCE / disclose |
| 6 | Numerical tables vs recomputation | none (PASS) |
| 7 | ResNet50 vs new models: different run scripts/commits | disclose |

## 12. Main-track novelty assessment
See `reports/MAIN_TRACK_PRIOR_ART.md`, `NOVELTY_BOUNDARY.md`, `CLAIM_PRIOR_ART_MATRIX.md`. Patient-overlap inflation is established prior art, and a very close brain-tumor/Cheng-dataset contamination audit (arXiv:2610.00421, 30 Sep 2026) including an architecture-ordering result appeared two weeks before this review. The unoccupied gap (reliability of model selection under dependence, with a seed-level control) is not yet addressed by our evidence.

## 13. Blocking scientific issues
1. No control separating protocol-induced ranking change from ordinary split/training-seed noise (needs A vs A′ replicates).
2. Single dataset / split seed / training seed.
3. The key contrast (ResNet50 vs DenseNet121) is statistically unresolved in both protocols.
4. Estimand is a protocol contrast, not an identified overlap effect.
5. Possible non-disjointness of original folds via ID-suffix siblings; MR/glioma confound.
6. Prior art (#1, and unread lung-CT paper) overlaps the empirical headline.

## 14. Recommended next action
Team decision on ONE extension: replicated-split ranking-stability experiment (see `paper/MAIN_TRACK_STORY_PROPOSAL.md`, ~120 runs ≈ 9 h on the existing Mac; needs explicit approval). Before that, free actions: read the lung-CT and 2610.00421 papers fully; resolve the ID-suffix question (e-mail dataset contact; run a stem-grouped fold sensitivity from existing predictions); fix wording per verification notes.

## Verdicts
* **MAIN_TRACK_READINESS: REQUIRES_SCIENTIFIC_REFORMULATION** (current evidence is a careful, verified replication-level result: CURRENTLY_REPLICATION_ONLY for novelty; the reformulated "model-selection reliability under dependence" question is the only route up, and it is BLOCKED on new evidence).
* **Oral-level readiness: NOT SUPPORTED.** No unusually strong insight or general principle is yet evidenced.

## Claim–evidence table
| Claim | Evidence (file → computed result) | Disposition |
|---|---|---|
| 3,064 images/233 patients/708-1426-930 | `reports/manifests/image_manifest.csv` → recomputed | PASS |
| 0 exact duplicates | manifest hashes → 0 duplicated | PASS (hash computation NOT REEXECUTED) |
| Cross-patient near-duplication absent | `cross_patient_high_similarity_pairs.csv` → max NCC 0.932, SSIM 0.855 | NEEDS_NEW_EVIDENCE (threshold unfrozen, recall-limited) — wording FIX |
| Original folds patient-disjoint | `data/splits/patient_disjoint_original.csv` → 0/233 patients across folds (by PID) | PASS by PID; individuals UNRESOLVED (ID stems) |
| Image-level arm has patient overlap | split B → patients spanning ≥2 folds (see `reproduction_checks.json`) | PASS |
| ResNet50 BA 0.9386 (A) / 0.9816 (B), Δ 0.0430 [0.0237, 0.0673] | `results/sanity/oof_predictions_*.csv` → recomputed | PASS |
| All 40 headline metrics, per-class, confusion matrices | `results/multimodel/*.csv` → max diff <1e-9 | PASS |
| Image-level ↑ BA for all 4 models | `protocol_differences.csv` → +0.032 to +0.046, CIs exclude 0 | PASS |
| Image-level ↓ ECE/Brier for all 4 | same → ECE −0.019…−0.029; Brier −0.050…−0.066, CIs exclude 0 | PASS (interpretation FIX) |
| Best architecture changes between protocols (point) | `ranking_summary.csv` | PASS (descriptive) |
| Best architecture is statistically different between protocols | pairwise CIs include 0 (4/4) | UNSUPPORTED |
| Ranking reliability reduced by leakage | no A-vs-A′ control; one seed | NEEDS_NEW_EVIDENCE |
| Causal per-image effect of overlap | not identified | UNSUPPORTED |
| General principle for model selection | none tested | REFORMULATE |
