# Duplicate-aware brain MRI classification: patient overlap, calibration and model-ranking reliability

Leakage-aware evaluation of ImageNet-pretrained CNNs on the Cheng et al. brain-tumor MRI dataset (3,064 slices, 233 patients,
3 classes). We compare the dataset's original five-fold **patient-disjoint** evaluation with a size- and class-matched
**image-level** five-fold evaluation in which slices of one patient can fall on both sides of the split, for four
architectures (ResNet50, MobileNetV2, EfficientNet-B0, DenseNet121), and quantify uncertainty with a class-stratified
patient-cluster bootstrap.

> **Scope and claim boundary.** One dataset, one image-level split seed (11), one training seed (0), a frozen 10-epoch recipe.
> The A/B contrast is an *evaluation-protocol contrast*, not an identified per-image causal effect of patient overlap.
> The best-model identity changes in the point estimates, but the ResNet50-vs-DenseNet121 interval includes zero under both
> protocols, so a reversal of the best architecture is **not** statistically established. Not a clinical-readiness claim.
> See `reports/FINAL_INDEPENDENT_AUDIT.md` for the independent verification and open issues.

## Protocol naming (important)
Prediction files and result tables use the **PR #1 labels**: **A = patient-disjoint** (`data/splits/patient_disjoint_original.csv`),
**B = image-level seed 11** (`data/splits/imagelevel_seed11.csv`). `EXPERIMENT_SPEC_MULTI_MODEL.md` uses the opposite letters;
the split files and SHA-256 values below are the source of truth. All "difference" columns are **B − A**.

| Split file | SHA-256 |
|---|---|
| `patient_disjoint_original.csv` (A) | `61bec5aaf9a5f4fce4adf89b186ffd6479eb0f6095ee20b8f60ef44274a3e49e` |
| `imagelevel_seed11.csv` (B) | `e8b0442c107396fe66a989b8d448aa0e524c1c554823a711db3cc84beeee74f2` |

## Quick check without data or training (≈ 5 s, CPU only)
```sh
python -m pip install numpy pandas scipy scikit-learn pytest      # torch / h5py are NOT needed for this
python scripts/reproduce_from_predictions.py                      # 200 checks; exit code 1 on any FAIL
python scripts/ranking_reliability_extras.py                      # exploratory ranking diagnostics
python scripts/pid_suffix_sensitivity.py                          # patient-ID-suffix provenance sensitivity
python scripts/validate_patient_ids.py                            # patient-ID suffix validation (reports/PATIENT_ID_VALIDATION.md)
python -m pytest -q tests
```
`reproduce_from_predictions.py` recomputes, from the committed out-of-fold (OOF) predictions and frozen splits only: split hashes and
patient disjointness, completeness of 3,064 OOF rows for each of 8 model×protocol sets, per-fold run records (40 runs: 10 ResNet50 +
30 new), accuracy / balanced accuracy / macro-F1 / per-class P-R-F1 / confusion matrices / 15-bin ECE / multiclass Brier, protocol
differences, the 5,000-replicate paired patient bootstrap (CIs, pairwise contrasts, top-ranked frequencies, Kendall τ-b), and compares every
value with the committed tables (`results/multimodel/*.csv`, `reports/MULTIMODEL_RESULTS.md`). Outputs: `results/audit/`.

## Repository map
| Path | Content |
|---|---|
| `reports/DATASET_AUDIT.md`, `reports/manifests/` | provenance, integrity, exact- and near-duplicate screening |
| `reports/SANITY_EXPERIMENT.md`, `results/sanity/` | ResNet50 A/B experiment (PR #1) |
| `reports/MULTIMODEL_RESULTS.md`, `results/multimodel/` | four-model analysis, OOF predictions, bootstrap draws (PR #2) |
| `reports/FINAL_INDEPENDENT_AUDIT.md` | independent audit; claim–evidence table |
| `reports/MAIN_TRACK_PRIOR_ART.md`, `NOVELTY_BOUNDARY.md`, `CLAIM_PRIOR_ART_MATRIX.md` | literature/novelty assessment |
| `paper/` | manuscript drafts (`METHODS.md`, `RESULTS.md`) and `*_VERIFICATION_NOTES.md`, `MAIN_TRACK_STORY_PROPOSAL.md` |
| `data/splits/` | frozen split files, `split_validation.json` |
| `src/data_audit`, `src/leakage` | audit and metric/bootstrap/split code |
| `scripts/` | download, audit, split, train, analyze, verify |
| `tests/` | unit tests (metrics vs scikit-learn, splits, audit) |

## Data
* Source: Cheng et al., *Brain tumor dataset*, Figshare article 1512427, **v8** (published 2024-12-21), https://doi.org/10.6084/m9.figshare.1512427, **CC BY 4.0**.
  Cite the dataset and the papers listed in its own `README 2024.txt` (Cheng et al., PLoS ONE 2015 and 2016 — check the exact references in that file).
* 3,064 T1-weighted contrast-enhanced slices (meningioma 708, glioma 1,426, pituitary 930), 233 patients, 512×512 (15 images are 256×256, all from patient 108931).
* `python scripts/download_figshare.py --extract` downloads ≈880 MB from the official Figshare API, verifies MD5s and extracts to the git-ignored `data/raw/`.
  Raw MRI is never committed. Folds come from the dataset's `cvind.mat`; the file-order mapping is strongly empirically supported (0/233 patients span folds) but **not author-confirmed**.

## Full re-execution (needs raw data, PyTorch and an Apple-MPS or CUDA GPU; ~2.5 h for the 30 new runs on one Mac)
```sh
python -m pip install -r requirements-audit.txt
python scripts/download_figshare.py --extract
python scripts/audit_dataset.py                 # dataset / duplicate audit  -> reports/manifests
python scripts/make_splits.py                   # regenerates the B split deterministically (seed 11)
python scripts/run_sanity.py                    # ResNet50, 10 runs
python scripts/serial_model.py mobilenet_v2     # likewise efficientnet_b0, densenet121 (5 folds x 2 protocols each)
python scripts/analyze_multimodel.py            # bootstrap + tables + figures
python scripts/verify_multimodel.py             # independent spot verification
```
**Not re-executed in the independent audit (raw data or GPU required):** MD5/pixel-hash recomputation, the near-duplicate screen, preprocessing, training.

## Frozen recipe
Per-image min-max → antialiased bilinear resize to 224×224 → 3-channel replication → ImageNet normalisation; torchvision ImageNet-1k V1 weights;
new 3-way linear head; batch 32; 10 epochs; AdamW (lr 1e-4, wd 1e-4); per-iteration cosine; unweighted cross-entropy; hflip p=0.5, rotation ±10°;
train seed 0 reset per fold; no early stopping or model selection (final epoch). Runs on Apple MPS (not bitwise deterministic).
Software: Python 3.13.6, torch 2.11.0, torchvision 0.26.0, numpy 2.3.2, pandas 2.3.3 (`results/*/environment*.json`).

## Prediction format
`results/multimodel/oof_predictions_<model>_<A|B>.csv`: `image_id, patient_id, test_fold, true_class (0 meningioma, 1 glioma, 2 pituitary), prob_0..2, pred_class, confidence, ...`;
ResNet50: `results/sanity/oof_predictions_<A|B>.csv` (also logits). 3,064 rows each, one prediction per image from the fold in which it was held out.

## License
Code: see `LICENSE`. Data: CC BY 4.0 (original authors).
