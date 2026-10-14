# Methods — verification notes (support for Sammie; does NOT replace `paper/METHODS.md`)

`paper/METHODS.md` currently contains section headings only. Everything below was checked against committed files; "source" gives the file.
Suggested text is deliberately conservative. Items tagged **[disclose]** are limitations a reviewer will ask about.

## Study design
Retrospective, single-dataset comparison of two evaluation protocols (not an intervention). Protocol A: the dataset's original five-fold patient-disjoint partition. Protocol B: a five-fold image-level partition with the same fold sizes and fold×class counts as A, generated with seed 11 (so patients' slices may fall in both training and test folds). All 3,064 images are tested exactly once per protocol (out-of-fold). Four ImageNet-pretrained CNNs are trained and evaluated under the same frozen recipe. Source: `data/splits/split_validation.json`, `reports/MULTIMODEL_RESULTS.md`.
> Use the labels *patient-disjoint* and *image-level*; mention that file/column letters are A and B in PR #1 notation (the multi-model spec reverses the letters).

## Dataset (source: `reports/DATASET_AUDIT.md`, `reports/manifests/provenance.json`)
Cheng et al. brain tumor dataset, Figshare article 1512427 v8 (2024-12-21), CC BY 4.0; 3,064 T1-weighted contrast-enhanced slices; 233 patients; meningioma 708 / glioma 1,426 / pituitary 930; 82 / 89 / 62 patients per class; 1–38 slices per patient (median 13). 3,049 images are 512×512 and 15 (one pituitary patient) are 256×256 [disclose; resized with all others]. No no-tumour class. Hospital/site is not recorded.

## Image and patient identifiers
Patient identifier = `cjdata.PID` (233 distinct strings, each with one class). 67 IDs begin with `MR` (1,221 images, all glioma) — association unresolved [disclose as possible source confound]. **[disclose, new]** Some IDs share a numeric stem with a letter suffix (e.g. `MR024780B/D/E/G`; 18 stems, 42 IDs, 811 images); the dataset documentation does not say whether these are distinct patients. In the original folds 16 of these stems have IDs in different folds (734 images). We treat each distinct PID as a patient, as the dataset README does. Source: `results/audit/pid_suffix_sensitivity.json`.

## Duplicate audit
> "We hashed the decoded int16 pixel array (SHA-256 of dtype, shape and C-order bytes), a lossless int32 canonical form and a min–max uint8 display form, and found no repeated hash (0 exact-duplicate groups). Candidate near-duplicates were retrieved among all 4,692,516 image pairs using 32×32 thumbnail NCC ≥ 0.90 or 64-bit DCT-hash Hamming distance ≤ 10 (33,432 candidates) and verified at full resolution with NCC and SSIM. No pair reached NCC 0.95 or SSIM 0.90 (maximum 0.949 and 0.873, both within-patient; cross-patient maximum 0.932 and 0.855). No near-duplicate threshold was fixed, and the retrieval stage has limited recall, so we do not claim the absence of cross-patient near-duplication; we report that none was found by this screen."
Source: `reports/manifests/audit_summary.json`, `similarity_summary.csv`.

## Evaluation protocols
Original folds: `cvind.mat` read in file order (image k ↔ `cvind[k-1]`); 0/233 patients appear in more than one fold under this mapping vs 217–233 for 1,000 random permutations. The mapping is **not author-confirmed** [disclose]. Image-level folds: `data/splits/imagelevel_seed11.csv` (SHA-256 `e8b0442c…74f2`), `numpy.random.default_rng(SeedSequence([11, class]))`, stratified so fold sizes and fold×class counts equal A; deterministically regenerated (check in `split_validation.json`). Original split SHA-256 `61bec5aa…9e`. Many patients span folds in B (see `results/audit/reproduction_checks.json`).

## Models and training (sources: `results/multimodel/environment_*.json`, run JSONs)
torchvision ResNet50 (IMAGENET1K_V1), MobileNetV2, EfficientNet-B0, DenseNet121 (each `IMAGENET1K_V1`), new 3-way linear head. Per-image min–max → antialiased bilinear resize to 224×224 → 3-channel replication → ImageNet normalisation. Batch 32; 10 epochs; AdamW lr 1e-4, wd 1e-4; per-iteration cosine to 0; unweighted cross-entropy; horizontal flip p=0.5 and rotation ±10°; train seed 0 reset for each fold; final epoch evaluated, no early stopping or model selection, no hyperparameter tuning. Five folds × two protocols × four models = 40 training runs (ResNet50 from PR #1; 30 new). Hardware: Apple MPS, torch 2.11.0, torchvision 0.26.0, Python 3.13.6 (non-deterministic backend) **[disclose]**. ResNet50 and the other models were produced by different scripts at different commits with the same documented recipe **[disclose]**.

## Performance and calibration metrics (source: `src/leakage/metrics.py`; independently re-implemented in `scripts/reproduce_from_predictions.py`)
Accuracy; balanced accuracy = mean per-class recall; macro-F1 = unweighted mean of per-class F1; per-class precision/recall/F1; confusion matrices; top-label ECE with 15 equal-width bins; multiclass Brier = mean over images of Σ_k (p_k − 1[y=k])² (range 0–2). All computed on the pooled 3,064 out-of-fold predictions. No post-hoc recalibration.

## Statistical analysis
Class-stratified patient-cluster bootstrap: 5,000 replicates, seed 2026; patients resampled with replacement within each tumour class (233 patients; per-class totals fixed); every metric recomputed from per-patient sufficient statistics; **the same patient multiplicities are used for both protocols and all four models** (paired). 95% percentile intervals. Protocol effect = image-level minus patient-disjoint. Ranking: models ordered by pooled balanced accuracy or macro-F1; Kendall τ-b between protocol orderings; bootstrap top-ranked frequency (ties split equally; none occurred); pairwise model-difference intervals. State: *intervals reflect patient sampling for the trained models only; they do not include training or split randomness*; top-ranked frequencies are descriptive bootstrap frequencies, not posterior probabilities. Exploratory analyses (patient-weighted metrics, interaction contrast, per-fold signs, fresh-seed bootstrap) are post hoc and labelled so.

## Reproducibility
Cite: split SHA-256 values, `python scripts/reproduce_from_predictions.py` (200 checks, no raw data required), `requirements-audit.txt`, environment JSONs, Figshare MD5 verification. State what was not re-executed from committed artifacts (raw-data hashing, training).
