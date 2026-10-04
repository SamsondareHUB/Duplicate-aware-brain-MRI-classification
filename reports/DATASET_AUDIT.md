# Dataset Audit

Dataset: Cheng et al. brain tumor dataset, Figshare article 1512427, v8. Branch `melody/dataset-audit-v1`.
Every number below is reproducible from `reports/manifests/audit_summary.json` and the CSVs next to it, produced by
`python scripts/download_figshare.py --extract` then `python scripts/audit_dataset.py`.
No split was created and no model was trained.

**Status labels:** CONFIRMED = directly measured, threshold-free. CANDIDATE / THRESHOLD-DEPENDENT = depends on a
similarity cutoff that has not been frozen. UNRESOLVED = cannot be established from the current evidence.

**Terminology (kept distinct throughout):**
- *Exact duplicate*: pixel-identical image content.
- *Near-image duplicate*: a pair of images that are near-identical in content; no definition/threshold is frozen.
- *Same-patient neighbouring-slice dependence* (patient-level overlap): different slices of the same patient. This is
  **not** called near-duplication here.

## Summary of conclusions
| Question | Status |
|---|---|
| Exact duplicate leakage | **CONFIRMED ABSENT** (0 groups) |
| Cross-patient near-identical duplication | **NOT SUPPORTED** by the current audit (max full-res NCC 0.932 / SSIM 0.855 for cross-patient candidate pairs; threshold not frozen) |
| Same-patient neighbouring-slice dependence | **CONFIRMED**, and scientifically distinct from duplication |
| `cvind` file-order mapping | **STRONGLY EMPIRICALLY SUPPORTED**, not author-confirmed |
| `MR`-prefixed patient-ID / class association | **UNRESOLVED** provenance / source-confounding concern; not confirmed source leakage |
| 256x256 images from patient 108931 | CONFIRMED dataset heterogeneity; documented, not interpreted further |

## Source and provenance
- Source: official Figshare only (https://doi.org/10.6084/m9.figshare.1512427, v8, published 2024-12-21, CC BY 4.0). No mirror.
- Files: four `brainTumorDataPublic_*.zip` (~880 MB total), `cvind.mat`, `README 2024.txt`.
- **CONFIRMED:** all 6 files match the Figshare-API MD5 values (`reports/manifests/provenance.json`: expected vs observed MD5, sizes, file IDs).
- Raw data live in git-ignored `data/raw/`; `.gitignore` also blocks `*.mat` and `*.zip`.
- The README states 233 patients; 708/1426/930 slices; 512x512; two hospitals; 2005.9-2010.10; "no further details are available". No per-image site field exists.
- `.mat` files are MATLAB v7.3 (HDF5), read with h5py. `cjdata.PID` is a MATLAB char array decoded to a string. Fields: `image` (int16), `label`, `PID`, `tumorBorder`, `tumorMask`. **No fold index is stored in the `.mat` files**; folds come only from `cvind.mat`, and no documentation of the `cvind`-to-image mapping was found.

## File integrity
| | |
|---|---|
| `FILES_FOUND` | 3064 (flat directory, `1.mat`..`3064.mat`, contiguous) |
| `FILES_READABLE` | 3064 |
| `FILES_UNREADABLE` | 0 |

**CONFIRMED.** Malformed-file handling is explicit (a failure becomes a row with `readable=False` and an error string, covered by tests) but was not exercised by the real data.

## Dataset composition
- **CONFIRMED** class counts: meningioma 708, glioma 1426, pituitary 930 (matches README). Exactly 3 classes; no no-tumor class.
- **CONFIRMED** shapes: 512x512 = 3049; 256x256 = 15. The 15 images all belong to a single pituitary patient (108931, fold 4; files 955-957, 1070-1076, 1203-1207). Not mentioned in the README. This is dataset heterogeneity only; its effect, if any, is not assessed. dtype is int16 for all images.

## Patient structure
- **CONFIRMED** unique patients: 233 (matches README). Patients per class: meningioma 82, glioma 89, pituitary 62. No patient has more than one label.
- Slices per patient: min 1, median 13, mean 13.15, max 38.
- **UNRESOLVED (provenance / possible source confounding):** all 67 non-numeric `MR...` patient IDs (1,221 images) are glioma; all meningioma and pituitary IDs are numeric (22 glioma patients, 205 images, also have numeric IDs). The ID format may reflect source, hospital or era, but nothing in the documentation says so. This is an association, not confirmed source leakage.

## Original fold structure
- `cvind` is a 1x3064 vector with values 1-5. The mapping hypothesis tested: `cvind[k-1]` belongs to file `k.mat`.
- Evidence: under this mapping **0 of 233 patients** appear in more than one fold, whereas 1,000 random permutations of `cvind` split 217-233 patients (mean 223.4).
- Status: **STRONGLY EMPIRICALLY SUPPORTED, not author-confirmed.** It is not claimed that the dataset authors confirm patient-disjoint folds; only that this mapping yields zero patient overlap. All fold numbers below depend on it.
- `ORIGINAL_FOLD_COUNTS` (images): fold1 542, fold2 679, fold3 572, fold4 628, fold5 643. Patients per fold: 45, 46, 48, 47, 47.
- Class by fold (meningioma/glioma/pituitary): f1 112/245/185, f2 167/338/174, f3 139/231/202, f4 124/325/179, f5 166/287/190. Approximately, not exactly, balanced.
- `PATIENT_CROSS_FOLD_VIOLATIONS` = 0 under the supported mapping.

## Exact duplicate audit
Method (`src/data_audit/hashing.py`): the image is read from HDF5 and transposed to MATLAB (row, col) orientation, original int16 values untouched. Three SHA-256 hashes, all **excluding metadata** (patient, label, fold, filename, mask, border):
- `raw_array_hash`: dtype + shape + C-order little-endian bytes (primary key).
- `canonical_image_hash`: shape + values cast to int32 (lossless).
- `display_hash`: README-style min-max uint8 rescale (lossy; reported only to show it creates no extra matches).

| | |
|---|---|
| `EXACT_DUPLICATE_GROUPS` | 0 (raw), 0 (canonical), 0 (display) |
| `EXACT_DUPLICATE_CROSS_FOLD_GROUPS` | 0 |
| `CROSS_PATIENT_EXACT_DUPLICATE_GROUPS` | 0 |
| `CROSS_CLASS_EXACT_DUPLICATE_GROUPS` | 0 |

**CONFIRMED ABSENT.** `exact_duplicate_groups.csv` contains only its header. No candidate pair shares more than 35.1% identical pixels.

## Near-duplicate audit
Two stages (`src/data_audit/near.py`):
- **Stage A, retrieval** (permissive; *not* a duplicate definition): 32x32 area-resized thumbnails; a pair is retained if thumbnail NCC >= 0.90 **or** 64-bit DCT perceptual-hash Hamming distance <= 10. 4,692,516 pairs screened; `NEAR_DUPLICATE_CANDIDATE_PAIRS` = **33,432** retained.
- **Stage B, verification** on candidates only: full-resolution NCC, SSIM, normalised mean absolute difference, fraction of equal pixels. The 15 images at 256x256 are compared after downsizing the larger image (`resized_for_comparison`).

Full-resolution distribution of the 33,432 candidates (`similarity_summary.csv`):

| population | metric | n | median | q95 | q99 | max |
|---|---|---|---|---|---|---|
| same patient | NCC | 4,852 | 0.797 | 0.906 | 0.927 | 0.949 |
| same patient | SSIM | 4,852 | 0.663 | 0.793 | 0.831 | 0.873 |
| cross patient | NCC | 28,580 | 0.705 | 0.801 | 0.847 | 0.932 |
| cross patient | SSIM | 28,580 | 0.700 | 0.773 | 0.795 | 0.855 |

Counts at labelled **provisional** cutoffs (`similarity_distribution.csv`; none is a frozen definition):

| full-res NCC >= | pairs | within-pt/within-fold | within-pt/cross-fold | cross-pt/within-fold | cross-pt/cross-fold | cross-class |
|---|---|---|---|---|---|---|
| 0.80 | 3848 | 2365 | 0 | 271 | 1212 | 563 |
| 0.85 | 1542 | 1293 | 0 | 39 | 210 | 88 |
| 0.90 | 334 | 315 | 0 | 7 | 12 | 7 |
| 0.95 and above | 0 | 0 | 0 | 0 | 0 | 0 |

SSIM >= 0.85: 15 pairs (14 within-patient); SSIM >= 0.90: 0.

Observations:
- **CONFIRMED:** no pair reaches NCC 0.95, SSIM 0.90, or 35% equal pixels. The similarity distribution is continuous: there is no separate high-similarity mode that distinguishes duplicates from ordinary anatomical similarity.
- Same-patient and cross-patient distributions overlap heavily at the top (NCC 0.949 vs 0.932). These metrics therefore cannot separate "same anatomy in another patient" from "same image", and **the audit does not support cross-patient near-identical duplication**. The 19 cross-patient pairs with NCC >= 0.90 (`cross_patient_high_similarity_pairs.csv`; 12 same-class, 7 cross-class, 12 cross-fold) are listed for inspection, not claimed as duplicates. The two top pairs (files 1878/2229 and 1877/2228, patients MR048994/MR036496B, both glioma) look worth a visual check.
- Nearest-neighbour similarity (thumbnail NCC): median best same-patient neighbour 0.962 vs median best other-patient neighbour 0.895.
- **Geometric robustness check** (`geometric_transform_diagnostic.csv`): 415 pairs look alike (thumbnail NCC >= 0.95) only after a flip/rotation of one image; at full resolution the best reaches NCC 0.942, SSIM 0.842. No hidden flipped/rotated duplicates were found.
- **UNRESOLVED:** Stage A recall. Pairs below thumbnail NCC 0.90 and pHash 10 are not examined, and shifts, crops or rescales (other than the dihedral set) were not tested.
- `NEAR_DUPLICATE_THRESHOLD_FROZEN = NO`. The full 6.7 MB candidate table is a local, git-ignored, regenerable intermediate; the compact files are `similarity_summary.csv`, `top_similarity_pairs.csv` (top 200) and `cross_patient_high_similarity_pairs.csv`.

## Same-patient neighbouring-slice dependence
**CONFIRMED.** 233 patients contribute 3,064 slices (median 13 per patient). The highest-similarity pairs in the whole dataset are overwhelmingly adjacent slices of one patient (e.g. files 1857/1858; 4,852 of 33,432 candidate pairs are same-patient), and each image's best same-patient neighbour is on average more similar than its best other-patient neighbour. This is a patient-level dependence structure, not duplication: these images are different slices, and none is pixel-identical. It is the main construct for the next experiment. All same-patient pairs sit within one original fold (0 within-patient/cross-fold pairs).

## Cross-fold leakage findings
| Pair type | Exact | Stage-A candidate pairs (retrieval, not leakage) |
|---|---|---|
| within-patient / within-fold | 0 | 4,852 |
| within-patient / cross-fold | 0 | **0** |
| cross-patient / within-fold | 0 | 5,182 |
| cross-patient / cross-fold | 0 | 23,398 (= `NEAR_DUPLICATE_CROSS_FOLD_CANDIDATE_PAIRS`) |
| cross-class | 0 | 8,659 |

- **CONFIRMED:** no exact-duplicate pair crosses folds (there are none). No same-patient pair crosses folds under the supported mapping.
- **CANDIDATE / THRESHOLD-DEPENDENT:** cross-patient/cross-fold similar pairs: 210 at NCC >= 0.85, 12 at >= 0.90, 0 at >= 0.95. These are not claimed as leakage.
- Not measured: performance under any split (splits are out of scope).

## Issues requiring team decision (status)
1. Terminology and construct are set: exact leakage absent; near-image duplicate leakage not established; patient-level overlap is the construct to test.
2. `cvind` mapping: strongly supported but not author-confirmed. Optionally ask the dataset contact listed in the README.
3. `MR`-prefixed ID / glioma association: UNRESOLVED provenance concern. Consider reporting results with and without it.
4. 15 images at 256x256: decide the resize policy for model inputs.
5. Original folds are imbalanced in size (542-679 images) and class mix; 233 patients means high patient-level variance.
6. Near-duplicate threshold: **not frozen**; revisit only if the project wants a near-duplicate arm.

## Recommended next step
Tested in `reports/SANITY_EXPERIMENT.md`: original patient-disjoint evaluation against a matched image-level stratified
evaluation (ResNet50, one split seed) to isolate the effect of patient overlap. Result: image-level evaluation gave clearly
higher apparent accuracy and better apparent calibration, decision `GO_TO_MULTI_MODEL`. The near-duplicate threshold remains
**not frozen**; it is not required for that experiment.
