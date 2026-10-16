# Patient-ID suffix validation

Reproduce: `python scripts/validate_patient_ids.py` (prediction/manifest-only; no raw MRI, no training). Outputs: `results/audit/patient_id/` (`patient_id_validation.json`, `suffix_sharing_groups.csv`, `raw_pid_table.csv`, `stem_table.csv`). Tests: `tests/test_patient_id_validation.py`. All numbers below are recomputed by the script from `image_manifest.csv` and the split files (manifest and split fold/label/PID alignment is asserted).

## Decision: CASE C — suffix semantics UNRESOLVED
Nothing we could find states what the trailing letters mean (§3). We therefore report raw-PID and hypothetical-stem results side by side, keep the uncertainty in the paper, and do **not** claim individual-level independence of the original folds. No retraining or split change was made.

## 1. Formats (233 raw PIDs, 3,064 image records)
| Format | Raw PIDs | Images | Classes | Examples |
|---|---|---|---|---|
| numeric (5–6 digits) | 166 | 1,843 | meningioma 708, pituitary 930, glioma 205 | `100360`, `100416` |
| `MR` + 6 digits, no suffix | 24 | 460 | glioma only | `MR033420`, `MR034694` |
| `MR` + 6 digits + one capital letter | 43 | 761 | glioma only | `MR017260F`, `MR024780B` |
Letters seen: B C D E F G I. Each raw PID has exactly one class.

## 2. Hypothetical normalization (sensitivity device only)
Rule: strip **one** trailing `[A-Z]` from IDs matching `^MR\d{6}[A-Z]$`; every other ID is left unchanged (unit-tested incl. numeric, bare-MR, wrong-length, multi-letter, lowercase, idempotence).
| Quantity | Value |
|---|---|
| unique raw PIDs | 233 |
| unique normalized stems | **209** |
| suffix-sharing groups (stems with ≥2 raw PIDs) | **18** (14 with 2 PIDs, 2 with 3, 2 with 4) |
| affected raw PIDs | **42** (of 67 MR PIDs) |
| affected images | **811** (all glioma; 26.5% of dataset) |
| stems spanning ≥2 original folds | **16** (the other 2: MR048944 in fold 5, MR053110 in fold 1) |
| images in cross-fold stems | **734** (23.96%) |
| groups containing the bare `MR…` PID plus suffixed PIDs | 12 groups (15 suffixed PIDs); the other 6 contain only suffixed PIDs, e.g. `MR024780B/D/E/G` |
These match the earlier audit (18 / 42 / 16 / 734), recomputed independently here. Per-group rows (PIDs, image count, class, folds, `spans_folds`): `results/audit/patient_id/suffix_sharing_groups.csv`. Largest: MR040240 (4 PIDs, 97 images, folds 2|3|4|5), MR024780 (4 PIDs, 51 images, folds 1|4), MR051644 (3 PIDs, 88 images, folds 3|4).

## 3. What do the suffixes mean?
| Source checked | Finding |
|---|---|
| Figshare README 2024.txt (MD5 `4188f7c3…`, copy: `reports/manifests/figshare_README_2024.txt`) | "`cjdata.PID`: patient ID" and "233 patients"; **no explanation of suffixes**; "No further details are available due to the long passage of time." |
| Figshare article description (API) | no mention; no linked references |
| Cheng et al., PLoS ONE 10(10):e0140381 (2015) and 11(6):e0157112 (2016), read via web fetch | state records were de-identified, "233 patients", and that the five folds were made by randomly partitioning the **233 patients** so slices of one patient never span train/test; **no ID format or suffix explanation** |
Status: *same individual, multiple scans* — **not confirmed**; *separate individuals* — **not confirmed**; **UNRESOLVED**. Observations (suggestive only, not evidence of identity): the shared 6-digit stem; suffix letters usually non-contiguous (B,D,E,G); 12/18 groups also contain the bare ID; the authors' own statement shows they treated the 233 raw IDs as the patient unit. Weak imaging evidence: none of the cross-ID pairs in the committed top-similarity tables (8 in the top-200 list, the 19 with NCC ≥ 0.90) is a same-stem pair (max cross-PID NCC 0.932) — uninformative, since those tables contain only the highest-similarity pairs.

**Draft question for the dataset author (NOT sent; needs your approval):** "In `cjdata.PID` some IDs share a six-digit `MR` number and differ only by a trailing letter (e.g. MR024780B, MR024780D, MR024780E, MR024780G). Do these denote (a) the same patient imaged in different studies/sessions, (b) different patients, or (c) something else (e.g. lesion or series index)? Was the 233-patient count and the 5-fold `cvind.mat` partition built on the full string, or on the number stem? Contact per the README: Jun Cheng, Shenzhen University (chengjun583@qq.com)."

## 4. Impact on the experiment (protocol A = original folds; existing predictions only)
| Interpretation | Identity groups | Groups crossing folds | Test images potentially exposed* | Class of exposed |
|---|---|---|---|---|
| raw PID | 233 | **0** | **0 (0%)** | – |
| hypothetical stem | 209 | **16** | **734 (23.96%)**; per fold 136 / 171 / 83 / 228 / 116 | glioma 734 (51.5% of glioma images) |
\*an image whose stem has another raw PID in a different fold (i.e. a training fold). Under protocol B (image-level) 224 of 233 raw PIDs already span folds.

**Model performance, protocol A, affected vs unaffected (accuracy; `…patient_id_validation.json`):**
| Model | Exposed (734, MR glioma) | MR glioma, not exposed (487) | numeric-ID glioma (205) |
|---|---|---|---|
| ResNet50 | 0.985 | 0.992 | 0.805 |
| MobileNetV2 | 0.992 | 0.996 | 0.761 |
| EfficientNet-B0 | 0.990 | 0.996 | 0.800 |
| DenseNet121 | 0.990 | 0.998 | 0.829 |
Exposed images are classified **no better** than the matched unexposed MR-glioma images (differences −0.4 to −0.7 points; stem-clustered CIs include 0 for all four models: `results/audit/pid_suffix_sensitivity.json`). All MR-glioma images are classified far better than numeric-ID glioma (0.76–0.83), which re-raises the unresolved ID-format/source confound but is not an exposure effect. Because accuracy is near ceiling, this is *absence of detectable inflation*, not proof of none.

**Descriptive subset (not a corrected estimate):** dropping the 734 exposed images (n = 2,330; 708 meningioma / 692 glioma / 930 pit; class mix changes) gives A balanced accuracy ResNet50 0.930, MobileNetV2 0.913, EfficientNet-B0 0.916, DenseNet121 0.938, versus B 0.980 / 0.967 / 0.964 / 0.976: the same orderings (DenseNet121 > ResNet50 under A; ResNet50 > DenseNet121 under B), and a protocol gap at least as large (ResNet50 +0.050).

**Stem-clustered vs PID-clustered bootstrap** (5,000 reps, seed 2026, class-stratified, 209 vs 233 units): every 95% CI moves by ≤ 0.0005 (e.g. ResNet50 A BA 0.9119–0.9609 vs 0.9114–0.9608; B−A for DenseNet121 0.0145–0.0527 vs 0.0143–0.0528). ResNet50−DenseNet121 BA: A (−0.0163, 0.0019) vs (−0.0160, 0.0015); B (−0.0015, 0.0104) both; macro-F1 likewise. **Still includes 0 under both identities.**

**Caveat (explicit):** these are prediction-only sensitivities. A stem-clustered bootstrap changes only the *uncertainty* calculation; it is NOT equivalent to retraining with stem-disjoint folds, and no corrected patient-disjoint performance estimate is claimed. Subsets/ceilings cannot show what accuracy would be had the 16 stems been kept together in training-free folds.

**Robustness of headline conclusions under the stem interpretation:**
* Image-level > patient-disjoint performance and apparent calibration advantage: *robust* (gap persists/grows when exposed images are dropped; CIs unchanged).
* Statistical non-resolution of ResNet50 vs DenseNet121 and the descriptive point-ranking change: *unchanged*.
* Absolute patient-disjoint performance level: *uncertain* — if suffixed IDs are one person, protocol A may be mildly optimistic for 734 glioma images; no sign of this in the data, but not excluded. "Subject-disjoint" must not be claimed without qualification.
* The A/B contrast itself would, if anything, be *understated* under CASE B (A would contain some overlap).

## 5. Scientific cases
* **CASE A (verified distinct)**: not verified. * **CASE B (verified same individual)**: not verified. * **CASE C (unresolved): current.**
Minimum corrected experiment *if* CASE B were confirmed (NOT executed, requires approval): regenerate the patient-disjoint arm with folds grouped by stem (`StratifiedGroupKFold`, 209 groups, matched fold sizes ±, seed fixed), keep the frozen recipe, and rerun protocol A for the four models = 4 × 5 = **20 runs ≈ 1.5 h** (4.4 min/run measured); the image-level arm B can stay (its matched sizes would need re-matching to the new fold sizes → +20 runs ≈ 1.5 h for a strictly matched comparison). Not needed under CASE A; not currently justified under CASE C.

## 6. Recommended paper wording (CASE C)
> "Patient identifiers are the dataset's `PID` strings (233 distinct values). The original five-fold partition has no PID in more than one fold. Forty-two PIDs (811 images, all glioma) share a six-digit MR number with at least one other PID and differ only by a trailing letter (209 distinct stems); sixteen such stems have PIDs in different folds (734 images). The dataset documentation does not state whether these denote one individual, so we cannot exclude subject overlap across the original folds for up to 23.96% of images. Sensitivity analyses on existing predictions (stem-clustered bootstrap; evaluation excluding potentially exposed images) did not change our conclusions; these are not substitutes for stem-disjoint retraining."

## 7. Unresolved questions
1. Meaning of the suffix letters (author question drafted, unsent). 2. Whether `cvind` was built on full PID strings (consistent with 0/233 overlap and the paper's text; not proven for stems). 3. MR/glioma ID-format confound (accuracy gap numeric vs MR glioma). 4. Suffix groups' imaging similarity could be checked on raw data (NOT REEXECUTED; not done).
