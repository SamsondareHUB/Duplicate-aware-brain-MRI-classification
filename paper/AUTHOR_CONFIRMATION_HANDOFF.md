# Author-confirmation handoff for the manuscript

For Sammie. This file is a standalone drafting aid. It does **not** modify `paper/METHODS.md` or `paper/RESULTS.md`; paste or adapt what you need. All numbers come from `reports/PATIENT_ID_VALIDATION.md` and the committed results; nothing was retrained.

## 1. Status in one paragraph
Dataset author Dr. Jun Cheng replied: *"Please treat the 233 patients as representing 233 different individuals, consistent with the published paper. Unfortunately, I can't find the original DICOM data, so I'm unable to verify the identifiers further."* Decision: the original 233-ID five-fold patient-disjoint evaluation is **primary**; the 209-group prediction-only analyses are **secondary robustness checks**; no 209-group retraining was done or is planned.

**Keep apart in the text:** *author-supported* (the 233 PIDs are 233 individuals) versus *independently measured* (fold disjointness by PID; group counts; sensitivity results). The author did not verify the identifiers, did not explain the `MR…` letter suffixes, and did not comment on the `cvind` fold mapping.

## 2. Publication-ready wording

### 2.1 Dataset and patient identity (Methods, "Dataset" / "Image and Patient Identifiers")
> We used the Cheng et al. brain tumor dataset (Figshare article 1512427, v8; CC BY 4.0), which contains 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients (meningioma 708, glioma 1,426, pituitary tumor 930 slices). The patient identifier is the dataset's `PID` field (233 distinct strings, each associated with a single tumor class). Following the dataset documentation and the clarification we obtained from the dataset author (see below), we treat the 233 `PID` values as 233 different individuals.

### 2.2 Original five-fold split (Methods, "Evaluation Protocols")
> The patient-disjoint protocol uses the five-fold partition distributed with the dataset (`cvind`). Reading `cvind` in file order assigns every `PID` to exactly one fold (0 of 233 patients appear in more than one fold). This mapping was inferred from the data rather than documented; the dataset author's reply did not address it. The matched image-level protocol has the same fold sizes and fold-by-class counts but allows slices of one patient to occur in both training and test data.

### 2.3 Author clarification (Methods, or Acknowledgements/Data note)
> In personal communication, the dataset author (J. Cheng) advised us to treat the 233 patients as 233 different individuals, consistent with the original publications, and reported that the original DICOM data could no longer be located. We were therefore unable to verify patient identifiers against source records. Forty-two `PID` values (811 images, all glioma) share a six-digit MR number with at least one other `PID` and differ only in a trailing letter; neither the dataset documentation nor the author's reply explains these suffixes, and we make no assumption about their meaning.

(Citing a personal communication: confirm with Dr. Cheng and your venue's policy before including the name. The quote in section 1 is verbatim.)

### 2.4 209-group sensitivity analysis (Results or Supplement)
> As a robustness check we considered a conservative alternative identity grouping in which an `MR` + six-digit identifier with one trailing capital letter was merged with its numeric stem, giving 209 groups (18 groups merging 42 `PID` values; 811 images). Sixteen of these groups have members in different original folds (734 images, 23.96%; all glioma). Using existing predictions only, (i) accuracy on these potentially exposed images was no higher than on matched unexposed `MR` glioma images for any of the four models (differences of −0.4 to −0.7 percentage points; stem-clustered confidence intervals include zero), (ii) re-running the patient-cluster bootstrap with 209 clusters moved every 95% confidence interval by at most 0.0005, and (iii) the model orderings and the conclusion that the top two patient-disjoint models are statistically indistinguishable were unchanged; excluding the 734 exposed images left the gap between protocols at least as large (ResNet50 balanced accuracy difference +0.050). These analyses re-use the trained models and do not substitute for retraining with group-disjoint folds, which we did not perform.

### 2.5 Limitations (Discussion)
> Patient identity rests on the dataset's identifiers and the dataset author's statement; the original DICOM records were unavailable, so identifiers could not be independently verified. The meaning of the letter suffixes on 67 `MR…` identifiers is unknown. Because accuracy on the affected images is near ceiling, our sensitivity analyses show an absence of detectable inflation rather than proof of none. All `MR`-prefixed patients are glioma, so identifier format may be confounded with tumor class or acquisition source; accuracy on numeric-ID glioma (0.76–0.83) was lower than on `MR` glioma (0.99+). The fold mapping was inferred rather than author-confirmed. Results come from a single dataset, one image-level split seed (11) and one training seed (0), with non-bitwise-deterministic Apple-MPS training; bootstrap intervals reflect patient sampling only. The protocol contrast is an evaluation-design contrast, not an identified per-image causal effect, and no clinical-readiness claim is made.

### 2.6 Claims to avoid
* "Subject-disjoint" / "verified independent patients" without the qualifier "per dataset author".
* Any explanation of the suffix letters (e.g., repeat visits, lesions, sites).
* Any corrected 209-group performance estimate: none exists.
* A statistically established change of best architecture (ResNet50 vs DenseNet121 interval includes zero under both protocols).

## 3. Unfinished Methods/Results placeholders
`paper/METHODS.md` is a headings-only skeleton: Study Design; Dataset; Image and Patient Identifiers; Duplicate Audit; Evaluation Protocols; Models and Training; Performance and Calibration Metrics; Statistical Analysis; Reproducibility. All nine sections are empty. Sources: `paper/METHODS_VERIFICATION_NOTES.md`, the frozen recipe in `README.md`/`PROJECT_PLAN.md`; use 2.1–2.3 above for the identity sections.

`paper/RESULTS.md` bracketed placeholders still to fill:
| Section | Placeholder |
|---|---|
| Dataset and leakage audit | `[0]` exact-duplicate groups; `[briefly state the near-duplicate finding and method/threshold]` (note: threshold was not frozen); also add the `cvind` inferred + author-clarification sentence (see `RESULTS_VERIFICATION_NOTES.md` line ~41) |
| Image-level vs patient-disjoint | `[insert metric names and exact values from PR #1]` (calibration); `[summarize … all four architectures …]`; `Table [X]` |
| Model rankings | `[model]` (twice), `[report estimate, interval, or test result]` |
| Calibration | `[name the calibration metric or metrics]`, `[describe the direction and magnitude …]`; reliability diagrams (`reports/figures/`) |
| Error patterns | `[insert findings from the error analysis]`, `[insert relevant per-class recall/F1 findings]` |
| (new) | Add a 209-group sensitivity paragraph (2.4) and limitations (2.5) |

Values for these placeholders are in `reports/MULTIMODEL_RESULTS.md` and `results/multimodel/*.csv`.

## 4. Readiness
Identity/evidence questions are closed; the documentation is consistent. Remaining work is drafting (Methods, filling Results placeholders, figures/tables) and your decision on how to cite the personal communication.
