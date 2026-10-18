# Project Plan

## Project Title

Patient-Overlap Leakage and Model-Selection Reliability in Brain-MRI Tumor Classification

## Project Description

This project investigates how patient-level overlap between training and test images affects reported performance, calibration, error patterns, and model-selection conclusions in brain-MRI tumor classification.

The initial audit of the Jun Cheng / Figshare brain-tumor dataset (3,064 slices, 233 patients) found no exact image duplicates and no strong evidence of cross-patient near-identical duplication. The dominant dependence structure is instead multiple MRI slices originating from the same patient.

A matched ResNet50 sanity experiment showed that image-level evaluation, where slices from the same patient may occur in both training and test data, produced higher apparent performance and substantially better apparent calibration than the original patient-disjoint five-fold evaluation.

Therefore, patient-overlap leakage is the primary research focus. Exact-duplicate and near-image-duplicate contamination remain secondary audited factors rather than the primary experimental axis.

## Primary Research Question

How much does patient overlap between training and test sets inflate apparent performance and calibration, and does it alter conclusions about which transfer-learning model performs best?

## Secondary Research Questions

- Are exact or near-image duplicates a meaningful source of leakage in this dataset?
- Does patient overlap affect tumor classes differently?
- Does patient overlap change the relative ranking or apparent certainty of different model architectures?
- Are model-ranking conclusions stable under patient-disjoint evaluation?

## Evidence From Initial Audit

Details are in `reports/DATASET_AUDIT.md`, `reports/SANITY_EXPERIMENT.md` and `EXPERIMENT_LOG.md` (PR #1).

- Exact duplicate leakage: confirmed absent (0 duplicate groups).
- Cross-patient near-identical duplication: not supported by the current audit.
- Near-duplicate threshold: not frozen.
- Same-patient slice dependence: confirmed, and scientifically distinct from duplication (same-patient slices are not called near-duplicates).
- Patient-disjoint original fold structure: strongly empirically supported (the `cvind` mapping gives 0 patients in more than one fold); the `cvind` mapping itself is not author-confirmed. The dataset author has asked that the 233 PIDs be treated as 233 different individuals (not independently verifiable; see `reports/PATIENT_ID_VALIDATION.md`).
- ResNet50 sanity experiment: image-level evaluation gave materially higher apparent performance and better apparent calibration than patient-disjoint evaluation; the pre-specified rule returned GO_TO_MULTI_MODEL. This is a single model, split seed and training seed, and says nothing yet about model rankings.

## Primary Evaluation Protocols

A. Patient-disjoint evaluation
- original five-fold patient-disjoint assignment

B. Matched image-level evaluation
- image-level stratified five-fold assignment
- matched to the patient-disjoint arm in fold size and fold-by-class counts
- permits slices from the same patient to occur across training and test data

The principal comparison is B versus A.

## Secondary Audited Factors

- exact image duplicates (confirmed absent)
- near-image similarity / candidate near-duplicates (threshold not frozen)
- unresolved provenance/source concerns (e.g. the association between `MR`-prefixed patient IDs and glioma; an open concern, not confirmed source leakage)

No exact-duplicate-aware or near-duplicate-aware model-training arm is planned unless later evidence or a later specification explicitly requires one.

## Planned Models

- ResNet50
- MobileNetV2
- EfficientNet-B0
- DenseNet121

All four models have been run under the same frozen protocol (ResNet50 in PR #1; MobileNetV2, EfficientNet-B0 and DenseNet121 in PR #2).

## Frozen Experimental Protocol

Unless a documented architecture-specific implementation requirement makes a change unavoidable, the multi-model phase must preserve the protocol established in PR #1. Architecture-specific hyperparameter tuning is not part of the primary comparison, because it would weaken the matched design.

**Preprocessing**
- original MRI array
- deterministic float32 conversion
- per-image min-max normalization to [0,1]
- resize to 224x224
- grayscale replicated to 3 channels
- ImageNet normalization

**Training**
- ImageNet pretrained weights appropriate to each torchvision architecture
- batch size 32
- 10 epochs, fixed a priori
- AdamW, learning rate 1e-4, weight decay 1e-4
- cosine LR schedule
- cross-entropy, no class weights
- no early stopping, no validation-based model selection
- final-epoch evaluation
- same approved augmentation policy as the sanity run (horizontal flip p=0.5, rotation +/-10 degrees)

**Evaluation**
- pooled out-of-fold predictions
- balanced accuracy, macro-F1, accuracy
- per-class precision / recall / F1
- confusion matrices
- ECE with 15 fixed equal-width bins
- multiclass Brier score
- reliability diagrams

**Statistics**
- class-stratified patient-cluster bootstrap
- 5,000 replicates
- shared patient multiplicities across paired arms
- bootstrap seed 2026
- paired B - A effects with 95% confidence intervals

**Seeds**
- the frozen sanity-run seeds are the baseline: image-level split seed 11, training seed 0, bootstrap seed 2026
- additional split/training seeds may be added only according to the upcoming multi-model experiment specification
- existing frozen seeds are not to be silently altered

## Model-Ranking Analysis

This analysis is complete (PR #2; `reports/MULTIMODEL_RESULTS.md`). See that report for what is and is not statistically established; the top-two ranking difference is not statistically conclusive.

The analysis assessed:
- whether architecture ordering changes between image-level and patient-disjoint evaluation
- pairwise model superiority uncertainty
- ranking stability across protocols
- Kendall rank correlation or another frozen ranking-stability statistic
- bootstrap probability / agreement regarding the top-ranked model, if included in the final specification

A ranking change will be reported only if the results demonstrate it.

## Metrics

Primary:
- Balanced accuracy
- Macro-F1

Secondary:
- Accuracy
- Per-class precision, recall, F1
- Confusion matrix

Calibration:
- ECE
- Brier score
- Reliability diagram

Model-selection reliability:
- ranking stability measures to be finalized in the multi-model specification

## Expected Contribution

The project aims to determine whether image-level patient overlap changes not only reported predictive performance and calibration but also the scientific conclusion of model comparison: which architecture appears to perform best.

The duplicate audit provides a negative but important result: exact duplication is absent and cross-patient near-identical duplication is not supported, allowing the study to distinguish image duplication from patient-level dependence rather than conflating them.

## Current Status

**Phase: experiments complete; moving to manuscript preparation.**

Completed:
- dataset selection
- full dataset audit and duplicate / similarity audit (`reports/DATASET_AUDIT.md`)
- patient-disjoint fold validation and matched image-level split construction
- ResNet50 sanity experiment and paired patient-bootstrap analysis (PR #1)
- all 40 original model runs: 4 architectures x 5 folds x 2 protocols (ResNet50 from PR #1, 30 new runs in PR #2)
- multi-model statistical analysis: paired patient bootstrap, calibration, per-class results, ranking-stability diagnostics (PR #2)
- independent reproducibility check from committed predictions (`scripts/reproduce_from_predictions.py`, 200/200 checks) and independent audit (`reports/FINAL_INDEPENDENT_AUDIT.md`)
- patient-ID (PID) validation and prediction-only 209-group sensitivity analyses (PR #3; `reports/PATIENT_ID_VALIDATION.md`)
- dataset-author clarification (Dr. Jun Cheng): treat the 233 patients as 233 different individuals; original DICOM data unavailable, so identifiers could not be verified further

Evaluation hierarchy (decided by Sammie):
- **PRIMARY:** original 233-ID five-fold patient-disjoint evaluation (protocol A) versus matched image-level evaluation (protocol B). PR #1 / PR #2 results, statistics and rankings are preserved unchanged.
- **SECONDARY (robustness):** 209-group prediction-only sensitivity analyses (stem-clustered bootstrap, exposed-image subset, exposed-vs-unexposed accuracy). These are existing-prediction analyses, not retraining.
- **Not required under the current team decision:** 209-group retraining (20-run and 40-run plans cancelled; `reports/RETRAINING_PLAN_209_GROUPS.md` is a historical contingency plan, not executed). No 209-group model has been trained.

Limits that remain (to be disclosed in the paper): the author's statement is not an independent metadata verification; the meaning of the `MR…` letter suffixes is not established; the `cvind` file-order mapping is not author-confirmed; the MR-prefix / glioma association is an unresolved source confound; single image-level split seed (11) and single training seed (0).

Next: **manuscript preparation.** See `paper/AUTHOR_CONFIRMATION_HANDOFF.md` for publication-ready wording and the list of unfinished Methods/Results placeholders.

## Timeline (original sprint plan; experiments and analysis phases are now complete)

A rapid two-person research sprint of roughly 2-3 weeks, scoped to four architectures under the two matched protocols. No large benchmark expansion is planned.

- Week 1: PR #1 review, multi-model specification finalized, three remaining architectures run under the frozen protocol.
- Week 2: paired bootstrap, calibration and per-class analysis, model-ranking stability analysis.
- Week 3 (if needed): additional seeds per the specification, documentation, and first paper draft.
