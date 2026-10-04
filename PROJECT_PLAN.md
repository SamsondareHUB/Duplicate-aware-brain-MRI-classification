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
- Patient-disjoint original fold structure: strongly empirically supported (the `cvind` mapping gives 0 patients in more than one fold); not author-confirmed.
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

ResNet50 has completed the initial sanity experiment.

The next planned phase extends the SAME frozen protocol to MobileNetV2, EfficientNet-B0, and DenseNet121 in order to test model-ranking stability. These three models have not been run yet.

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

This is the next major analysis, not an established result. Model-ranking distortion has not yet been demonstrated.

Planned analysis will assess:
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

Completed:
- dataset selection
- full dataset audit
- duplicate / similarity audit
- patient-disjoint fold validation
- matched image-level split construction
- ResNet50 sanity experiment
- paired patient-bootstrap analysis
- reproducibility package in PR #1

Current decision: GO_TO_MULTI_MODEL (approved by Sammie; multi-model phase not yet launched)

Next:
- team review of PR #1
- freeze/finalize multi-model experiment specification
- extend unchanged protocol to MobileNetV2, EfficientNet-B0 and DenseNet121
- evaluate model-ranking stability

## Timeline

A rapid two-person research sprint of roughly 2-3 weeks, scoped to four architectures under the two matched protocols. No large benchmark expansion is planned.

- Week 1: PR #1 review, multi-model specification finalized, three remaining architectures run under the frozen protocol.
- Week 2: paired bootstrap, calibration and per-class analysis, model-ranking stability analysis.
- Week 3 (if needed): additional seeds per the specification, documentation, and first paper draft.
