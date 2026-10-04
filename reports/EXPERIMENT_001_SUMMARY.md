# Experiment 001 — Patient-Overlap Sanity Experiment

## Status

Completed and merged through PR #1.

## Objective

Compare matched image-level evaluation against the original patient-disjoint five-fold evaluation using ResNet50.

## Dataset

- Jun Cheng Brain Tumor Dataset
- 3,064 T1-weighted contrast-enhanced MRI slices
- 233 patients
- Classes: Meningioma, Glioma, Pituitary tumor

## Main Finding

Matched image-level evaluation produced higher apparent performance than patient-disjoint evaluation.

- Balanced accuracy, image-level: 0.9816
- Balanced accuracy, patient-disjoint: 0.9386
- Difference: +0.0430
- 95% patient-bootstrap confidence interval: +0.0237 to +0.0673
- Macro-F1 difference: +0.0436
- Calibration appeared substantially better under image-level evaluation.

## Duplicate Audit Finding

No exact duplicate groups were found. No strong evidence of cross-patient near-identical images was found. The main dependence in the dataset is patient overlap, because neighboring slices from the same patient can appear across image-level train and test splits.

## Decision

Proceed to the multi-model experiment using the frozen PR #1 protocol.
