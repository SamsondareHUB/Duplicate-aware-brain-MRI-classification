# Manuscript Outline

## Working Title

Patient-Overlap Leakage Inflates Performance and Calibration in Brain-MRI Tumor Classification: A Multi-Model Evaluation Study

## Core Claim

Image-level evaluation can inflate apparent performance and calibration in brain-MRI tumor classification when slices from the same patient appear across training and test partitions.

## Primary Analysis

The primary analysis uses the original 233-patient patient-disjoint five-fold evaluation, following confirmation from the dataset author that the 233 patient identifiers represent distinct individuals.

## Secondary Analyses

- Exact and near-duplicate image-leakage audit
- 209-group sensitivity analysis
- Split-seed and training-seed replication analyses
- Controlled patient-slice exposure analysis
- Tumor-region diagnostic analysis

## Manuscript Sections

1. Abstract
2. Introduction
3. Related Work
4. Methods
5. Results
6. Discussion
7. Limitations
8. Conclusion
9. Data and Code Availability

## Results Policy

Only results from merged PRs #1–#4 may appear in the primary Results section.

PR #5 results may be included only as a clearly labeled robustness and exploratory extension section or as supplementary material.

## Claims Policy

- Claim patient-overlap leakage inflates performance and calibration.
- Do not claim a stable model-ranking reversal.
- Do not claim a patient-specific mechanism from tumor-mask analysis.
- Do not claim external generalization or clinical readiness.
