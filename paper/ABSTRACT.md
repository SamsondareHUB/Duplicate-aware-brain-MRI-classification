# Abstract

## Background

Brain-MRI tumor-classification studies often evaluate models using image-level splits, although multiple slices from the same patient may appear in both training and test partitions. This can inflate apparent performance and distort calibration.

## Objective

To quantify how patient-overlap leakage affects performance, calibration, error patterns, and model rankings in brain-MRI tumor classification.

## Methods

We audited the Jun Cheng Brain Tumor Dataset, which contains 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients with meningioma, glioma, and pituitary tumor. We evaluated transfer-learning models under matched image-level and patient-disjoint five-fold protocols. We assessed accuracy, balanced accuracy, macro-F1, calibration, error patterns, and model rankings using patient-level bootstrap uncertainty.

## Results

[Insert the exact main results from PR #1–#4.]

## Conclusions

Patient-overlap leakage can materially inflate apparent performance and calibration in brain-MRI tumor classification. Evaluation protocols should preserve patient-level separation and report uncertainty alongside performance.
