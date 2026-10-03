# Project Plan

## Project Title

Duplicate-Aware Evaluation of Transfer-Learning Models for Brain-MRI Tumor Classification

## Project Description

This project investigates whether exact and near-duplicate brain-MRI images can leak across training and test sets and artificially improve reported model performance.

We will compare conventional random data splitting with stricter duplicate-aware evaluation protocols. We will examine whether duplicate leakage affects classification accuracy, calibration, confidence, error patterns, and the ranking of transfer-learning models.

## Research Question

How much do exact and near-duplicate image leakage affect the performance, confidence calibration, error patterns, and relative rankings of transfer-learning models for brain-MRI tumor classification?

## Main Objective

Build a reproducible duplicate-aware evaluation pipeline for brain-MRI tumor classification.

## Evaluation Protocols

- Random stratified train, validation, and test split
- Exact-duplicate-aware split
- Near-duplicate-aware split
- Source-level or patient-level split, if dataset metadata allow it

## Planned Models

- ResNet50
- MobileNetV2
- EfficientNet-B0
- DenseNet121

## Metrics

- Accuracy
- Balanced accuracy
- Macro-F1 score
- Per-class precision, recall, and F1-score
- Confusion matrix
- Expected Calibration Error
- Brier score
- Reliability diagram

## Expected Contribution

This project will show whether duplicate and near-duplicate leakage can inflate reported performance, distort model confidence, change error patterns, or change which transfer-learning model appears best.

## Timeline

Two weeks for dataset auditing, duplicate detection, split creation, model training, evaluation, documentation, and a first research-paper draft.
