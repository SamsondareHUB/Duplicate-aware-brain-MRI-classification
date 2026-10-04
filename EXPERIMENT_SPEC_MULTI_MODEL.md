# Multi-Model Experiment Specification

## Objective

Extend the frozen PR #1 protocol to additional transfer-learning models and determine whether patient-overlap leakage changes model performance, calibration, error patterns, and model rankings.

## Frozen Rules

- Use the exact preprocessing pipeline from PR #1.
- Use the exact training configuration from PR #1.
- Use the same random seeds as PR #1.
- Use the same patient-disjoint five-fold assignment.
- Do not tune hyperparameters during this phase.
- Save predictions for every run.

## Models

1. ResNet50 — already completed in PR #1
2. MobileNetV2
3. EfficientNet-B0
4. DenseNet121

## Protocols

- Protocol A: Matched image-level evaluation
- Protocol B: Original patient-disjoint five-fold evaluation

## Required Metrics

- Accuracy
- Balanced accuracy
- Macro-F1
- Per-class precision, recall, and F1-score
- Confusion matrix
- Expected Calibration Error
- Brier score
- Reliability diagram

## Required Prediction Fields

Every prediction file must include:

- Image ID
- Patient ID
- Fold ID
- True label
- Predicted label
- Predicted class name
- Confidence score

## Main Analysis

For each model, compare:

- Image-level balanced accuracy versus patient-disjoint balanced accuracy
- Image-level macro-F1 versus patient-disjoint macro-F1
- Image-level ECE versus patient-disjoint ECE
- Model ranking under both protocols
- Patient-level bootstrap confidence intervals for performance differences

## Completion Criteria

The experiment is complete when all four models have results under both protocols and every reported metric can be regenerated from saved predictions.
