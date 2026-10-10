# Results

## Dataset and leakage audit

The Jun Cheng Brain Tumor Dataset contained 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients. The three classes were meningioma (708 slices), glioma (1,426 slices), and pituitary tumor (930 slices).

The audit found [0] exact duplicate groups and [briefly state the near-duplicate finding and method/threshold]. The principal evaluation concern was patient overlap: multiple slices from the same patient could appear in different partitions under image-level splitting. The original five-fold assignment was validated as patient-disjoint under the file-order `cvind` mapping; however, this mapping was inferred empirically because the dataset documentation does not explicitly specify it.

## Image-level versus patient-disjoint evaluation

Across the matched ResNet50 sanity experiment, image-level evaluation produced a balanced accuracy of 0.9816, compared with 0.9386 under patient-disjoint evaluation. The image-level minus patient-disjoint difference was 0.0430 (95% patient-bootstrap CI: 0.0237 to 0.0673). Macro-F1 differed by 0.0436. Calibration also differed between protocols [insert metric names and exact values from PR #1].

In the multi-model experiments, [summarize the consistent performance and calibration pattern across all four architectures, citing exact results from the merged tables]. Report the model-wise estimates, uncertainty intervals, and the number of runs/folds used in Table [X].

## Model rankings

The highest observed image-level performance was obtained by [model], while [model] ranked first under patient-disjoint evaluation. Thus, the observed ranking changed between evaluation protocols. However, the difference between the top two models under patient-disjoint evaluation was not statistically conclusive ([report estimate, interval, or test result]). We therefore do not conclude that one architecture is definitively superior.

## Calibration

For [name the calibration metric or metrics], [describe the direction and magnitude of the difference between image-level and patient-disjoint evaluation using exact values]. The results indicate that confidence calibration depended on the evaluation protocol as well as on the model. Include reliability diagrams and report the calibration metric’s definition and aggregation method.

## Error patterns

Under patient-disjoint evaluation, the main changes in errors were [insert findings from the error analysis]. Class-specific results showed [insert relevant per-class recall/F1 findings]. These results should be interpreted as performance on this dataset and split design, not as evidence of clinical readiness.

## Summary of findings

The experiments show that evaluation protocol materially affected measured performance and calibration. The observed model ranking also changed between image-level and patient-disjoint evaluation, although the difference between the top two patient-disjoint models was not statistically conclusive. Exact and near-duplicate leakage were not identified as the main source of inflation in this dataset; patient overlap was the more important concern.
