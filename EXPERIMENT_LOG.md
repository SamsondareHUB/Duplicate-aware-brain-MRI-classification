# Experiment Log

Use one entry for every completed or failed experiment.

---

## Experiment Template

### Experiment ID

Example: EXP-001

### Date

YYYY-MM-DD

### Objective

What question does this experiment answer?

### Dataset

- Dataset name:
- Dataset version/source:
- Number of images:
- Classes:
- Notes:

### Split Protocol

- Random split / exact-duplicate-aware split / near-duplicate-aware split / source-aware split
- Train images:
- Validation images:
- Test images:
- Duplicate clusters crossing splits: 0

### Model Configuration

- Model:
- Pretrained weights:
- Input size:
- Batch size:
- Optimizer:
- Learning rate:
- Epochs:
- Early stopping:
- Random seed:

### Results

- Accuracy:
- Balanced accuracy:
- Macro-F1:
- ECE:
- Brier score:
- Notes:

### Decision

- Keep, repeat, modify, or reject:
- Reason:


---

## EXP-001 - Patient-overlap sanity experiment (ResNet50)

### Date
2026-10-03 to 2026-10-04

### Objective
Measure how much image-level evaluation (patient overlap) changes performance and calibration relative to the original
patient-disjoint folds, with fold sizes and fold x class counts matched.

### Dataset
- Dataset name: Cheng et al. brain tumor dataset (Figshare 1512427 v8, CC BY 4.0)
- Number of images: 3,064 (233 patients)
- Classes: meningioma 708, glioma 1426, pituitary 930
- Notes: 0 exact duplicates; near-duplicate threshold not frozen (see reports/DATASET_AUDIT.md)

### Split Protocol
- Arm A: original patient-disjoint folds; Arm B: image-level stratified, seed 11, matched sizes and class counts
- Train images per fold: 2,385-2,522; test images per fold: 542/679/572/628/643 (both arms)
- Duplicate clusters crossing splits: n/a (no exact duplicates); patient overlap: A 0 patients, B 224 of 233 patients in >1 test fold

### Model Configuration
- Model: ResNet50; pretrained weights: torchvision IMAGENET1K_V1; input size 224; batch size 32
- Optimizer: AdamW, lr 1e-4, weight decay 1e-4, cosine; epochs 10 (fixed); early stopping: none
- Random seed: training seed 0, image-level split seed 11; device: MPS

### Results (pooled out-of-fold; B - A with 95% paired patient-bootstrap CI)
- Accuracy: A 0.9465, B 0.9853
- Balanced accuracy: A 0.9386, B 0.9816, delta +0.0430 (+0.0237, +0.0673)
- Macro-F1: A 0.9394, B 0.9830, delta +0.0436 (+0.0243, +0.0673)
- ECE: A 0.0321, B 0.0034, delta -0.0287 (-0.0456, -0.0134)
- Brier: A 0.0879, B 0.0224, delta -0.0655 (-0.1007, -0.0376)
- Notes: details in reports/SANITY_EXPERIMENT.md; commit SHA of frozen inputs e78a2b7

### Decision
- Keep / expand: GO_TO_MULTI_MODEL (pre-specified rule)
- Reason: paired CIs exclude 0 for all primary and calibration metrics; next, more architectures, split seeds and training seeds, then ranking stability
