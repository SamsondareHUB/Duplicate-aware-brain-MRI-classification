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

## Multi-model continuation status, 2026-10-09 (incomplete)

- Frozen input commit: `919d6f0fb3ca462983d6df6ec06846e41abd9645`; all training uses the original MPS recipe with batch size 32 and 10 epochs.
- Verified new runs: MobileNetV2 10/10, EfficientNet-B0 10/10, DenseNet121 8/10. Each complete model has 3,064 out-of-fold predictions in each of the original A and B arms. DenseNet121 has 2,421 in each arm; A and B fold 5 remain untrained.
- DenseNet121 folds ran in separate, sequential MPS processes with process-scoped `caffeinate`, live RAM/swap/thermal checks, and at least five minutes of cooling. Each of the eight completed DenseNet121 folds was validated, committed, pushed, and checked against the remote branch before the next fold.
- The first completed DenseNet121 fold took 395 seconds. An initial attempt at A fold 5 was safely interrupted during epoch 5 when swap grew more than 0.5 GiB above the original 4.365 GiB baseline. The remaining eight completed DenseNet121 fold outputs were preserved. The guarded retry did not start: swap did not return within the original safety cutoff after two hours of recovery. No thermal warning was recorded.
- The 30-run four-model analysis, scientific report, final PR, and task-specific local cleanup are pending until the two remaining folds can run safely. No partial four-model ranking is reported as final.
- Validation at this checkpoint: all 28 new run pairs passed integrity checks; `python3 -m pytest -q tests` passed 24 tests. The temporary dataset and workspace are retained to resume without retraining completed folds.

## Multi-model completion, 2026-10-09

- The preceding section is a historical safety checkpoint. The final status is **30/30 new runs complete**: MobileNetV2 10/10, EfficientNet-B0 10/10, and DenseNet121 10/10. The original ResNet50 out-of-fold predictions were reused without retraining.
- DenseNet121 A fold 5 and B fold 5 each ran in its own fresh MPS process and took 406 and 404 seconds, respectively. Both completed 10 epochs at the frozen physical batch size 32. No microbatching, gradient accumulation, mixed precision, or activation checkpointing was used.
- The final two folds used a guarded 8 GiB available-RAM preflight, at least 35% free macOS memory pressure, MPS availability, and no thermal warning. Swap was logged for diagnosis but was not a stop condition. During training the monitor retained the 5 GiB RAM, 25% free memory-pressure, thermal, and responsiveness stops. Across 52 monitoring checks in these two runs, the minimum available RAM was 5.33 GiB, minimum free memory pressure 30%, maximum check latency 0.38 seconds, and no thermal warning or safety stop occurred.
- Every new run was individually validated and published to the working branch. Each of the four models has 3,064 unique out-of-fold predictions in each protocol. The final report and figures are in `reports/MULTIMODEL_RESULTS.md` and `results/multimodel/figures/`.
- The joint analysis used 5,000 class-stratified patient-cluster bootstrap draws shared across all models and both protocols. `scripts/verify_multimodel.py` independently confirmed all 40 headline point metrics and 24 selected bootstrap replicate checks. The repository test suite passed 24 tests.
