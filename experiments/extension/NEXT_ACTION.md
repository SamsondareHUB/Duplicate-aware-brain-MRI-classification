# NEXT ACTION / resume (status snapshot; see EXPERIMENT_MANIFEST.csv for exact per-run status)

Completed: the full exposure seed-0 matrix (36 runs: ResNet50 + DenseNet121, folds 2 and 4, levels 0/25/50/75/100). Split-seed 101 has started. Everything else in the 172-run queue is NOT_STARTED. No result report has been written yet: only the analysis code exists.

Resume (idempotent; skips verified runs, one MPS job at a time):
    python3 experiments/extension/ext_driver.py          # queue order: expo s0 -> split 101 -> expo s1 -> split 202,303 -> tseed 1,2
Analysis once runs exist:
    python3 experiments/extension/analyze_seeds.py split|tseed
    python3 experiments/extension/analyze_exposure.py
    python3 experiments/extension/mask_diagnostics.py <run_ids>   # after training, on MPS
    python3 experiments/extension/build_manifest.py
Not started: mask diagnostics, the dose-response slope, Holm correction, and the SPLIT_SEED / TRAINING_SEED / PATIENT_EXPOSURE / TUMOR_REGION / SCIENTIFIC_CONCLUSIONS reports.
EXTERNAL_VALIDATION_DATA_BLOCKED: Kaggle "Brain Tumor MRI" repackages Cheng; BRISC 2025 has no patient IDs; the pituitary-136 set is single-class; BraTS is glioma-only.
