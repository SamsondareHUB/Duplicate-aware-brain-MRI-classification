# NEXT ACTION (status: all 172 runs verified COMPLETE; analyses and reports written)
Nothing is running. Reports: SPLIT_SEED / TRAINING_SEED / PATIENT_EXPOSURE / TUMOR_REGION / EXTERNAL_VALIDATION RESULTS, SCIENTIFIC_CONCLUSIONS, EXPERIMENT_MANIFEST.csv.
Reproduce analyses (CPU, minutes): `python3 experiments/extension/analyze_seeds.py split|tseed; analyze_exposure.py; analyze_dose.py; analyze_mask.py` (mask_diagnostics.py needs MPS + data/ckpt/, not committed).
Recommended next science (not started): (1) exposure on folds 1,3,5 and more training seeds; (2) patient-identity probe (embedding retrieval) for the mechanism; (3) independent patient-identified multi-class MRI cohort — EXTERNAL_VALIDATION_DATA_BLOCKED (see EXTERNAL_VALIDATION_RESULTS.md).
Merge decision for PR #5 and PR #4 is the owner's; neither has been merged.
