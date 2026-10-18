# Split-seed replication — results
**Status: COMPLETE.** 60/60 runs verified (ResNet50, DenseNet121 × protocol A patient-disjoint / B matched image-level × split seeds 101, 202, 303 × 5 folds; training seed 0; frozen PR #1/#2 recipe). Reference = historical split seed 11. Splits: `splits/split_{A,B}_seed*.csv`, validated in `splits/split_validation_ext.json` (A patient-disjoint; B margin-matched to A). Tables: `split_seed_bootstrap_table.csv` (all metrics, patient-cluster bootstrap CIs, 5,000 draws, seed 2026), `split_seed_summary.csv`.

## Balanced accuracy, pooled out-of-fold (95% bootstrap CI within replicate)
| split seed | R50 A | R50 B | R50 B−A | D121 A | D121 B | D121 B−A |
|---|---|---|---|---|---|---|
| 11 (hist.) | .9386 | .9816 | +.0430 (.024,.067) | .9456 | .9774 | +.0318 (.014,.053) |
| 101 | .9460 | .9831 | +.0370 (.020,.058) | .9489 | .9838 | +.0349 (.019,.056) |
| 202 | .9394 | .9816 | +.0422 (.023,.066) | .9424 | .9825 | +.0400 (.022,.063) |
| 303 | .9328 | .9799 | +.0471 (.030,.068) | .9432 | .9852 | +.0419 (.026,.061) |
| mean ± SD (4) | .9392 ± .005 | .9816 ± .001 | +.0423 ± .004 | .9450 ± .003 | .9822 ± .003 | +.0372 ± .005 |

Macro-F1, ECE, Brier follow the same pattern (B−A, 4 replicates): macro-F1 +.042 (R50), +.037 (D121); ECE −.027 (R50), −.018 (D121); Brier −.062 (R50), −.051 (D121); every replicate has the same sign.

## Model ranking (ResNet50 − DenseNet121, BA)
| seed | protocol A | protocol B |
|---|---|---|
| 11 (hist.) | −.0069 (−.016, .002) | **+.0043** (−.002, .010) |
| 101 | −.0028 (−.011, .005) | −.0007 (−.007, .005) |
| 202 | −.0031 (−.016, .008) | −.0008 (−.006, .005) |
| 303 | −.0104 (−.022, −.0001) | −.0053 (−.012, .001) |

## Answers
* **Does the image-level advantage persist?** Yes. B−A is positive for both models in 8/8 new and 2/2 historical replicates; every CI excludes 0; range +.031 to +.047 BA.
* **Does the ranking reversal replicate?** No. Historically DenseNet led under A and ResNet50 led under B (an inversion, itself not significant). In the 3 new split seeds DenseNet121 ≥ ResNet50 under both protocols; the B-protocol sign flips back (reversal replicated 0/3). All ranking differences (|Δ| ≤ .010) have CIs that include 0, except seed 303 protocol A (CI upper −.0001, borderline, uncorrected). The model ranking is not distinguishable from noise; the historical reversal should be regarded as non-reproducible.
* Between-split-seed SD of BA is .001–.005, comparable to or smaller than within-replicate bootstrap SE (~.01).
* Multiple-comparison correction: none was pre-specified for this family (all comparisons descriptive/replication); no p-values are claimed as confirmatory.

Limitations: training seed 0 only for these runs (training-seed effect in `TRAINING_SEED_RESULTS.md`); 233 PIDs treated as 233 individuals (author confirmation); dataset has 3 tumor classes, ~3,064 slices.
Reproduce: `python3 experiments/extension/make_split_seeds.py; python3 experiments/extension/ext_driver.py; python3 experiments/extension/analyze_seeds.py split`
