# Training-seed replication — results
**Status: COMPLETE.** 40/40 runs verified (training seeds 1, 2; both models; protocols A and B; the historical 233-ID splits, seed 11; 5 folds). Reference = historical training seed 0. Tables: `tseed_seed_bootstrap_table.csv`, `tseed_seed_summary.csv`.

## Balanced accuracy, pooled OOF
| training seed | R50 A | R50 B | R50 B−A | D121 A | D121 B | D121 B−A |
|---|---|---|---|---|---|---|
| 0 (hist.) | .9386 | .9816 | +.0430 | .9456 | .9774 | +.0318 |
| 1 | .9365 | .9760 | +.0395 (.020,.063) | .9425 | .9799 | +.0374 (.019,.060) |
| 2 | .9441 | .9772 | +.0331 (.017,.052) | .9508 | .9814 | +.0306 (.015,.050) |
| mean ± SD | .9397 ± .004 | .9783 ± .003 | +.0385 ± .005 | .9463 ± .004 | .9795 ± .002 | +.0333 ± .004 |

## Ranking (ResNet50 − DenseNet121, BA)
Protocol A: −.0069, −.0060, −.0067 (all three seeds DenseNet121 ahead; every CI includes 0, e.g. seed 1 (−.018, .005)). Protocol B: +.0043, −.0039, −.0042 (sign flips with the training seed; all CIs include 0).

## Answers
* **Training-seed variability:** SD of BA across 3 seeds is .002–.004 per model/protocol, similar to split-seed SD; the B-protocol ranking sign changes with the training seed alone → ranking variation of this size is attributable to training randomness and is not interpretable as a model difference.
* B−A advantage reproduces for both models in both extra seeds (+.031 to +.040), CIs exclude 0.
* No multiple-comparison correction pre-specified; descriptive.
Limitations: training seed varied on a single split (seed 11); no training-seed resampling inside the bootstrap CIs.
Reproduce: `python3 experiments/extension/ext_driver.py; python3 experiments/extension/analyze_seeds.py tseed`
