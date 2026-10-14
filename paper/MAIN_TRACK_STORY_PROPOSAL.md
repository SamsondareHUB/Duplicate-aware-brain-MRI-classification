# Main-track story proposal (candid)

**Verdict now: REQUIRES_SCIENTIFIC_REFORMULATION.** Current evidence = verified replication of a known effect (patient overlap inflates accuracy, here also apparent calibration) on one dataset, plus an *unresolved* top-two ranking change. Oral-level: not justified. This document does not assert the framing is novel (prior art: `reports/MAIN_TRACK_PRIOR_ART.md`; closest: arXiv:2610.00421, and an unread 2026 lung-CT protocol paper).

## Candidate framing (to be earned, not claimed)
*Patient dependence and the reliability of model selection in medical-imaging benchmarks.* Question: when evaluation data are dependent (multiple slices per patient), do researchers' **model-selection conclusions** (which architecture is best, which pairs differ) become less reliable or more confidently wrong — beyond the known accuracy inflation?

Why the present data cannot answer it: (i) one split seed and one training seed → no sampling distribution of the ranking; (ii) no patient-disjoint replicate to separate protocol effect from seed noise; (iii) the key pair is unresolved in both protocols; (iv) the interaction contrast (+0.011 BA [0.0005, 0.022]) is borderline and post hoc.

## Smallest high-value upgrade (recommend ONE; needs team approval; no training has been started)
**E1 — Replicated-split ranking stability with a seed-noise control.**
* **Hypothesis (pre-specify):** the probability that the observed best-model ordering of the top-two architectures reproduces on an independent evaluation replicate is *lower or equal* under patient-disjoint than image-level evaluation, while the claimed/nominal confidence (bootstrap interval excluding 0) is *higher* under image-level — i.e. dependence yields over-confident selection. (Directional alternative: no difference → report as null; both outcomes publishable as a diagnostic study.)
* **Primary estimand:** for protocol P ∈ {patient-disjoint, image-level} and each model pair, the replicate-level *ordering agreement* (Kendall τ-b between two independent replicates' model scores) and the *false-resolution rate*: fraction of replicates where the cluster-bootstrap CI excludes 0 for an ordering that disagrees with the pooled-replicate ordering.
* **Data:** existing Cheng dataset (no new data). New splits: 3 additional patient-grouped (stratified group) 5-fold partitions with matched fold sizes, and 3 additional matched image-level partitions (seeds ≠ 11). Existing A and B count as replicate 0 → 4 replicates per protocol.
* **Models/runs:** 4 models × 5 folds × 2 protocols × 3 new seeds = **120 runs**. Same frozen recipe, no tuning. (Optional cheaper variant: 3 models × 2 new seeds = 60 runs.) Training seed fixed to 0, and one extra training seed on existing splits for ResNet50/DenseNet121 (20 runs) to separate training from split noise — optional.
* **Controls:** A-vs-A′ (patient-disjoint vs another patient-disjoint replicate) is the null for "protocol effect on ranking"; matched fold sizes/class counts; same shared bootstrap; pre-registered contrasts (top-two pair; all-pairs τ) and Holm correction.
* **Statistical test:** permutation/bootstrap over replicates of the difference in mean τ between protocols; hierarchical bootstrap (patients within replicates). With 4 replicates per protocol the power is limited — state this; report effect sizes and intervals rather than p-values.
* **Success criterion (pre-specified):** protocol difference in mean τ (image-level − patient-disjoint) with 95% interval excluding 0, **and** false-resolution rate difference ≥ 0.10. **Failure:** intervals include 0 → the ranking change in the present data is consistent with seed noise; reframe as a negative/robustness result.
* **Compute:** ≈ 4.4 min per run on the existing Mac (30 runs = 133 min) → **≈ 9 h for 120 runs**, serial, no paid compute. Wall-clock 1–2 days with the memory-safety guard.
* **Reviewer concern addressed:** "one seed, one dataset, a protocol contrast rather than a finding" (partly: still one dataset).
* **Does not address:** second dataset (a patient-identified independent replication would be the next step, e.g. another multi-slice MRI/CT dataset with subject IDs), causal identification of overlap.

## Free analyses available now from existing predictions (already done, exploratory)
Patient-weighted vs slice-weighted (same orderings), metric sensitivity (same top model for BA/F1/acc/Brier/NLL), per-fold consistency, naive vs cluster CI, interaction contrasts. All in `results/audit/`. Further free options: a stem-grouped re-bootstrap of the original folds; error overlap analysis between models.

## Not recommended now
More datasets or hundreds of runs; new architectures; a new method; recalibration; claims of clinical relevance.

## Honest fallback positioning
If E1 is not run: a rigorous empirical audit / reproducibility-style short paper or workshop/dataset-and-benchmarks-track-style contribution (with the audit, tests, released OOF predictions and paired-bootstrap protocol). Report that the top-two ranking is unresolved.
