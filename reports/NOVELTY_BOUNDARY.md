# Novelty boundary

## Not novel (do not claim)
1. Patient/slice-level leakage inflates accuracy (Yagis 2021; Bussola 2021; Rosenblatt 2024; OCT 2022; Kapoor & Narayanan 2023).
2. Brain-tumor MRI public corpora contain duplicate/patient contamination, including Cheng/Figshare lineage (Vangala et al., arXiv:2610.00421, 30 Sep 2026).
3. Architecture orderings can differ between evaluation conditions (same, arXiv:2610.00421; general: Maier-Hein 2018; Kapoor & Narayanan 2023).
4. Using a bootstrap to attach uncertainty to rankings (Maier-Hein 2018); patient-cluster bootstrap itself is standard.
5. "Our four architectures on this dataset" — a dataset/model combination is not a contribution.
6. The title "Patient Dependence and the Reliability of Model Selection in Medical Imaging Benchmarks" — **not verified as novel**; our search found no paper with that framing, but the search was non-exhaustive and the Lung-CT paper (#6) was not read.

## Plausibly distinctive but currently unproven
* Treating **model-selection reliability** (not accuracy inflation) as the estimand: how often an independent replication of the evaluation would reproduce the ordering / pairwise decisions, under dependent vs patient-disjoint evaluation.
* Evidence that dependence changes **confidence in** conclusions (CI width, resolved pairs) rather than only point estimates.
* Joint accuracy + calibration + ranking analysis with shared paired patient draws.

## Currently supported by our data (narrow, exploratory)
* On this dataset/seed the two protocols give different point orderings of the top two models (ResNet50 vs DenseNet121), but the paired difference CI includes 0 in both (BA: −0.0069 [−0.0160, 0.0015]; +0.0043 [−0.0015, 0.0104]).
* Exploratory interaction contrast: (ResNet50−DenseNet121) B−A = +0.0112 [0.0005, 0.0220] (BA); +0.0103 [0.0001, 0.0210] (macro-F1). Borderline, post hoc, 6 of 18 interaction contrasts exclude 0 without multiplicity control, single seed → **hypothesis-generating only**.
* DenseNet121 benefits least from patient overlap (ΔBA 0.032 vs 0.042–0.046): consistent with, not evidence of, architecture-dependent leakage exploitation.

## Boundary of the safe claim
"On the Cheng dataset, with one split and one training seed, protocol changed absolute performance and calibration for all four CNNs; the nominal best architecture differed between protocols, but the difference between the top two was statistically unresolved under both protocols." Everything stronger needs new evidence.

## Alternative explanation that must be excluded before claiming a protocol-induced ranking change
Ranking changes between A and B may simply reflect **training/split noise**: two independent patient-disjoint replications could also reorder ResNet50/DenseNet121. Without replicates, A vs B and A vs A′ cannot be told apart.
