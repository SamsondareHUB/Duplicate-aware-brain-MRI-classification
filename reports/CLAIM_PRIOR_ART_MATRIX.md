# Claim × prior-art matrix

Status: NOVEL? = novelty relative to the sources in `MAIN_TRACK_PRIOR_ART.md` (non-exhaustive search). EVID = support in our committed data.

| # | Claim | Closest prior art | NOVEL? | EVID | Disposition |
|---|---|---|---|---|---|
| 1 | Image-level evaluation inflates accuracy vs patient-disjoint | Yagis 2021; Bussola 2021; OCT 2022; lung-CT 2026 | **No** | Strong (ΔBA 0.032–0.046; CIs exclude 0) | Report as replication/confirmation |
| 2 | Image-level evaluation makes calibration look better (ECE, Brier) | Albzour 2026 (ECE/Brier, abstract-level); arXiv:2610.00421 (ECE only, stable) | Weak/partial | Strong (ECE Δ −0.019…−0.029; Brier Δ −0.050…−0.066), but ECE of near-perfect classifiers is dominated by accuracy | Report; do not oversell |
| 3 | Exact duplicates absent in Cheng dataset | arXiv:2610.00421 (duplicates in Kaggle copies, not necessarily Figshare) | Not claimed novel | CONFIRMED from hashes in manifest (0 duplicate hashes) | PASS |
| 4 | Same-patient slices are dependent observations, not near-duplicates | Vangala et al. (duplicate criterion conservative) | No | Confirmed | PASS |
| 5 | Best architecture changes between protocols (point estimates) | Vangala et al. (ordering not preserved across conditions) | Partly | Observed, **unresolved** (CIs include 0) | Report as descriptive only |
| 6 | Statistically established reversal of best architecture | – | would be novel | **Not supported** | UNSUPPORTED |
| 7 | Evaluation dependence changes the reliability of model-selection conclusions | Maier-Hein 2018 (general ranking stability); Kapoor & Narayanan 2023 | Possibly | **Insufficient** (one seed; no A-vs-A′ control) | NEEDS_NEW_EVIDENCE |
| 8 | Architecture-dependent benefit from patient overlap (DenseNet121 smallest) | none found | Possibly | Exploratory, borderline, not multiplicity-corrected | NEEDS_NEW_EVIDENCE |
| 9 | Paired class-stratified patient-cluster bootstrap with shared draws | standard methodology | No | Verified implementation | PASS (method, not contribution) |
| 10 | General principle / diagnostic for benchmark design | Kapoor & Narayanan taxonomy | Open | none | REFORMULATE |
| 11 | Per-image causal effect of patient overlap | – | – | Not identified (folds/training sets differ) | UNSUPPORTED |
