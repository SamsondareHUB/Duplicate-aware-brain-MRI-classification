# Dataset Selection Report

## Status

**Candidate for team approval. No data have been downloaded or preprocessed.**

## Recommended Primary Dataset

**Jun Cheng Brain Tumor Dataset (Figshare, Version 8)**

- Official source: Figshare
- DOI: https://doi.org/10.6084/m9.figshare.1512427
- Current release shown on Figshare: Version 8 (2024-12-21)
- License: CC BY 4.0
- Modality: T1-weighted contrast-enhanced brain MRI
- Data unit: 2D MRI slices
- Total images/slices: **3,064**
- Total patients: **233**
- Image size: **512 x 512**
- Acquisition period reported in the original study: **2005–2010**
- Source institutions reported at dataset level: **Nanfang Hospital, Guangzhou, China** and **General Hospital, Tianjin Medical University, China**

## Class Labels

| Label | Tumor class | Number of slices |
|---|---|---:|
| 1 | Meningioma | 708 |
| 2 | Glioma | 1,426 |
| 3 | Pituitary tumor | 930 |
| **Total** |  | **3,064** |

**Important:** this dataset contains three tumor classes and does **not** contain a normal/no-tumor class.

## Patient and Source Metadata

### Patient metadata

**Available.** The dataset MATLAB records include a patient identifier (`cjdata.PID`). This is particularly useful for this project because it allows us to construct and verify patient-disjoint train/validation/test splits rather than relying only on image-level deduplication.

The records also include the tumor label, image data, tumor border, and tumor mask.

### Source/site metadata

The original publication states that the images were acquired at two hospitals, but a per-image hospital/site identifier is **not listed among the standard record fields** in the public dataset documentation that I found. Therefore, source-aware splitting should currently be treated as **not available / not verified** for this dataset.

## Existing Split Structure

The dataset distribution provides **5-fold cross-validation indices (`cvind.mat`)**. In the original study, the **233 patients were partitioned into five subsets**, with patients of each tumor category kept approximately balanced across folds. The paper explicitly states that patient-level partitioning was used so that slices from the same patient would not appear in both training and test sets.

This gives us an important reference protocol for the proposed leakage study:

1. conventional image-level random split;
2. exact-duplicate-aware split;
3. near-duplicate-aware split;
4. patient-disjoint split using patient IDs / the original patient-level fold structure.

## Why This Dataset Fits the Project

I recommend this dataset as the primary candidate because:

- it is public and relatively small, so the planned two- to three-week project is computationally manageable;
- it has multiple slices from the same patients, creating a real setting in which image-level random splitting can introduce subject dependence;
- patient IDs are available, so leakage can be checked against known subject identity rather than inferred only from perceptual similarity;
- the original study provides a patient-level five-fold protocol that can serve as a leakage-safe reference;
- the CC BY 4.0 license permits reuse with attribution;
- the dataset is widely used in brain-tumor MRI classification, making the evaluation issue practically relevant.

## Limitations Relevant to This Project

- It contains only 233 patients, despite having 3,064 slices; uncertainty should therefore be assessed at the patient/cluster level rather than treating all slices as independent observations.
- It has three tumor classes and no normal/no-tumor class.
- Per-image acquisition-site metadata do not appear to be available, so a true source-disjoint experiment may not be possible with this dataset alone.
- Exact and near-duplicate prevalence must still be measured after the dataset is approved and downloaded; it should not be assumed in advance.

## Recommendation

**Recommend selecting the Jun Cheng Figshare dataset as the primary dataset, pending team approval.**

If approved, the next step should be a frozen dataset audit that inventories image IDs, labels and patient IDs, then measures exact duplicates and near-duplicate candidates before any model training.

## References

1. Cheng, J. *brain tumor dataset*. Figshare. DOI: https://doi.org/10.6084/m9.figshare.1512427
2. Cheng J, Huang W, Cao S, et al. Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition. *PLOS ONE*. 2015;10(10):e0140381. https://doi.org/10.1371/journal.pone.0140381
3. Public README mirror documenting `cjdata.PID`, class labels, and `cvind.mat`: https://github.com/jasonnan2/Deep-Learning-MRI-tumor
