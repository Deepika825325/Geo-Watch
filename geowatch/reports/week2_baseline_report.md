# GeoWatch Week 2 — EDA and Classical Baselines

## Evaluation protocol

- Quantitative evaluation uses only the labelled OSCD benchmark.
- The unlabelled Hyderabad AOI is excluded from all reported metrics.
- OSCD's official 14-region training and 10-region testing split is preserved.
- Thresholds, PCA, scaling and clustering are fitted using training imagery only.
- Precision, recall, F1 and IoU are reported for the positive change class.
- Overall pixel accuracy is retained only as a secondary diagnostic.

## Dataset analysis

- Overall changed-pixel fraction: **3.2184%**
- Training changed-pixel fraction: **2.2976%**
- Test changed-pixel fraction: **5.1683%**
- Overall unchanged-to-change ratio: **30.07:1**

The severe class imbalance explains why overall pixel accuracy is not an appropriate headline metric. A method can classify most unchanged pixels correctly while still performing poorly on the change class.

## OSCD test results

| Baseline | Precision | Recall | F1 | IoU | Accuracy* | Predicted change |
|---|---:|---:|---:|---:|---:|---:|
| Band Difference + Otsu | 0.115846 | 0.513713 | 0.189058 | 0.104398 | 0.772232 | 22.92% |
| CVA + PCA + K-Means | 0.060915 | 0.669217 | 0.111665 | 0.059134 | 0.449695 | 56.78% |

\*Accuracy is shown only as a secondary diagnostic because unchanged pixels dominate the dataset.

## Result interpretation

**Band Difference + Otsu** is the stronger Week 2 baseline by both F1 and IoU.

- F1 improvement over CVA + PCA + K-Means: **69.31%** relative improvement, or **7.74 percentage points**.
- IoU improvement over CVA + PCA + K-Means: **76.54%** relative improvement, or **4.53 percentage points**.
- CVA + PCA + K-Means achieved higher recall (0.669217) but only 0.060915 precision, indicating substantial over-prediction.
- The first three PCA components retained 98.77% of training-vector variance.
- The two K-Means clusters differed in mean CVA magnitude by only 1.18%, suggesting weak separation between unchanged and changed pixels under the unsupervised cluster rule.

## Baseline limitations

- Radiometric and seasonal differences can be mistaken for real land-cover change.
- Neither baseline learns spatial context, object shape or semantic land-use patterns.
- No morphological cleanup was applied, preserving a simple and reproducible comparison.
- The methods operate on four native 10 m bands: B02, B03, B04 and B08.

## Week 3 target

The Siamese U-Net must exceed the strongest classical test baseline of **F1=0.189058** and **IoU=0.104398** while producing more spatially coherent change masks.

## Source artifacts

- EDA: `reports\week2\eda\oscd_region_statistics.csv`
- Otsu report: `reports\week2\baselines\band_diff_otsu\band_diff_otsu_report.json`
- CVA report: `reports\week2\baselines\cva_pca_kmeans\cva_pca_kmeans_report.json`
