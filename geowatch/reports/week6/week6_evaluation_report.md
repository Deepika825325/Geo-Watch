# GeoWatch Week 6 Evaluation Report

## Completion status

The frozen official OSCD evaluation, quantitative failure analysis, and failure gallery are complete. Hyderabad evaluation is treated as an optional external qualitative demonstration and is never used for model selection or reported as labelled test performance.

- Week 6 status: `pending_input`
- Official evaluation invocations recorded: `2`
- Test-based tuning: `false`
- Threshold retuning after test access: `false`

## Frozen protocol

- Checkpoint epoch: `24`
- Checkpoint SHA-256: `61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94`
- Probability threshold: `0.76`
- Bands: `B02, B03, B04, B08`
- Patch size: `256`
- Stride: `256`
- Official test regions: `10`
- Official evaluated pixels: `3077936`

## Official test results

| Aggregation | Precision | Recall | F1 | IoU | Accuracy |
|---|---:|---:|---:|---:|---:|
| Micro | 0.405040 | 0.604959 | 0.485213 | 0.320318 | 0.933657 |
| Macro mean | 0.338395 | 0.580310 | 0.387690 | 0.266283 | 0.935332 |
| Macro median | 0.431233 | 0.605746 | 0.486096 | 0.321530 | 0.933212 |

- Change prevalence: `0.051683`
- Predicted change fraction: `0.077193`

## Validation-to-test generalization

- Precision delta: `-0.020238`
- Recall delta: `0.183612`
- F1 delta: `0.061910`
- IoU delta: `0.051843`
- Micro minus macro-mean F1: `0.097523`

The higher micro score than macro-mean score indicates substantial geographic variability. Performance is therefore reported with both global pixel aggregation and equal-weight regional aggregation.

## Per-region results

| Rank | Region | Precision | Recall | F1 | IoU | Prevalence | Prediction fraction |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | montpellier | 0.620804 | 0.816072 | 0.705170 | 0.544604 | 0.067945 | 0.089316 |
| 2 | lasvegas | 0.565506 | 0.768412 | 0.651527 | 0.483159 | 0.076731 | 0.104262 |
| 3 | brasilia | 0.452716 | 0.686761 | 0.545703 | 0.375235 | 0.025813 | 0.039158 |
| 4 | rio | 0.409749 | 0.777479 | 0.536664 | 0.366740 | 0.056870 | 0.107908 |
| 5 | dubai | 0.554188 | 0.478917 | 0.513810 | 0.345723 | 0.099220 | 0.085744 |
| 6 | chongqing | 0.513835 | 0.413730 | 0.458381 | 0.297337 | 0.072112 | 0.058063 |
| 7 | milano | 0.099138 | 0.551468 | 0.168063 | 0.091741 | 0.007954 | 0.044247 |
| 8 | saclay_w | 0.087649 | 0.660024 | 0.154748 | 0.083863 | 0.011414 | 0.085951 |
| 9 | norcia | 0.054932 | 0.406683 | 0.096790 | 0.050856 | 0.013224 | 0.097904 |
| 10 | valencia | 0.025428 | 0.243550 | 0.046049 | 0.023567 | 0.004445 | 0.042572 |

## Failure analysis

- Strongest region: `montpellier`
- Weakest region: `valencia`
- False-positive focus regions: `norcia, saclay_w, rio`
- False-negative focus regions: `dubai, chongqing, lasvegas`
- Gallery focus regions: `norcia, saclay_w, rio, dubai, chongqing, lasvegas`
- Gallery images: `12`
- Gallery review rows: `12`
- Gallery rows classified: `0`
- Gallery rows pending review: `12`

Failure-gallery selection is post-hoc diagnostic analysis only. It does not alter the checkpoint, threshold, preprocessing, or official metrics.

## Hyderabad qualitative status

Not executed because the eight aligned Hyderabad Sentinel-2 GeoTIFF bands are not available. No substitute city or synthetic result was used.

- Input root: `/home/tihan40904/Documents/Deepika/Geo-Watch/geowatch/data/qualitative/hyderabad`
- Present required files: `0`
- Missing required files: `8`

## Reproducibility and integrity

- Checkpoint SHA-256 verified: `61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94`
- Test result SHA-256 verified: `e45e6265608707e3d1e439737bca59e9fde10010dd8c73cca420d7881bb9a8f9`
- Failure analysis SHA-256 verified: `077bd0c56d200f3e7cc14019ab49b832aed020dfa72342908f4f7a6539e024d9`
- Gallery entries verified: `12`
- Official result overwrite protection: `enabled`
- Official test data used for tuning: `false`

## Limitations

- Regional performance varies substantially.
- Several low-prevalence regions show strong false-positive behaviour.
- Accuracy is secondary because unchanged pixels dominate the dataset.
- The Hyderabad external demonstration cannot be claimed until genuine aligned imagery is available.
- Repeated execution produced identical console metrics, but only the currently hashed JSON artifact is treated as the frozen result.
