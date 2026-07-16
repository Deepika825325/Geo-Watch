# GeoWatch Week 3 Completion Report

**Status:** Complete  
**Completed:** 2026-07-16T16:19:00+05:30

## Architecture

- Weight-shared Siamese ResNet-18 encoder
- Four-band and six-band Sentinel-2 support
- ImageNet first-convolution adaptation
- Absolute-difference fusion at five feature scales
- GroupNorm U-Net decoder
- Full-resolution one-channel raw logits
- Date-order symmetry verified

## Dataset

- Dataset: OSCD official training regions
- Audit region: `abudhabi`
- Audit patch: `abudhabi_r00_y00000_x00000_h64_w64`
- Before shape: `(1, 4, 64, 64)`
- After shape: `(1, 4, 64, 64)`
- Mask shape: `(1, 1, 64, 64)`
- Binary mask values: `[0.0, 1.0]`
- Patch change fraction: `0.01147461`
- OSCD test labels requested: **No**

## Model integration

- Output shape: `(1, 1, 64, 64)`
- Output values finite: **Yes**
- Shared encoder: **Yes**
- Date-order symmetric: **Yes**
- Maximum swap error: `0.0`
- Total parameters in audit model: `11,521,897`
- Trainable parameters: `11,521,897`

## Verification

- Required files: Passed
- Python compilation: Passed
- Automated tests: `27 passed in 12.55s`
- Dependency consistency: `No broken requirements found.`
- Git whitespace check: Passed

## Evaluation discipline

No OSCD test-label path was supplied to the dataset or model audit.
Week 3 performed architecture and training-data validation only. Quantitative
benchmark comparison remains governed by the frozen Week 2 OSCD baseline and
the later held-out evaluation protocol.
