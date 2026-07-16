# GeoWatch

GeoWatch is a bi-temporal satellite-image change-detection system built using
Sentinel-2 imagery and a Siamese U-Net architecture.

## Current status

Week 1 — Data Foundation

- [x] Repository structure initialized
- [x] Central data configuration created
- [x] Dataset manifest schema created
- [ ] Area of interest defined
- [ ] Sentinel-2 scenes discovered
- [ ] Bi-temporal imagery downloaded
- [ ] SCL cloud filtering implemented
- [ ] Imagery reprojected and tiled
- [ ] Geographic train/validation/test split created

## Data policy

- Files under `data/raw/` are immutable source imagery.
- Derived outputs are written under `data/processed/`.
- Dataset artifacts are tracked in `data/manifest.csv`.
- Train, validation, and test partitions are separated geographically.
