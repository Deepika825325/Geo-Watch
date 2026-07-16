# GeoWatch Week 1 Data Pipeline Summary

**Status:** COMPLETE

**Completion:** 9/9 stages (100%)

## Operational Hyderabad dataset

- Raw Sentinel-2 rasters: 14
- Aligned rasters: 14
- Joint valid-pixel fraction: 99.23%
- Accepted patches: 25
- Split: 15 train / 5 validation / 5 test
- Patch size: 256 × 256
- Input channels: 6

## OSCD benchmark

- Image regions: 24
- Training regions: 14
- Testing regions: 10
- Validated band files: 624
- Archive checksum verification: True

## Pipeline stages

| Stage | Status |
|---|---|
| scene_pair_selection | passed |
| raw_band_acquisition | passed |
| raw_raster_validation | passed |
| aoi_alignment | passed |
| cloud_masking | passed |
| paired_patch_generation | passed |
| geographic_split | passed |
| oscd_benchmark_acquisition | passed |
| dataset_lineage_manifest | passed |

## Week 1 outcome

The acquisition, validation, spatial alignment, cloud masking, patch generation, geographic splitting and public benchmark acquisition stages are complete.

Week 2 begins with exploratory data analysis, radiometric statistics, change-label imbalance analysis and a classical image-difference baseline.
