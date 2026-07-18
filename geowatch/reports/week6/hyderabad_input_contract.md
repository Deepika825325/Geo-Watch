# Hyderabad Qualitative Input Contract

The Hyderabad qualitative evaluation requires two spatially aligned Sentinel-2 observations covering the same geographic area.

## Required directory structure

data/qualitative/hyderabad/
├── before/
│   ├── B02.tif
│   ├── B03.tif
│   ├── B04.tif
│   └── B08.tif
└── after/
    ├── B02.tif
    ├── B03.tif
    ├── B04.tif
    └── B08.tif

## Input requirements

- Required ordered bands are B02, B03, B04 and B08.
- Every TIFF must contain exactly one raster band.
- Before and after rasters must have identical height and width.
- All rasters must use the same data type.
- All available CRS metadata must be consistent.
- All raster transforms must be consistent.
- Images must cover the same geographic extent.
- Images must be spatially aligned.
- Pixel resolution must be 10 metres.
- Sentinel-2 values must support reflectance scaling by 10,000.
- Ground-truth labels are not required.
- Hyderabad outputs are qualitative only.
- Checkpoint epoch remains 24.
- Probability threshold remains 0.76.
- Hyderabad imagery must not modify official OSCD test results.
