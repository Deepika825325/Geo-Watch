# OSCD Dataset Card

## Dataset

Onera Satellite Change Detection Dataset (OSCD).

OSCD contains 24 registered pairs of Sentinel-2 multispectral images
captured between 2015 and 2018. Each region provides 13 spectral bands
at native 10 m, 20 m and 60 m resolutions.

## Labels

Pixel-level urban-change labels are provided for:

- 14 training regions
- 10 testing regions

The primary labelled changes are urban developments such as new
buildings and roads. Original masks use 0 for unchanged pixels and
255 for changed pixels.

## GeoWatch intended use

OSCD is GeoWatch's primary standardized multispectral benchmark. It is
kept separate from the custom Hyderabad operational AOI. Metrics from
OSCD must be labelled as benchmark results and must not be presented as
Hyderabad operational performance.

GeoWatch model inputs will later use:

- B02
- B03
- B04
- B08
- B11
- B12

Band resampling and preprocessing are performed in a derived directory.
The downloaded raw dataset remains immutable.

## Limitations

- Only 24 geographic regions are available.
- Labels primarily represent urban changes.
- Sentinel-2 resolution limits detection of very small buildings.
- Seasonal, atmospheric and radiometric differences may resemble change.
- Dataset results do not establish global operational performance.

## Legal and redistribution note

The official project page does not provide a clear machine-readable
licence statement. Use the dataset for research and benchmarking, retain
the required citation, and verify redistribution or commercial-use terms
before publishing dataset copies.

## Citation

Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch and Yann
Gousseau. Urban Change Detection for Multispectral Earth Observation
Using Convolutional Neural Networks. IGARSS 2018.

DOI: 10.1109/IGARSS.2018.8518015

Official project page:
https://rcdaudt.github.io/oscd/
