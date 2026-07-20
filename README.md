<div align="center">

# GeoWatch

### Satellite Change Detection and Spatial Intelligence Platform

GeoWatch detects and visualizes geographical changes between historical and recent satellite observations using deep learning and geospatial processing.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20GeoWatch-black?style=for-the-badge)](https://geowatch-app-825325.onrender.com)
[![API Health](https://img.shields.io/badge/API-Health%20Check-white?style=for-the-badge&logo=fastapi&logoColor=black)](https://geowatch-api-825325.onrender.com/health)

</div>

---

<p align="center">
  <a href="https://geowatch-app-825325.onrender.com">
    <img
      src="docs/assets/Geo-watch_Dashboard.png"
      alt="GeoWatch change detection dashboard"
      width="100%"
    />
  </a>
</p>

## Overview

GeoWatch processes bi-temporal Sentinel-2 imagery and generates pixel-level change predictions. Detected regions are converted into geospatial polygons, stored in PostGIS, served through FastAPI, and displayed in an interactive Next.js dashboard.

### Core Capabilities

- Bi-temporal satellite image comparison
- Multispectral Sentinel-2 processing
- Deep learning-based change detection
- Change-probability and binary-mask generation
- Raster-to-polygon vectorization
- PostGIS spatial storage
- FastAPI inference and spatial APIs
- Interactive before-and-after visualization
- Natural and enhanced rendering modes

---

## System Architecture

<p align="center">
  <img
    src="docs/assets/Artichitecture.png"
    alt="GeoWatch system architecture"
    width="100%"
  />
</p>

```text
Historical and Recent Satellite Images
                    ↓
        Validation and Preprocessing
                    ↓
        Siamese ResNet-18 U-Net
                    ↓
         Change Probability Raster
                    ↓
     Thresholding and Polygon Generation
                    ↓
                 PostGIS
                    ↓
              FastAPI Backend
                    ↓
          Next.js Analyst Dashboard
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Machine Learning | Python, PyTorch, ResNet-18, U-Net, ONNX Runtime |
| Geospatial Processing | Rasterio, GeoPandas, Shapely, OpenCV |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL, PostGIS |
| Frontend | Next.js, React, TypeScript, Leaflet |
| Infrastructure | Docker, Docker Compose, Render |
| Data | Sentinel-2 multispectral satellite imagery |

---

## Dashboard Visualizations

### Natural Rendering

<p align="center">
  <img
    src="docs/assets/Natural.png"
    alt="GeoWatch natural satellite rendering"
    width="100%"
  />
</p>

Natural rendering preserves the original visual appearance of the satellite observations for geographical context.

### Enhanced Rendering

<p align="center">
  <img
    src="docs/assets/Enhanced.png"
    alt="GeoWatch enhanced satellite rendering"
    width="100%"
  />
</p>

Enhanced rendering increases visual contrast to support closer inspection of model-generated change regions.

### Geographic Navigation

<p align="center">
  <img
    src="docs/assets/World_view.png"
    alt="GeoWatch wider geographic map view"
    width="100%"
  />
</p>

The interactive map supports zooming and navigation while maintaining the historical and recent imagery comparison layers.

---

## Deployment

| Service | Link |
|---|---|
| Web Application | [Open GeoWatch](https://geowatch-app-825325.onrender.com) |
| Backend Health | [Check API Status](https://geowatch-api-825325.onrender.com/health) |

The deployed demonstration displays model-generated change polygons for the selected Kokapet, Hyderabad region.

> The Hyderabad deployment is a qualitative demonstration. Detected polygons represent candidate change regions and should be reviewed together with the source satellite imagery.

---

## Processing Workflow

1. Load aligned historical and recent Sentinel-2 observations.
2. Validate bands, dimensions, metadata, and spatial alignment.
3. Normalize and divide the imagery into model-ready patches.
4. Predict pixel-level change probabilities.
5. Apply thresholding and connected-component filtering.
6. Convert detected regions into valid polygons.
7. Store polygon geometries and statistics in PostGIS.
8. Serve results through FastAPI.
9. Display spatial evidence in the Next.js dashboard.

---

## Project Structure

```text
geowatch/
├── artifacts/
├── configs/
├── data/
├── deploy/
├── docs/
│   ├── assets/
│   │   ├── Artichitecture.png
│   │   ├── Enhanced.png
│   │   ├── Geo-watch_Dashboard.png
│   │   ├── Natural.png
│   │   └── World_view.png
│   ├── datasets/
│   ├── backend-postgis-runbook.md
│   └── week1_data_pipeline_summary.md
├── experiments/
├── migrations/
├── notebooks/
├── reports/
├── src/
└── tests/
```

---

## Responsible Use

GeoWatch outputs are model-generated candidate change regions intended to support visual analysis. They should not be treated as independently verified real-world events or used for safety-critical, legal, or enforcement decisions without human validation.

---

## Author

**Deepika Kumari**

Computer Vision and AI Engineer

---

<div align="center">

[Live Demo](https://geowatch-app-825325.onrender.com) ·
[API Health](https://geowatch-api-825325.onrender.com/health)

</div>