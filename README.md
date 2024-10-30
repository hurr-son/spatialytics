

# yolo-geospatial-implentations

This repo is a collection of scripts and experiments for running YOLO-based object detection on geospatial imagery. It’s a bit of a work-in-progress, mostly aimed at testing different ways to detect objects in raster images (TIFFs, Cloud Optimized GeoTIFFs, and STAC catalog images) using YOLO. Both standard bounding boxes (BBox) and Oriented Bounding Boxes (OBB) are supported.

## What’s Here

- **GeoInference Class**: A script that slides a window across images to detect objects using a YOLO model.
- **Flexible Input Options**: Works with TIFF images, Cloud Optimized GeoTIFFs, or images in a STAC catalog.
- **Geospatial Outputs**: Converts pixel detections to geographic coordinates and saves them to a Parquet file.

## Getting Started

To play around with this, install the main dependencies:

```bash
pip install -r requirements.txt
```

## How to Use

1. **Initialize the `GeoInference` Class**:
   ```python
   from pathlib import Path
   from yolo_geospatial_implementations import GeoInference

   geo_inference = GeoInference(
       model_path=Path("path/to/your/yolo/model.pt"),
       class_yaml_path=Path("path/to/class_yaml.yaml"),
       output_path=Path("path/to/output/detections.parquet"),
       window_size=1280,       # Window size (default 1280)
       stride=640,             # Step size (default 640)
       conf_threshold=0.25,    # Confidence threshold
       iou_threshold=0.3,      # IoU threshold
       detection_type="obb"    # Set to "obb" or "bbox"
   )
   ```

2. **Run Inference**:
   Pick your input type: local TIFF, COG, or STAC catalog.
   ```python
   geo_inference.run(tif_path="path/to/image.tif")
   # or
   geo_inference.run(stac_catalog_url="path/to/stac/catalog.json")
   # or
   geo_inference.run(cog_url="https://example.com/image.cog.tif")
   ```

3. **Results**:  
   The detections (with geographic coordinates) will be saved as a Parquet file in the path you specified.

