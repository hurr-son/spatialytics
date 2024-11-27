# spatialytics

This repository is a collection of scripts and experiments for running YOLO-based object detection on geospatial imagery. It's a work-in-progress aimed at testing different methods for detecting objects in raster images (TIFFs, Cloud Optimized GeoTIFFs, and STAC catalog images) using YOLO. Both standard bounding boxes (BBox) and Oriented Bounding Boxes (OBB) are supported.

## What's Here

- **GeoTrain Class**: A script for preparing geospatial data and training a YOLO model.
- **GeoInference Class**: A script that slides a window across images to detect objects using a YOLO model.
- **Flexible Input Options**: Works with TIFF images, Cloud Optimized GeoTIFFs, or images in a STAC catalog.
- **Geospatial Outputs**: Converts pixel detections to geographic coordinates and saves them to a Parquet file.

## Getting Started

Tested on `WSL2` with `Ubuntu 22.04.5 LTS` and `Python 3.10.15`.

Install the main dependencies:

```bash
git clone https://github.com/hurr-son/spatialytics.git
cd yolo-geospatial-implementations
pip install -r requirements.txt
```

## Models

- **YOLO11-obb**: Pretrained on the [DOTA v1 dataset](https://captain-whu.github.io/DOTA/index.html) for oriented object detection.
- **YOLO11n-obb**, **YOLO11n**, and **YOLO8n**: Trained on the [COCO dataset](https://cocodataset.org/#home). While not trained on geospatial imagery, they can still yield useful results when applied to such data.

## How to Use

### 1. GeoTrain Class (Training)

#### Purpose

Prepares geospatial data and trains a YOLO model by creating image chips from large raster files and generating corresponding label files from vector data.

#### Usage

```python
from geo_train import GeoTrain

geo_train = GeoTrain(
    vector_path="path/to/vector_labels.geojson",
    raster_path="path/to/raster_image.tif",
    model_path="path/to/pretrained_yolo.pt",
    output_dir="path/to/output_directory",
    chip_size=640,        # Size of image chips
    stride=320,           # Stride for sliding window
    val_split=0.2,        # Fraction of data for validation
    class_names=["class1", "class2"],  # List of class names
    device="cuda"         # "cuda" or "cpu"
)

geo_train.run()
```

**Results**: Creates image chips and labels, splits the dataset, and trains the YOLO model. Outputs, including the trained model and metrics, are saved in the specified output directory.

### 2. GeoInference Class (Inference)

#### Purpose

Slides a window across images to detect objects using a YOLO model.

#### Usage

```python
from inference import GeoInference

geo_inference = GeoInference(
    model_path="path/to/your/yolo/model.pt",
    output_path="path/to/output/detections.parquet",
    class_list=[0, 1, 4, 5],  # Specify class indices as needed
    window_size=1280,
    stride=640,
    conf_threshold=0.25,
    iou_threshold=0.3,
    detection_type="obb"  # "obb" or "bbox"
)

# Run inference on different input types:
geo_inference.run(tif_path="path/to/image.tif")
# or
geo_inference.run(stac_catalog_url="path/to/stac/catalog.json")
# or
geo_inference.run(cog_url="https://example.com/image.cog.tif")

# Optionally, save detection crops:
geo_inference.run(
    cog_url="https://example.com/image.cog.tif",
    generate_crops=True,
    crops_output_dir="path/to/crops/"
)
```

**Results**: Detections with geographic coordinates are saved as a Parquet file at the specified output path.

## Utils

### obb_nms.py

When using the sliding window technique, overlapping windows can produce duplicate detections. `obb_nms.py` provides a custom Non-Maximum Suppression (NMS) function to consolidate these detections post-inference.

#### Example Usage

```python
from utils.obb_nms import load_and_run_nms

load_and_run_nms(
    input_parquet="path/to/your/input_detections.parquet",
    output_parquet="path/to/your/output_filtered_detections.parquet",
    iou_threshold=0.05  # Adjust IoU threshold as needed
)
```
