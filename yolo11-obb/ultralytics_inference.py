import rasterio
from rasterio.windows import Window
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from ultralytics import YOLO
import pystac
from pathlib import Path
import yaml
import torch
from torchvision.ops import nms


class GeoInference:
    def __init__(self, model_path, class_yaml_path, output_path, window_size=1280, stride=640, conf_threshold=0.1, iou_threshold=0.5, classes_list=None):
        self.model_path = model_path
        self.class_yaml_path = class_yaml_path
        self.output_path = Path(output_path)
        self.window_size = window_size
        self.stride = stride
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes_list = classes_list  # Allow specifying class IDs
        self.all_detections = []
        self.load_model()
        self.classes = self.load_classes()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        self.model = YOLO(self.model_path)
        print(f"Loaded model from {self.model_path}")
        
    def load_classes(self):
        try:
            with open(self.class_yaml_path, 'r') as file: 
                classes = yaml.safe_load(file)
            print(f"Loaded classes: {classes}")
            return classes
        except FileNotFoundError:
            print("Class YAML file not found. Ensure the file path is correct.")
            return None
        
    def pixel_to_geo(self, transform, x, y):
        lon, lat = transform * (x, y)
        return lon, lat
        
    def process_window(self, dataset, x, y):
        window = Window(x, y, self.window_size, self.window_size)
        img = dataset.read([1, 2, 3], window=window)
        if img.shape[1] < self.window_size or img.shape[2] < self.window_size:
            padded_img = np.zeros((img.shape[0], self.window_size, self.window_size), dtype=img.dtype)
            padded_img[:, :img.shape[1], :img.shape[2]] = img
            img = padded_img
        img_np = np.moveaxis(img, 0, -1)
        try:
            results = self.model(
                img_np, 
                device='cuda:0',
                verbose=True, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold, 
                classes=self.classes_list
            )
            return results
        except Exception as e:
            print(f"Error during inference at window x: {x}, y: {y}: {e}")
            return None
        
    def process_image(self, tif_href):
        print(f"Processing image: {tif_href}")
        try:
            with rasterio.open(tif_href) as dataset:
                width, height = dataset.width, dataset.height
                transform = dataset.transform
                crs = dataset.crs
                image_id = Path(tif_href).stem
                x_windows = list(range(0, width, self.stride))
                y_windows = list(range(0, height, self.stride))
                print(f"Generated {len(x_windows)} x_windows and {len(y_windows)} y_windows for image {tif_href}")
                for y in y_windows:
                    for x in x_windows:
                        results = self.process_window(dataset, x, y)
                        if not results:
                            continue
                        for result in results:
                            obb = result.obb
                            if obb is None:
                                continue
                            for idx in range(len(obb.conf)):
                                confidence = obb.conf[idx].item()
                                class_index = int(obb.cls[idx].item())
                                if confidence < self.conf_threshold:
                                    continue
                                coords = obb.xyxyxyxy[idx].cpu().numpy().flatten()
                                coords[::2] += x
                                coords[1::2] += y
                                self.all_detections.append({
                                    'class_index': class_index,
                                    'coords': coords,
                                    'confidence': confidence,
                                    'source_image': image_id,
                                    'transform': transform,
                                    'crs': crs
                                })
                print(f"Detections collected from image {tif_href}: {len(self.all_detections)}")
        except Exception as e:
            print(f"Error processing image {tif_href}: {e}")
        
    def process_stac_catalog(self, stac_catalog_url):
        catalog = pystac.Catalog.from_file(stac_catalog_url)
        print(f"Loaded STAC catalog from {stac_catalog_url}")
        items = list(catalog.get_all_items())
        print(f"Found {len(items)} items in the catalog.")
        for item in items:
            for asset_key, asset in item.assets.items():
                asset_href = asset.get_absolute_href()
                asset_media_type = asset.media_type
                if asset_media_type in ['image/tiff', 'image/geotiff']:
                    print(f"Processing asset {asset_key} with href {asset_href}")
                    self.process_image(asset_href)
                else:
                    print(f"Skipping asset {asset_key} with media type {asset_media_type}")
        self.convert_and_save_detections()
        
    def convert_and_save_detections(self):
        if not self.all_detections:
            print("No detections to convert.")
            return None
        geo_detections = []
        for det in self.all_detections:
            class_index = det['class_index']
            coords = det['coords']
            confidence = det['confidence']
            source_image = det['source_image']
            transform = det['transform']
            crs = det['crs']
            pixel_coords = list(zip(coords[::2], coords[1::2]))
            geo_coords = [self.pixel_to_geo(transform, x_pixel, y_pixel) for x_pixel, y_pixel in pixel_coords]
            geometry = Polygon(geo_coords)
            geo_detections.append({
                'geometry': geometry,
                'confidence': confidence,
                'class_index': class_index,
                'source_image': source_image
            })
        gdf = gpd.GeoDataFrame(geo_detections, geometry='geometry')
        gdf.crs = crs
        print(f"Saving {len(geo_detections)} detections to {self.output_path}")
        gdf.to_parquet(self.output_path, index=False)
        
    def run(self, tif_path=None, stac_catalog_url=None, cog_url=None):
        if tif_path:
            self.process_image(tif_path)
            self.convert_and_save_detections()
        elif stac_catalog_url:
            self.process_stac_catalog(stac_catalog_url)
        elif cog_url:
            self.process_image(cog_url)
            self.convert_and_save_detections()
        else:
            print("No input provided. Please specify a tif_path, stac_catalog_url, or cog_url.")
    
if __name__ == '__main__':
    model_path = '/home/hurr_son/repos/yolo-geospatial-implementations/models/yolo11n-obb.pt'
    class_yaml_path = '/home/hurr_son/repos/yolo-geospatial-implementations/models/class-yaml/dotav1.yaml'
    output_path = '/home/hurr_son/repos/yolo-geospatial-implementations/test/detection.parquet'
    stac_catalog_url = 'https://coastalimagery.blob.core.windows.net/digitalcoast/TampaBayFL_RGBN_2023_9995/stac/catalog.json'
    # cog_url = 'https://coastalimagery.blob.core.windows.net/digitalcoast/TampaBayFL_RGBN_2023_9995/357000e3090000n.tif'
    window_size = 1280
    stride = 640
    conf_threshold = 0.2
    iou_threshold = 0.1
    classes_list = [1] 

    geo_inference = GeoInference(
        model_path=model_path,
        class_yaml_path=class_yaml_path,
        output_path=output_path,
        window_size=window_size,
        stride=stride,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes_list=classes_list
    )
    geo_inference.run(stac_catalog_url=stac_catalog_url)
