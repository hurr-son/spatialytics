import os
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, Polygon
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union, Dict, Any
import yaml
from ultralytics import YOLO
import torch
from tqdm import tqdm
from itertools import product
import random
import shutil


class GeoTrain:
    """
    A class for preparing geospatial data and training a YOLO model.

    This class handles the creation of image chips from large raster files,
    generates corresponding label files in YOLO format from vector data,
    splits the dataset into training and validation sets, and trains a YOLO model.
    """

    def __init__(
        self,
        vector_path: Union[str, Path],
        raster_path: Union[str, Path],
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        chip_size: int = 640,
        stride: int = 320,
        val_split: float = 0.2, 
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the GeoTrain class with the specified parameters.
        """
        self.vector_path: Path = Path(vector_path)
        self.raster_path: Path = Path(raster_path)
        self.model_path: Path = Path(model_path)
        self.output_dir: Path = Path(output_dir)
        self.chip_size: int = chip_size
        self.stride: int = stride
        self.val_split: float = val_split
        self.class_names: List[str] = class_names if class_names else ['object']
        self.device: str = device if device else self.get_device()

        self.images_dir: Path = self.output_dir / 'images'
        self.labels_dir: Path = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.train_images_dir: Path = self.output_dir / 'train' / 'images'
        self.train_labels_dir: Path = self.output_dir / 'train' / 'labels'
        self.val_images_dir: Path = self.output_dir / 'val' / 'images'
        self.val_labels_dir: Path = self.output_dir / 'val' / 'labels'

        self.vector_data: Optional[gpd.GeoDataFrame] = None
        self.raster: Optional[rasterio.io.DatasetReader] = None
        self.model: Optional[YOLO] = None

        self.load_files()

    def get_device(self) -> str:
        """
        Determines the device to use for computation ('cuda' or 'cpu').
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return device

    def load_files(self) -> None:
        """
        Loads the vector and raster files and the YOLO model.
        """
        print(f"Loading vector data from {self.vector_path}")
        self.vector_data = gpd.read_file(self.vector_path)
        self.vector_crs = self.vector_data.crs

        print(f"Loading raster data from {self.raster_path}")
        self.raster = rasterio.open(self.raster_path)
        self.raster_crs = self.raster.crs

        if self.vector_crs != self.raster_crs:
            print("Reprojecting vector data to match raster CRS")
            self.vector_data = self.vector_data.to_crs(self.raster_crs)

        print(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)

    def create_annotated_chips(self) -> None:
        """
        Creates image chips and corresponding YOLO-format labels.
        """
        width = self.raster.width
        height = self.raster.height

        x_steps = list(range(0, width, self.stride))
        y_steps = list(range(0, height, self.stride))

        total_steps = len(x_steps) * len(y_steps)
        print(f"Creating chips with chip_size={self.chip_size}, stride={self.stride}")
        print(f"Total chips to create: {total_steps}")

        chip_count = 0

        for x, y in tqdm(product(x_steps, y_steps), total=total_steps, desc='Creating chips'):
            window = Window(x, y, self.chip_size, self.chip_size)
            transform = self.raster.window_transform(window)
            chip_bounds = rasterio.windows.bounds(window, transform=self.raster.transform)

            chip_data = self.raster.read(window=window, boundless=True, fill_value=0)
            if chip_data.size == 0:
                continue

            chip_image = np.moveaxis(chip_data, 0, -1)
            if chip_image.shape[2] >= 3:
                chip_image = chip_image[:, :, :3]
            else:
                continue

            chip_filename = f'chip_{x}_{y}.png'
            chip_image_path = self.images_dir / chip_filename
            Image.fromarray(chip_image).save(chip_image_path)

            chip_geom = box(*chip_bounds)
            chip_labels = self.vector_data[self.vector_data.intersects(chip_geom)]

            label_filename = f'chip_{x}_{y}.txt'
            label_path = self.labels_dir / label_filename

            if not chip_labels.empty:
                with open(label_path, 'w') as f:
                    for _, row in chip_labels.iterrows():
                        geom = row['geometry'].intersection(chip_geom)
                        if geom.is_empty:
                            continue

                        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
                            continue

                        geom = geom.intersection(chip_geom)
                        if geom.is_empty:
                            continue

                        minx, miny, maxx, maxy = geom.bounds

                        minx_pixel, miny_pixel = ~transform * (minx, miny)
                        maxx_pixel, maxy_pixel = ~transform * (maxx, maxy)

                        x_center = ((minx_pixel + maxx_pixel) / 2) / self.chip_size
                        y_center = ((miny_pixel + maxy_pixel) / 2) / self.chip_size
                        width_norm = abs(maxx_pixel - minx_pixel) / self.chip_size
                        height_norm = abs(maxy_pixel - miny_pixel) / self.chip_size

                        x_center = max(min(x_center, 1.0), 0.0)
                        y_center = max(min(y_center, 1.0), 0.0)
                        width_norm = max(min(width_norm, 1.0), 0.0)
                        height_norm = max(min(height_norm, 1.0), 0.0)

                        class_name = row.get('class_name', 'object')
                        if class_name in self.class_names:
                            class_idx = self.class_names.index(class_name)
                        else:
                            continue

                        if width_norm == 0 or height_norm == 0:
                            continue

                        f.write(f"{class_idx} {x_center} {y_center} {width_norm} {height_norm}\n")
            else:
                open(label_path, 'a').close()

            chip_count += 1

        print(f"Finished creating {chip_count} annotated chips.")

    def split_dataset(self) -> None:
        """
        Splits the dataset into training and validation sets.
        """
        print("Splitting dataset into training and validation sets...")

        image_files = list(self.images_dir.glob('*.png'))
        label_files = list(self.labels_dir.glob('*.txt'))

        image_files.sort()
        label_files.sort()

        data_pairs = list(zip(image_files, label_files))

        data_pairs = [
            (img, lbl) for img, lbl in data_pairs if lbl.exists()
        ]

        random.seed(42)  
        random.shuffle(data_pairs)

        split_index = int(len(data_pairs) * (1 - self.val_split))
        train_data = data_pairs[:split_index]
        val_data = data_pairs[split_index:]

  
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)

      
        for img_path, lbl_path in train_data:
            shutil.copy(img_path, self.train_images_dir / img_path.name)
            shutil.copy(lbl_path, self.train_labels_dir / lbl_path.name)

        for img_path, lbl_path in val_data:
            shutil.copy(img_path, self.val_images_dir / img_path.name)
            shutil.copy(lbl_path, self.val_labels_dir / lbl_path.name)

        print(f"Training set size: {len(train_data)} samples")
        print(f"Validation set size: {len(val_data)} samples")

    def train_model(self, epochs: int = 100, batch_size: int = 8) -> None:
        """
        Trains the YOLO model using the prepared dataset.
        """
        # Prepare data YAML
        data_yaml = {
            'train': str(self.train_images_dir),
            'val': str(self.val_images_dir),
            'nc': len(self.class_names),
            'names': self.class_names
        }

        data_yaml_path = self.output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        print(f"Data YAML saved to {data_yaml_path}")

        print("Starting model training...")
        self.model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=self.chip_size,
            project=str(self.output_dir),
            name='yolo_training',
            exist_ok=True,
            device=self.device 
        )
        print("Model training completed.")

    def save_model_and_benchmarks(self) -> None:
        """
        Saves the trained model and benchmark metrics.
        """
        print("Model and benchmarks are saved in the output directory.")

    def run(self) -> None:
        """
        Executes the full training pipeline: data preparation, dataset splitting, and model training.
        """
        self.create_annotated_chips()
        self.split_dataset()
        self.train_model()
        self.save_model_and_benchmarks()


# if __name__ == "__main__":
#     geo_train = GeoTrain(
#         vector_path='/home/hurr_son/repos/yolo-geospatial-implementations/data/train/bounding_boxes.geojson',
#         raster_path='https://coastalimagery.blob.core.windows.net/digitalcoast/WI_NAIP_2020_9514/m_4308757_se_16_060_20200902.tif',
#         model_path='/home/hurr_son/repos/yolo-geospatial-implementations/models/pretrained/yolo11n.pt',
#         output_dir='/home/hurr_son/repos/yolo-geospatial-implementations/data/train/output',
#         chip_size=640,
#         stride=640,
#         val_split=0.2,
#         class_names=['buldings']
#     )
#     geo_train.run()
