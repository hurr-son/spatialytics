import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch
from typing import List, Optional, Union
import random
import shutil
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import product


class BaseTrainer:
    """
    A base class for training YOLO models.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the BaseTrainer class with the specified parameters.
        """
        self.model_path: Path = Path(model_path)
        self.output_dir: Path = Path(output_dir)
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.img_size: int = img_size
        self.device: str = device if device else self.get_device()

        self.model: Optional[YOLO] = None
        self.load_model()

    def get_device(self) -> str:
        """
        Determines the device to use for computation ('cuda' or 'cpu').
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return device

    def load_model(self) -> None:
        """
        Loads the YOLO model.
        """
        print(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(self.model_path)

    def train_model(self, data_yaml: Union[str, Path]) -> None:
        """
        Trains the YOLO model using the prepared dataset.
        """
        print("Starting model training...")
        self.model.train(
            data=str(data_yaml),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            project=str(self.output_dir),
            name='yolo_training',
            exist_ok=True,
            device=self.device
        )
        print("Model training completed.")

    def save_model_and_benchmarks(self) -> None:
        """
        Placeholder for any additional saving or benchmarking after training.
        """
        print("Model and benchmarks are saved in the output directory.")

    def run(self) -> None:
        """
        Executes the full training pipeline.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class GeoTrain(BaseTrainer):
    """
    A class for preparing geospatial data and training a YOLO model.

    Inherits from BaseTrainer.
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
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the GeoTrain class with the specified parameters.
        """
        super().__init__(model_path, output_dir, epochs, batch_size, img_size, device)

        self.vector_path: Path = Path(vector_path)
        self.raster_path: Path = Path(raster_path)
        self.chip_size: int = chip_size
        self.stride: int = stride
        self.val_split: float = val_split
        self.class_names: List[str] = class_names if class_names else ['object']

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

        self.load_files()

    def load_files(self) -> None:
        """
        Loads the vector and raster files.
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
            window_transform = self.raster.window_transform(window)
            inv_window_transform = ~window_transform
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

                        minx, miny, maxx, maxy = geom.bounds

                        min_col, min_row = inv_window_transform * (minx, miny)
                        max_col, max_row = inv_window_transform * (maxx, maxy)

                        x_center = ((min_col + max_col) / 2) / self.chip_size
                        y_center = ((min_row + max_row) / 2) / self.chip_size
                        width_norm = abs(max_col - min_col) / self.chip_size
                        height_norm = abs(max_row - min_row) / self.chip_size
         
                        x_center = max(min(x_center, 1.0), 0.0)
                        y_center = max(min(y_center, 1.0), 0.0)
                        width_norm = max(min(width_norm, 1.0), 0.0)
                        height_norm = max(min(height_norm, 1.0), 0.0)
               
                        if width_norm == 0 or height_norm == 0:
                            continue
           
                        class_name = row.get('class_name', 'object')
                        if class_name in self.class_names:
                            class_idx = self.class_names.index(class_name)
                        else:
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

    def prepare_data_yaml(self) -> Path:
        """
        Prepares the data YAML file required for training.
        """
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

        return data_yaml_path

    def run(self) -> None:
        """
        Executes the full training pipeline: data preparation, dataset splitting, and model training.
        """
        self.create_annotated_chips()
        self.split_dataset()
        data_yaml_path = self.prepare_data_yaml()
        self.train_model(data_yaml_path)
        self.save_model_and_benchmarks()


class YOLOTrain(BaseTrainer):
    """
    A class for training a YOLO model on pre-chipped datasets.

    Inherits from BaseTrainer.
    """

    def __init__(
        self,
        data_yaml: Union[str, Path],
        model_path: Union[str, Path],
        output_dir: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """
        Initializes the YOLOTrain class with the specified parameters.
        """
        super().__init__(model_path, output_dir, epochs, batch_size, img_size, device)
        self.data_yaml: Path = Path(data_yaml)

    def run(self) -> None:
        """
        Executes the full training pipeline: model training and saving.
        """
        self.train_model(self.data_yaml)
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
#         class_names=['buldings'],
#         epochs=100,
#         batch_size=8,
#         img_size=640,
#         device='cuda'
#     )
#     geo_train.run()

    # yolo_train = YOLOTrain(
    #     data_yaml='/home/hurr_son/repos/yolo-geospatial-implementations/data/train/diorsubset/DIOR subset.v2-trained.yolov11/data.yaml',  # For YOLO format datasets
    #     model_path='/home/hurr_son/repos/yolo-geospatial-implementations/models/pretrained/yolo11n.pt',
    #     output_dir='/home/hurr_son/repos/yolo-geospatial-implementations/data/train/output',
    #     epochs=5,
    #     batch_size=8,
    #     img_size=640,
    #     device='cuda'
    # )
    # yolo_train.run()
