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
from typing import List, Optional, Dict, Any, Union, Tuple
from PIL import Image
import uuid


class GeoInference:
    """
    A class for performing geospatial inference using a YOLO model on raster images.

    This class supports processing individual TIFF images, Cloud Optimized GeoTIFFs (COGs),
    and images referenced in a STAC catalog. It can handle both Oriented Bounding Boxes (OBBs)
    and regular bounding boxes (BBox).

    Attributes:
        model_path (Path): Path to the trained YOLO model.
        class_yaml_path (Path): Path to the YAML file containing class mappings.
        output_path (Path): Path where the output detections will be saved.
        window_size (int): Size of the sliding window for processing the image.
        stride (int): Stride of the sliding window.
        conf_threshold (float): Confidence threshold for detections.
        iou_threshold (float): Intersection over Union threshold for NMS.
        classes_list_input (Optional[List[Union[int, str]]]): List of class indices or names to detect.
        classes_list (Optional[List[int]]): Processed list of class indices to detect.
        detection_type (str): Type of detection ('obb' or 'bbox').
        all_detections (List[Dict[str, Any]]): Accumulated detections from the image.
        model (YOLO): Loaded YOLO model for inference.
        classes_index_to_name (Dict[str, Any]): Mapping of class indices to class names.
        classes_name_to_index (Dict[str, int]): Mapping of class names to class indices.
        device (str): The device to run inference on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        class_yaml_path: Union[str, Path],
        output_path: Union[str, Path],
        window_size: int = 1280,
        stride: int = 640,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.5,
        classes_list: Optional[List[Union[int, str]]] = None,
        detection_type: str = 'obb',
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the GeoInference class with the specified parameters.

        Args:
            model_path (Union[str, Path]): Path to the trained YOLO model.
            class_yaml_path (Union[str, Path]): Path to the YAML file with class mappings.
            output_path (Union[str, Path]): Path where output detections will be saved.
            window_size (int, optional): Size of the sliding window. Defaults to 1280.
            stride (int, optional): Stride of the sliding window. Defaults to 640.
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.1.
            iou_threshold (float, optional): IoU threshold for Non-Maximum Suppression. Defaults to 0.5.
            classes_list (Optional[List[Union[int, str]]], optional): List of class indices or names to detect.
            detection_type (str, optional): Type of detection ('obb' or 'bbox'). Defaults to 'obb'.
            device (Optional[str], optional): Device to use for computation ('cuda' or 'cpu'). Defaults to None.
        """
        self.device = device if device else self.get_device()
        self.model_path: Path = Path(model_path)
        self.class_yaml_path: Path = Path(class_yaml_path)
        self.output_path: Path = Path(output_path)
        self.window_size: int = window_size
        self.stride: int = stride
        self.conf_threshold: float = conf_threshold
        self.iou_threshold: float = iou_threshold
        self.classes_list_input: Optional[List[Union[int, str]]] = classes_list
        self.classes_list: Optional[List[int]] = None  # Will hold the processed indices
        self.detection_type: str = detection_type
        self.all_detections: List[Dict[str, Any]] = []
        self.load_model()
        self.load_classes()
        self.process_classes_list()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def get_device(self) -> str:
        """
        Detects the available device ('cuda' if GPU is available, else 'cpu').

        Returns:
            str: The device to use for computation.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return device

    def load_model(self) -> None:
        """
        Loads the YOLO model from the specified path.
        """
        self.model: YOLO = YOLO(self.model_path)
        print(f"Loaded model from {self.model_path}")

    def load_classes(self) -> None:
        """
        Loads class mappings from the YAML file and creates mappings from indices to names and names to indices.
        """
        try:
            with open(self.class_yaml_path, 'r') as file:
                class_data = yaml.safe_load(file)
            self.classes_index_to_name = class_data.get('names', {})
            # Create reverse mapping
            self.classes_name_to_index = {name: int(index) for index, name in self.classes_index_to_name.items()}
            print(f"Loaded classes: {self.classes_index_to_name}")
        except FileNotFoundError:
            print("Class YAML file not found. Ensure the file path is correct.")
            self.classes_index_to_name = {}
            self.classes_name_to_index = {}

    def process_classes_list(self) -> None:
        """
        Processes the classes_list input, converting class names to indices if necessary.
        """
        if self.classes_list_input is None:
            self.classes_list = None
            return
        self.classes_list = []
        for cls in self.classes_list_input:
            if isinstance(cls, int):
                self.classes_list.append(cls)
            elif isinstance(cls, str):
                cls_index = self.classes_name_to_index.get(cls)
                if cls_index is not None:
                    self.classes_list.append(cls_index)
                else:
                    print(f"Warning: Class name '{cls}' not found in class mappings.")
            else:
                print(f"Warning: Unsupported class type '{type(cls)}' in classes_list. Expected int or str.")

    def pixel_to_geo(
        self,
        transform: rasterio.Affine,
        x: float,
        y: float
    ) -> Tuple[float, float]:
        """
        Converts pixel coordinates to geographic coordinates using the affine transform.

        Args:
            transform (rasterio.Affine): Affine transformation matrix of the raster.
            x (float): X-coordinate in pixel space.
            y (float): Y-coordinate in pixel space.

        Returns:
            Tuple[float, float]: Longitude and latitude corresponding to the pixel coordinates.
        """
        lon, lat = transform * (x, y)
        return lon, lat

    def process_window(
        self,
        dataset: rasterio.io.DatasetReader,
        x: int,
        y: int
    ) -> Optional[Tuple[List[Any], np.ndarray]]:
        """
        Processes a window of the image and performs inference.

        Args:
            dataset (rasterio.io.DatasetReader): Opened raster dataset.
            x (int): X-coordinate (column) of the top-left corner of the window.
            y (int): Y-coordinate (row) of the top-left corner of the window.

        Returns:
            Optional[Tuple[List[Any], np.ndarray]]: Tuple of detection results and image array, or None if an error occurs.
        """
        window: Window = Window(x, y, self.window_size, self.window_size)
        img: np.ndarray = dataset.read([1, 2, 3], window=window, boundless=True)
        img_np: np.ndarray = np.moveaxis(img, 0, -1)
        if img_np.size == 0:
            return None
        try:
            results = self.model(
                img_np,
                device=self.device,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes_list
            )
            return results, img_np
        except Exception as e:
            print(f"Error during inference at window x: {x}, y: {y}: {e}")
            return None

    def process_image(
        self,
        tif_href: Union[str, Path],
        generate_crops: bool = False,
        crops_output_dir: Optional[Path] = None
    ) -> None:
        """
        Processes a TIFF image and collects detections.

        Args:
            tif_href (Union[str, Path]): Path or URL to the TIFF image.
            generate_crops (bool, optional): If True, saves image crops of detections. Defaults to False.
            crops_output_dir (Optional[Path], optional): Directory to save image crops. Defaults to None.
        """
        print(f"Processing image: {tif_href}")
        if generate_crops and crops_output_dir:
            crops_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(tif_href) as dataset:
                width: int = dataset.width
                height: int = dataset.height
                transform: rasterio.Affine = dataset.transform
                crs: Any = dataset.crs
                image_id: str = Path(tif_href).stem
                x_windows: List[int] = list(range(0, width, self.stride))
                y_windows: List[int] = list(range(0, height, self.stride))
                print(f"Generated {len(x_windows)} x_windows and {len(y_windows)} y_windows for image {tif_href}")
                for y in y_windows:
                    for x in x_windows:
                        result_tuple = self.process_window(dataset, x, y)
                        if not result_tuple:
                            continue
                        results, img_np = result_tuple
                        for result in results:
                            detections = self.extract_detections(result)
                            for det in detections:
                                # Generate a unique detection ID
                                detection_id = str(uuid.uuid4())
                                det['detection_id'] = detection_id
                                det['coords'][::2] += x
                                det['coords'][1::2] += y
                                det['source_image'] = image_id
                                det['transform'] = transform
                                det['crs'] = crs
                                self.all_detections.append(det)
                                if generate_crops and crops_output_dir:
                                    try:
                                        self.save_detection_crop(
                                            img_np, det, x, y, crops_output_dir
                                        )
                                    except Exception as e:
                                        print(f"Error saving crop: {e}")
            print(f"Detections collected from image {tif_href}: {len(self.all_detections)}")
        except Exception as e:
            print(f"Error processing image {tif_href}: {e}")

    def extract_detections(self, result) -> List[Dict[str, Any]]:
        """
        Extracts detections from the model result.

        Args:
            result: The result object from the model inference.

        Returns:
            List[Dict[str, Any]]: List of detection dictionaries.
        """
        detections = []
        if self.detection_type == 'obb' and hasattr(result, 'obb'):
            obb = result.obb
            if obb is None:
                return detections
            for idx in range(len(obb.conf)):
                confidence: float = obb.conf[idx].item()
                class_index: int = int(obb.cls[idx].item())
                if confidence < self.conf_threshold:
                    continue
                coords: np.ndarray = obb.xyxyxyxy[idx].cpu().numpy().flatten()
                detections.append({
                    'class_index': class_index,
                    'coords': coords,
                    'confidence': confidence,
                    'type': 'obb'
                })
        elif self.detection_type == 'bbox' and hasattr(result, 'boxes'):
            boxes = result.boxes
            if boxes is None:
                return detections
            for idx in range(len(boxes.conf)):
                confidence: float = boxes.conf[idx].item()
                class_index: int = int(boxes.cls[idx].item())
                if confidence < self.conf_threshold:
                    continue
                coords: np.ndarray = boxes.xyxy[idx].cpu().numpy()
                # Convert bbox to polygon coordinates
                coords = np.array([coords[0], coords[1], coords[2], coords[1],
                                   coords[2], coords[3], coords[0], coords[3]])
                detections.append({
                    'class_index': class_index,
                    'coords': coords,
                    'confidence': confidence,
                    'type': 'bbox'
                })
        else:
            print(f"Unknown detection type: {self.detection_type}")
        return detections

    def save_detection_crop(
        self,
        img_np: np.ndarray,
        det: Dict[str, Any],
        x_offset: int,
        y_offset: int,
        crops_output_dir: Path
    ) -> None:
        """
        Saves the crop of a detection.

        Args:
            img_np (np.ndarray): The image data of the window.
            det (Dict[str, Any]): Detection dictionary.
            x_offset (int): The x-coordinate offset of the window.
            y_offset (int): The y-coordinate offset of the window.
            crops_output_dir (Path): Directory to save the crops.
        """
        coords = det['coords']
        detection_id = det['detection_id']
        # Adjust coordinates to window local
        coords_local = coords.copy()
        coords_local[::2] -= x_offset
        coords_local[1::2] -= y_offset
        # Create a mask for the polygon
        polygon = Polygon(zip(coords_local[::2], coords_local[1::2]))
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny = int(max(minx, 0)), int(max(miny, 0))
        maxx, maxy = int(min(maxx, img_np.shape[1])), int(min(maxy, img_np.shape[0]))
        if minx >= maxx or miny >= maxy:
            return  # Invalid crop
        crop_img = img_np[miny:maxy, minx:maxx]
        # Generate the crop filename using the detection_id
        crop_filename = f"{detection_id}.png"
        crop_path = crops_output_dir / crop_filename
        Image.fromarray(crop_img).save(crop_path)

    def process_stac_catalog(
        self,
        stac_catalog_url: Union[str, Path],
        generate_crops: bool = False,
        crops_output_dir: Optional[Path] = None
    ) -> None:
        """
        Processes images referenced in a STAC catalog.

        Args:
            stac_catalog_url (Union[str, Path]): URL or path to the STAC catalog.
            generate_crops (bool, optional): If True, saves image crops of detections. Defaults to False.
            crops_output_dir (Optional[Path], optional): Directory to save image crops. Defaults to None.
        """
        catalog: pystac.Catalog = pystac.Catalog.from_file(str(stac_catalog_url))
        print(f"Loaded STAC catalog from {stac_catalog_url}")
        items: List[pystac.Item] = list(catalog.get_all_items())
        print(f"Found {len(items)} items in the catalog.")
        for item in items:
            for asset_key, asset in item.assets.items():
                asset_href: str = asset.get_absolute_href()
                asset_media_type: Optional[str] = asset.media_type
                if asset_media_type in ['image/tiff', 'image/geotiff']:
                    print(f"Processing asset {asset_key} with href {asset_href}")
                    self.process_image(asset_href, generate_crops, crops_output_dir)
                else:
                    print(f"Skipping asset {asset_key} with media type {asset_media_type}")
        self.convert_and_save_detections()

    def convert_and_save_detections(self) -> None:
        """
        Converts pixel-based detections to geographic coordinates and saves them as a Parquet file.
        """
        if not self.all_detections:
            print("No detections to convert.")
            return
        geo_detections: List[Dict[str, Any]] = []
        for det in self.all_detections:
            class_index: int = det['class_index']
            coords: np.ndarray = det['coords']
            confidence: float = det['confidence']
            source_image: str = det['source_image']
            transform: rasterio.Affine = det['transform']
            crs: Any = det['crs']
            detection_id: str = det['detection_id']  # Get the detection_id
            # Convert flat array of coordinates to list of (x, y) tuples
            pixel_coords: List[Tuple[float, float]] = list(zip(coords[::2], coords[1::2]))
            geo_coords: List[Tuple[float, float]] = [
                self.pixel_to_geo(transform, x_pixel, y_pixel)
                for x_pixel, y_pixel in pixel_coords
            ]
            geometry: Polygon = Polygon(geo_coords)
            class_name = self.classes_index_to_name.get(str(class_index), str(class_index))
            geo_detections.append({
                'geometry': geometry,
                'confidence': confidence,
                'class_index': class_index,
                'class_name': class_name,
                'source_image': source_image,
                'detection_id': detection_id  # Include the detection_id in the GeoDataFrame
            })
        gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(geo_detections, geometry='geometry')
        gdf.crs = crs
        print(f"Saving {len(geo_detections)} detections to {self.output_path}")
        gdf.to_parquet(self.output_path, index=False)

    def run(
        self,
        tif_path: Optional[Union[str, Path]] = None,
        stac_catalog_url: Optional[Union[str, Path]] = None,
        cog_url: Optional[Union[str, Path]] = None,
        generate_crops: bool = False,
        crops_output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Executes the geospatial inference process based on the provided input.

        Args:
            tif_path (Optional[Union[str, Path]], optional): Path to a TIFF image. Defaults to None.
            stac_catalog_url (Optional[Union[str, Path]], optional): URL or path to a STAC catalog. Defaults to None.
            cog_url (Optional[Union[str, Path]], optional): URL to a Cloud Optimized GeoTIFF (COG). Defaults to None.
            generate_crops (bool, optional): If True, saves image crops of detections. Defaults to False.
            crops_output_dir (Optional[Union[str, Path]], optional): Directory to save image crops. Defaults to None.
        """
        if generate_crops:
            crops_output_dir = Path(crops_output_dir) if crops_output_dir else self.output_path.parent / "detection_crops"
        else:
            crops_output_dir = None

        if tif_path:
            self.process_image(tif_path, generate_crops, crops_output_dir)
            self.convert_and_save_detections()
        elif stac_catalog_url:
            self.process_stac_catalog(stac_catalog_url, generate_crops, crops_output_dir)
        elif cog_url:
            self.process_image(cog_url, generate_crops, crops_output_dir)
            self.convert_and_save_detections()
        else:
            print("No input provided. Please specify a tif_path, stac_catalog_url, or cog_url.")
