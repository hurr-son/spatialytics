import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from numpy.typing import NDArray

def calculate_obb_iou(poly1: Polygon, poly2: Polygon) -> float:
    """
    Calculates the Intersection over Union (IoU) for two oriented bounding boxes (OBBs).

    The OBBs are represented as Shapely polygons. This function computes the area of
    the intersection divided by the area of the union of the two polygons.

    Args:
        poly1 (Polygon): The first oriented bounding box as a Shapely polygon.
        poly2 (Polygon): The second oriented bounding box as a Shapely polygon.

    Returns:
        float: The IoU value ranging from 0.0 to 1.0. If either polygon is invalid,
        or if the union area is zero, the function returns 0.0.
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area: float = poly1.intersection(poly2).area
    union_area: float = poly1.union(poly2).area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def run_secondary_obb_nms(
    detections: gpd.GeoDataFrame, 
    iou_threshold: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Performs Non-Maximum Suppression (NMS) on a set of Oriented Bounding Boxes (OBBs)
    using IoU as the overlap metric.

    This function keeps the highest confidence detections and removes overlapping
    duplicates based on the specified IoU threshold.

    Args:
        detections (gpd.GeoDataFrame): A GeoDataFrame containing detection results,
            including geometries and confidence scores.
        iou_threshold (float, optional): The IoU threshold for suppressing overlapping
            detections. Defaults to 0.5.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered detections after NMS.
    """
    if len(detections) == 0:
        print("No detections to process.")
        return detections

    polys: List[Polygon] = list(detections['geometry'])
    scores: NDArray[np.float_] = np.array(detections['confidence'], dtype=np.float_)

    keep: List[int] = []

    indices: NDArray[np.int_] = np.argsort(-scores)

    total_detections: int = len(indices)
    processed_count: int = 0

    with tqdm(total=total_detections, desc="Running OBB NMS") as pbar:
        while len(indices) > 0:
            current_index: int = indices[0]
            keep.append(current_index)

            processed_count += 1

            if len(indices) == 1:
                break

            remaining_indices: NDArray[np.int_] = indices[1:]
            ious: NDArray[np.float_] = np.array([
                calculate_obb_iou(polys[current_index], polys[i]) for i in remaining_indices
            ], dtype=np.float_)

            indices = np.array([
                index for i, index in enumerate(remaining_indices) if ious[i] < iou_threshold
            ], dtype=np.int_)

            pbar.update(1)

        pbar.update(total_detections - processed_count)

    filtered_detections: gpd.GeoDataFrame = detections.iloc[keep]
    print(f"After OBB NMS, {len(filtered_detections)} detections remain.")
    return filtered_detections

def load_and_run_nms(
    input_parquet: Path,
    output_parquet: Path,
    iou_threshold: float = 0.5
) -> None:
    """
    Loads detections from a Parquet file, runs OBB NMS, and saves the filtered detections.

    Args:
        input_parquet (Path): The file path to the input Parquet file containing detections.
        output_parquet (Path): The file path where the filtered detections will be saved.
        iou_threshold (float, optional): The IoU threshold for NMS. Defaults to 0.5.
    """
    detections: gpd.GeoDataFrame = gpd.read_parquet(input_parquet)
    print(f"Loaded {len(detections)} detections from {input_parquet}")

    filtered_detections: gpd.GeoDataFrame = run_secondary_obb_nms(detections, iou_threshold)

    filtered_detections.to_parquet(output_parquet, index=False)
    print(f"Filtered detections saved to {output_parquet}")


