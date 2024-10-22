import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm

def calculate_obb_iou(poly1, poly2):
    """
    Calculate the Intersection over Union (IoU) for two oriented bounding boxes (OBBs).
    The OBBs are represented as shapely polygons.
    """
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def run_secondary_obb_nms(detections, iou_threshold=0.5):
    """
    Run Non-Maximum Suppression (NMS) on a set of Oriented Bounding Boxes (OBBs) 
    using IoU as the overlap metric. Keeps the highest confidence detections 
    and removes overlapping duplicates.
    """
    if len(detections) == 0:
        print("No detections to process.")
        return detections

 
    polys = list(detections['geometry'])
    scores = np.array(detections['confidence'])

   
    keep = []

   
    indices = np.argsort(-scores)

    total_detections = len(indices)
    processed_count = 0

    
    with tqdm(total=total_detections, desc="Running OBB NMS") as pbar:
        while len(indices) > 0:
           
            current_index = indices[0]
            keep.append(current_index)

           
            processed_count += 1

            if len(indices) == 1:
                break

           
            remaining_indices = indices[1:]
            ious = np.array([calculate_obb_iou(polys[current_index], polys[i]) for i in remaining_indices])

           
            indices = np.array([index for i, index in enumerate(remaining_indices) if ious[i] < iou_threshold])

         
            pbar.update(1)

        
        pbar.update(total_detections - processed_count)

  
    filtered_detections = detections.iloc[keep]
    print(f"After OBB NMS, {len(filtered_detections)} detections remain.")
    return filtered_detections

def main(input_parquet, output_parquet, iou_threshold=0.5):
    """
    Main function to load detections from a Parquet file, run OBB NMS, 
    and save the filtered detections to a new Parquet file.
    """
 
    detections = gpd.read_parquet(input_parquet)
    print(f"Loaded {len(detections)} detections from {input_parquet}")

   
    filtered_detections = run_secondary_obb_nms(detections, iou_threshold)

 
    filtered_detections.to_parquet(output_parquet, index=False)
    print(f"Filtered detections saved to {output_parquet}")

def main(input_parquet, output_parquet, iou_threshold=0.5):

    detections = gpd.read_parquet(input_parquet)
    print(f"Loaded {len(detections)} detections from {input_parquet}")

  
    filtered_detections = run_secondary_obb_nms(detections, iou_threshold)

    
    filtered_detections.to_parquet(output_parquet, index=False)
    print(f"Filtered detections saved to {output_parquet}")
    
if __name__ == "__main__":
    input_parquet = "/home/hurr_son/repos/yolo-geospatial-implementations/test/detection.parquet"
    output_parquet = "/home/hurr_son/repos/yolo-geospatial-implementations/test/filtered_detections.parquet"
    iou_threshold = 0.05

    main(input_parquet, output_parquet, iou_threshold)
