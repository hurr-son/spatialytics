from ultralytics_inference import GeoInference
from utils.obb_nms import load_and_run_nms

model_path = '/home/hurr_son/repos/yolo-geospatial-implementations/models/pretrained/yolo11n-obb.pt'
class_yaml_path = '/home/hurr_son/repos/yolo-geospatial-implementations/models/class-yaml/dotav1.yaml'
output_path = '/home/hurr_son/repos/yolo-geospatial-implementations/test/detection.parquet'
# stac_catalog_url = 'https://coastalimagery.blob.core.windows.net/digitalcoast/TampaBayFL_RGBN_2023_9995/stac/catalog.json'
cog_url = 'https://coastalimagery.blob.core.windows.net/digitalcoast/TampaBayFL_RGBN_2023_9995/357000e3090000n.tif'
window_size = 1280
stride = 640
conf_threshold = 0.25
iou_threshold = 0.3
classes_list = [3,4,5,6,13]  
detection_type = 'obb' 


   model_path: Path = Path('/home/hurr_son/repos/yolo-geospatial-implementations/models/pretrained/yolo11n-obb.pt')
    class_yaml_path: Path = Path('/home/hurr_son/repos/yolo-geospatial-implementations/models/class-yaml/dotav1.yaml')
    output_path: Path = Path('/home/hurr_son/repos/yolo-geospatial-implementations/test/detection.parquet')
    # stac_catalog_url: Union[str, Path] = 'https://example.com/your_catalog.json'
    cog_url: str = 'https://coastalimagery.blob.core.windows.net/digitalcoast/TampaBayFL_RGBN_2023_9995/357000e3090000n.tif'
    window_size: int = 1280
    stride: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.3
    classes_list: List[int] = [1]  # Detecting a specific class
    detection_type: str = 'obb'  # Or 'bbox'
    device: str = 'cuda:0'

    geo_inference = GeoInference(
        model_path=model_path,
        class_yaml_path=class_yaml_path,
        output_path=output_path,
        window_size=window_size,
        stride=stride,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes_list=classes_list,
        detection_type=detection_type
    )
    # geo_inference.run(tif_path='/path/to/your/image.tif')
    # geo_inference.run(stac_catalog_url=stac_catalog_url)
    geo_inference.run(cog_url=cog_url)

load_and_run_nms(input_parquet=output_path, output_parquet="/home/hurr_son/repos/yolo-geospatial-implementations/test/filtered_detections.parquet", iou_threshold=0.05)