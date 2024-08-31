
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy
from config_parser import config

def draw_yolo_target(image, target, threshold=0.5, config=config, mode="output"):
    if mode == "output":
        boxes, labels, _ = yolo_output_to_xyxy(target, threshold, config)
    elif mode == "target":
        boxes, labels, _ = yolo_target_to_xyxy(target, threshold, config)
    return v2.ToPILImage()(
        draw_bounding_boxes(
            image,
            boxes,
            labels=[config.VOC_DETECTION_CATEGORIES[i].upper() for i in labels],
            width=3,
            font="Courier",
            font_size=25,
        )
    )