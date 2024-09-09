import torch

from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes, make_grid

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy

from typing import List
from torch import Tensor
from config_parser import YOLOConfig

COLORS = [
    # "#000000",  # Black (background)
    "#FF0000",  # Red (aeroplane)
    "#00FF00",  # Green (bicycle)
    "#0000FF",  # Blue (bird)
    "#FFFF00",  # Yellow (boat)
    "#00FFFF",  # Cyan (bottle)
    "#FF00FF",  # Magenta (bus)
    "#FFA500",  # Orange (car)
    "#800080",  # Purple (cat)
    "#00FF00",  # Lime (chair)
    "#FFC0CB",  # Pink (cow)
    "#008080",  # Teal (diningtable)
    "#E6E6FA",  # Lavender (dog)
    "#A52A2A",  # Brown (horse)
    "#F5F5DC",  # Beige (motorbike)
    "#800000",  # Maroon (person)
    "#808000",  # Olive (pottedplant)
    "#000080",  # Navy (sheep)
    "#FF7F50",  # Coral (sofa)
    "#40E0D0",  # Turquoise (train)
    "#FFD700",  # Gold (tvmonitor)
]


def draw_yolo(
    image: Tensor,
    target: Tensor,
    config: YOLOConfig,
    threshold=0.5,
    mode="output",
    pil=True,
) -> Tensor:
    assert image.dim() == 3 and target.dim() == 3, "Only one sample"

    image = v2.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )(image)

    if mode == "output":
        boxes, labels, _ = yolo_output_to_xyxy(target, config, threshold)
    elif mode == "target":
        boxes, labels, _ = yolo_target_to_xyxy(target, config, threshold)
    boxes, labels = boxes[0], labels[0]
    if boxes.numel() == 0:
        return image.detach().cpu()
    image_with_bbox = draw_bounding_boxes(
        image,
        boxes,
        labels=[config.VOC_DETECTION_CATEGORIES[i].upper() for i in labels],
        colors=[COLORS[i] for i in labels],
        width=3,
        font="Courier",
        font_size=25,
    )
    if pil:
        image_with_bbox = v2.ToPILImage()(image_with_bbox)

    return image_with_bbox


def draw_yolo_batch(
    images: Tensor,
    targets: Tensor,
    config: YOLOConfig,
    threshold=0.5,
    mode="output",
    pil=True,
) -> Tensor:
    assert images.dim() == 4 and targets.dim() == 4, "Input must be batched"
    assert images.size(0) == targets.size(0), "Batch size mismatch"

    images_with_bbox = [
        draw_yolo(
            image=image,
            target=target,
            threshold=threshold,
            config=config,
            mode=mode,
            pil=False,
        )
        for image, target in zip(images, targets)
    ]
    if not pil:
        images_with_bbox = torch.stack(images_with_bbox)

    return images_with_bbox


def draw_yolo_grid(
    images: Tensor, outputs: Tensor, targets: Tensor, config: YOLOConfig, threshold=0.5
) -> Tensor:
    assert (
        images.dim() == 4 and outputs.dim() == 4 and targets.dim() == 4
    ), "Input must be batched"
    assert images.size(0) == outputs.size(0) == targets.size(0), "Batch size mismatch"

    images_with_output = draw_yolo_batch(
        images=images,
        targets=outputs,
        threshold=threshold,
        config=config,
        mode="output",
        pil=False,
    )
    images_with_target = draw_yolo_batch(
        images=images,
        targets=targets,
        threshold=threshold,
        config=config,
        mode="target",
        pil=False,
    )
    images_with_bbox = torch.cat([images_with_output, images_with_target], dim=0)
    images_with_bbox = make_grid(images_with_bbox, nrow=images.size(0))

    return images_with_bbox


def draw_yolo_from_dict(
    image: Tensor,
    bboxes: dict,
    config: YOLOConfig,
    pil=True,
) -> Tensor:
    assert image.dim() == 3, "Only one sample"

    image = v2.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )(image)

    coords, labels = bboxes["boxes"], bboxes["labels"]
    confidences = bboxes.get("confidences", torch.ones_like(labels))
    confidences *= 100

    labels_with_conf = [
        f"{config.VOC_DETECTION_CATEGORIES[i].upper()} ({conf:.2f})"
        for i, conf in zip(labels, confidences)
    ]

    if coords.numel() == 0:
        return image.detach().cpu()

    image_with_bbox = draw_bounding_boxes(
        image,
        coords,
        labels=labels_with_conf,
        colors=[COLORS[i] for i in labels],
        width=3,
        font="Courier",
        font_size=25,
    )
    if pil:
        image_with_bbox = v2.ToPILImage()(image_with_bbox)

    return image_with_bbox


def draw_yolo_batch_from_dict(
    images: Tensor,
    bboxes_list: List[dict],
    config: YOLOConfig,
    pil=True,
) -> Tensor:
    assert len(images) == len(bboxes_list), "Batch size mismatch"

    images_with_bbox = [
        draw_yolo_from_dict(
            image=image,
            bboxes=bboxes,
            config=config,
            pil=False,
        )
        for image, bboxes in zip(images, bboxes_list)
    ]

    if not pil:
        images_with_bbox = torch.stack(images_with_bbox)

    return images_with_bbox


def draw_yolo_grid_from_dict(
    images: Tensor,
    outputs_list: List[dict],
    targets_list: List[dict],
    config: YOLOConfig,
) -> Tensor:
    assert len(images) == len(outputs_list) == len(targets_list), "Batch size mismatch"

    images_with_output = draw_yolo_batch_from_dict(
        images=images,
        bboxes_list=outputs_list,
        config=config,
        pil=False,
    )

    images_with_target = draw_yolo_batch_from_dict(
        images=images,
        bboxes_list=targets_list,
        config=config,
        pil=False,
    )

    images_with_bbox = torch.cat([images_with_output, images_with_target], dim=0)
    images_with_bbox = make_grid(images_with_bbox, nrow=len(images))

    return images_with_bbox
