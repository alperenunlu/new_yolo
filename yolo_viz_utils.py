import torch

from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes, make_grid

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy

from torch import Tensor
from config_parser import YOLOCONFIG


def draw_yolo(
    image: Tensor,
    target: Tensor,
    config: YOLOCONFIG,
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
        width=3,
        colors="red",
        font="Courier",
        font_size=25,
    )
    if pil:
        image_with_bbox = v2.ToPILImage()(image_with_bbox)
    return image_with_bbox


def draw_yolo_batch(
    images: Tensor,
    targets: Tensor,
    config: YOLOCONFIG,
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
    images: Tensor, outputs: Tensor, targets: Tensor, config: YOLOCONFIG, threshold=0.5
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
