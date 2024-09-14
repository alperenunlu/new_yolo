import torch
import torch.nn.functional as F

from torchvision.ops import box_convert, box_iou, nms

from typing import Tuple, List, Dict, Union
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torch import Tensor
from config_parser import YOLOConfig


def xyxy_to_yolo_target(
    boxes: BoundingBoxes, labels: Tensor, config: YOLOConfig
) -> Tensor:
    """
    boxes: BoundingBoxes object with shape (N, 4) or (4)
    labels: Tensor of shape (N,) or ()

    boxes format: [x1, y1, x2, y2]
    labels format: [label1, label2, ..., labelN]

    target format: [c1, c2, ..., C, conf, cx, cy, w, h]
        where:
            c1, c2, ..., C = one-hot encoded labels
            conf = 1 if object exists in cell
            cx, cy = center of the bbox relative to the cell
            w, h = width and height of the bbox relative to the image size
    """
    S = config.S
    C = config.C

    if boxes.dim() == 1:
        boxes = boxes.unsqueeze(0)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    assert boxes.format == BoundingBoxFormat.XYXY

    # Since the boxes could be in the same cell we shuffle them to avoid bias
    indices = torch.randperm(len(boxes))
    boxes = boxes[indices]
    labels = labels[indices]

    cx, cy, w, h = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh").unbind(-1)
    canvas_size = config.IMAGE_SIZE
    cell_w = canvas_size[1] / S
    cell_h = canvas_size[0] / S

    center_col = (cx // cell_w).long().clamp(0, S - 1)
    center_row = (cy // cell_h).long().clamp(0, S - 1)

    norm_center_x = (cx % cell_w) / cell_w
    norm_center_y = (cy % cell_h) / cell_h
    norm_bndbox_w = w / canvas_size[0]
    norm_bndbox_h = h / canvas_size[1]

    coord = torch.stack(
        (norm_center_x, norm_center_y, norm_bndbox_w, norm_bndbox_h), dim=-1
    )

    target = torch.zeros(S, S, 5 + C)
    # -1 to match the labels of torchvision
    target[center_row, center_col, :C] = F.one_hot(labels - 1, C).float()
    target[center_row, center_col, C] = 1
    target[center_row, center_col, C + 1 : C + 5] = coord

    return target


def yolo_multi_bbox_to_xyxy(bbox: Tensor, config: YOLOConfig) -> Tensor:
    """
    bbox: Tensor of shape (N, S, S, B, 4) or (S, S, 4)
    bbox format: [x1, y1, x2, y2]

    Returns:
        xyxy: Tensor of shape (N, S, S, B, 4) or (1, S, S, 4)

    Converts the bounding boxes from yolo format to xyxy format
    """
    if bbox.dim() == 3:
        bbox = bbox[None, :, :, None, :]

    N, S1, S2, B, _ = bbox.shape
    canvas_size = config.IMAGE_SIZE
    cell_size_w = canvas_size[1] / S1
    cell_size_h = canvas_size[0] / S2

    # Create meshgrid for cell indices
    y_grid, x_grid = torch.meshgrid(
        torch.arange(S1, device=bbox.device).float(),
        torch.arange(S2, device=bbox.device).float(),
        indexing="ij",
    )

    # Reshape offsets to match bbox shape
    x_grid = x_grid[None, :, :, None, None].expand(N, S1, S2, B, 1)
    y_grid = y_grid[None, :, :, None, None].expand(N, S1, S2, B, 1)

    # Extract cx, cy, w, h from bbox
    cx, cy, w, h = bbox.split(1, dim=-1)

    # Convert cx and cy to absolute coordinates
    abs_center_x = (x_grid + cx) * cell_size_w
    abs_center_y = (y_grid + cy) * cell_size_h

    # Convert w and h to absolute sizes
    width_abs = w * canvas_size[0]
    height_abs = h * canvas_size[1]
    width_abs = (width_abs**2).sqrt()
    height_abs = (height_abs**2).sqrt()

    # Calculate x1, y1, x2, y2
    xyxy = box_convert(
        torch.cat([abs_center_x, abs_center_y, width_abs, height_abs], dim=-1),
        "cxcywh",
        "xyxy",
    )

    # Set zero boxes to remain zero
    zero_mask = (bbox == 0).all(dim=-1, keepdim=True)
    xyxy = xyxy * (~zero_mask)
    xyxy = xyxy.squeeze(-2)

    return xyxy


def yolo_target_to_xyxy(
    target: Tensor, config: YOLOConfig, threshold=0.5
) -> List[Dict[str, Union[Tensor, BoundingBoxes]]]:
    """
    target: Tensor of shape (N, S, S, 5 + C) or (S, S, 5 + C)

    target format: [c1, c2, ..., C, conf, cx, cy, w, h]
        where:
            c1, c2, ..., C = one-hot encoded labels
            conf = 1 if object exists in cell
            cx, cy = center of the bbox relative to the cell
            w, h = width and height of the bbox relative to the image size

    Returns:
        boxes: List of BoundingBoxes objects, one for each sample in the batch
        labels: List of Tensors, one for each sample in the batch
        confidences: List of Tensors, one for each sample in the batch
    """

    C = config.C
    canvas_size = config.IMAGE_SIZE

    if target.dim() == 3:
        target = target.unsqueeze(0)
    batch_size = target.size(0)

    bboxes = yolo_multi_bbox_to_xyxy(target[..., C + 1 :].unsqueeze(-2), config)

    center_batch, center_row, center_col = torch.where(target[..., C] > threshold)

    batch_indices = (
        torch.arange(batch_size, device=target.device)
        .unsqueeze(-1)
        .expand(-1, center_batch.size(0))
    )

    valid_mask = center_batch.unsqueeze(0) == batch_indices

    valid_bboxes = bboxes[center_batch, center_row, center_col]
    valid_labels = torch.argmax(
        target[center_batch, center_row, center_col, :C], dim=-1
    )
    valid_confidences = target[center_batch, center_row, center_col, C]

    bboxes_list = [
        dict(
            boxes=BoundingBoxes(
                valid_bboxes[mask].ceil(),
                format=BoundingBoxFormat.XYXY,
                canvas_size=canvas_size,
            ),
            labels=valid_labels[mask],
            confidences=valid_confidences[mask],
        )
        for mask in valid_mask
    ]

    for i in range(batch_size):
        keep = nms(
            bboxes_list[i]["boxes"],
            bboxes_list[i]["confidences"],
            iou_threshold=0.5,
        )
        bboxes_list[i]["boxes"] = bboxes_list[i]["boxes"][keep]
        bboxes_list[i]["labels"] = bboxes_list[i]["labels"][keep]
        bboxes_list[i]["confidences"] = bboxes_list[i]["confidences"][keep]

    return bboxes_list


def yolo_output_to_xyxy(
    output: Tensor, config: YOLOConfig, threshold=0.5
) -> List[Dict[str, Union[Tensor, BoundingBoxes]]]:
    """
    output: Tensor of shape (N, S, S, B * 5 + C)

    output format: [c1, c2, ..., C, B1[conf, cx, cy, w, h], B2[conf, cx, cy, w, h], B]

    Returns:
        boxes: BoundingBoxes object with shape (N, 4) or (4)
        labels: Tensor of shape (N,) or ()
        confidences: Tensor of shape (N,) or ()

    Returns the boxes, labels, and confidences of the best boxes
    """
    S = config.S
    C = config.C
    B = config.B

    if output.dim() == 3:
        output = output.unsqueeze(0)
    classes = output[..., :C]
    boxes = output[..., C:].reshape(-1, S, S, B, 5)
    box_max_indices = boxes[..., 0].argmax(dim=-1, keepdim=True)
    best_boxes = boxes[
        torch.arange(boxes.size(0))[:, None, None],
        torch.arange(S)[None, :, None],
        torch.arange(S)[None, None, :],
        box_max_indices.squeeze(-1),
    ]
    output_best_boxes = torch.cat([classes, best_boxes], dim=-1)

    bboxes = yolo_target_to_xyxy(output_best_boxes, config, threshold)

    return bboxes


def yolo_resp_bbox(
    output: Tensor, target: Tensor, config: YOLOConfig
) -> Tuple[Tensor, Tensor]:
    S = config.S
    B = config.B
    batch_size = output.size(0)
    size = S * S * batch_size

    output_coords = yolo_multi_bbox_to_xyxy(output, config)
    target_coords = yolo_multi_bbox_to_xyxy(target.unsqueeze(-2), config).squeeze(-2)

    ious = (
        box_iou(output_coords.view(-1, 4), target_coords.view(-1, 4))
        .view(size, B, size)
        .transpose(1, 2)
    )
    ious = ious.diagonal(dim1=0, dim2=1).transpose(0, 1).reshape(batch_size, S, S, B)
    ious, best_bbox = ious.max(dim=-1)

    # if ious is 0 then responsible bbox is the one with the lowest rmse
    zero_batch, zero_i, zero_j = torch.where(ious == 0)
    zero_output = output_coords[zero_batch, zero_i, zero_j]
    zero_target = target_coords[zero_batch, zero_i, zero_j].unsqueeze(1).repeat(1, B, 1)
    rmse = (
        F.mse_loss(
            zero_output,
            zero_target,
            reduction="none",
        )
        .sum(-1)
        .sqrt()
    )

    best_bbox[zero_batch, zero_i, zero_j] = rmse.argmin(dim=-1)

    return best_bbox, ious


if __name__ == "__main__":
    from config_parser import load_config

    config = load_config("yolo_config.yaml")

    def test_xyxy_to_yolo_target():
        coord1 = torch.randint(0, 224, (1, 2))
        coord2 = coord1 + torch.randint(0, 224, (1, 2))
        boxes = torch.cat([coord1, coord2], 1)
        boxes = BoundingBoxes(
            boxes,
            format=BoundingBoxFormat.XYXY,
            canvas_size=(448, 448),
        )
        print(boxes)
        target = xyxy_to_yolo_target(boxes, torch.tensor([19]), config)

        print(target.shape)
        # print(target[*torch.where(target[..., 20] == 1), :])
        print(yolo_target_to_xyxy(target, config))

    def random_output_and_target(BATCH_SIZE=1, S=7, B=2, C=20):
        torch.manual_seed(0)
        classes = F.one_hot(
            torch.randint(0, C, (S, S)),
            num_classes=C,
        )
        coords = torch.randn(S, S, B * 5)
        output = torch.cat((classes, coords), dim=-1)
        output.unsqueeze_(0)
        output = torch.cat([output] * BATCH_SIZE, dim=0)

        target_classes = F.one_hot(
            torch.randint(0, C, (S, S)),
            num_classes=C,
        )
        target_coords = torch.cat(
            (torch.randint(0, 2, (S, S, 1)), torch.rand(S, S, 4)), dim=-1
        )
        target = torch.cat((target_classes, target_coords), dim=2)
        target.unsqueeze_(0)
        target = torch.cat([target] * BATCH_SIZE, dim=0)

        return output, target

    def different_batch_size(b):
        pred, target = random_output_and_target(b)
        try:
            yolo_output_to_xyxy(pred, config)
            yolo_target_to_xyxy(target, config)
        except Exception as e:
            print(e)

    different_batch_size(1)
    different_batch_size(5)
    test_xyxy_to_yolo_target()
