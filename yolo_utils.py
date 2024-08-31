import torch

import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou

from typing import Tuple, List
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torch import Tensor

from config_parser import config


def yolo_multi_bbox_to_xyxy(bbox: Tensor, config=config) -> Tensor:
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
    cell_size_w = canvas_size[0] / S1
    cell_size_h = canvas_size[1] / S2

    # Create meshgrid for cell indices
    cx_offset, cy_offset = torch.meshgrid(
        torch.arange(S1, device=bbox.device).float(),
        torch.arange(S2, device=bbox.device).float(),
        indexing="ij",
    )

    # Reshape offsets to match bbox shape
    cx_offset = cx_offset.view(1, S1, S2, 1, 1).expand(N, S1, S2, B, 1)
    cy_offset = cy_offset.view(1, S1, S2, 1, 1).expand(N, S1, S2, B, 1)

    # Extract cx, cy, w, h from bbox
    cx, cy, w, h = bbox.split(1, dim=-1)

    # Convert cx and cy to absolute coordinates
    cx_abs = (cx_offset + cx) * cell_size_w
    cy_abs = (cy_offset + cy) * cell_size_h

    # Convert w and h to absolute sizes
    w_abs = w * canvas_size[0]
    h_abs = h * canvas_size[1]
    w_abs = (w_abs**2).sqrt()
    h_abs = (h_abs**2).sqrt()

    # Calculate x1, y1, x2, y2
    xyxy = box_convert(
        torch.cat([cx_abs, cy_abs, w_abs, h_abs], dim=-1), "cxcywh", "xyxy"
    )

    # Set zero boxes to remain zero
    zero_mask = (bbox == 0).all(dim=-1, keepdim=True)
    xyxy = xyxy * (~zero_mask)
    xyxy = xyxy.squeeze(-2)

    return xyxy


def xyxy_to_yolo_target(boxes: BoundingBoxes, labels: Tensor, config=config) -> Tensor:
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
    cell_w = canvas_size[0] / S
    cell_h = canvas_size[1] / S

    center_row = (cx / cell_w).floor().long().clamp(0, S - 1)
    center_col = (cy / cell_h).floor().long().clamp(0, S - 1)

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


def yolo_target_to_xyxy(
    target: Tensor, threshold=0.5, config=config
) -> Tuple[List[BoundingBoxes], List[Tensor], List[Tensor]]:
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

    # Get indices of cells with confidence above threshold
    center_batch, center_row, center_col = torch.where(target[..., C] > threshold)

    # Create a tensor of batch indices
    batch_indices = (
        torch.arange(batch_size, device=target.device)
        .unsqueeze(-1)
        .expand(-1, center_batch.size(0))
    )

    # Mask for valid detections
    valid_mask = center_batch.unsqueeze(0) == batch_indices

    # Extract valid bboxes, labels, and confidences
    valid_bboxes = bboxes[center_batch, center_row, center_col]
    valid_labels = (
        torch.argmax(target[center_batch, center_row, center_col, :C], dim=-1) + 1
    )
    valid_confidences = target[center_batch, center_row, center_col, C]

    # Split the results by batch
    boxes_list = [
        BoundingBoxes(
            valid_bboxes[mask].ceil(),
            format=BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        )
        for mask in valid_mask
    ]

    labels_list = [valid_labels[mask] for mask in valid_mask]
    confidences_list = [valid_confidences[mask] for mask in valid_mask]

    return boxes_list, labels_list, confidences_list


def yolo_output_to_xyxy(
    output: Tensor, threshold=0.5, config=config
) -> Tuple[BoundingBoxes, Tensor, Tensor]:
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

    bboxes, labels, confidences = yolo_target_to_xyxy(
        output_best_boxes, threshold, config
    )

    return bboxes, labels, confidences


def yolo_resp_bbox(output, target, config=config):
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

    return best_bbox


if __name__ == "__main__":

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
        target = xyxy_to_yolo_target(boxes, torch.tensor([19]))

        print(target.shape)
        print(target[*torch.where(target[..., 20] == 1), :])
        print(yolo_target_to_xyxy(target))

    def random_output_and_target(BATCH_SIZE=1, S=7, B=2, C=20):
        torch.manual_seed(0)
        classes = F.one_hot(
            torch.randint(
                0,
                C,
                (
                    S,
                    S,
                ),
            ),
            num_classes=C,
        )
        coords = torch.randn(S, S, B * 5)
        output = torch.cat((classes, coords), dim=-1)
        output.unsqueeze_(0)
        output = torch.cat([output] * BATCH_SIZE, dim=0)

        target_classes = F.one_hot(
            torch.randint(
                0,
                C,
                (
                    S,
                    S,
                ),
            ),
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
            yolo_output_to_xyxy(pred)
            yolo_target_to_xyxy(target)
        except Exception as e:
            print(e)

    different_batch_size(1)
    different_batch_size(5)
    # test_xyxy_to_yolo_target()