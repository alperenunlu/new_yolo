import torch

from torchvision.ops import box_convert

from config_parser import config


def xyxy_to_yolo_target(boxes, labels, config=config):
    """
    Convert from xyxy format to YOLO format
    """
    S = config.S
    B = config.B
    C = config.C
    canvas_size_x = config.IMAGE_SIZE[0]
    canvas_size_y = config.IMAGE_SIZE[1]

    # x1, y1, x2, y2 = boxes.unbind(1)

    # cx = (x1 + x2) / 2
    # cy = (y1 + y2) / 2
    # w = x2 - x1
    # h = y2 - y1

    cx, cy, w, h = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh").unbind(1)

    cell_size_x = canvas_size_x / S
    cell_size_y = canvas_size_y / S

    cell_index_x = (cx / cell_size_x).long().clamp(0, config.S - 1)
    cell_index_y = (cy / cell_size_y).long().clamp(0, config.S - 1)

    norm_cx = cx % cell_size_x / cell_size_x
    norm_cy = cy % cell_size_y / cell_size_y
    norm_w = w / canvas_size_x
    norm_h = h / canvas_size_y

    target = torch.zeros(S, S, B * 5 + C)
    target[cell_index_x, cell_index_y, labels - 1] = 1
    target[cell_index_x, cell_index_y, C] = 1
    target[cell_index_x, cell_index_y, C + 1 : C + 5] = torch.stack(
        [norm_cx, norm_cy, norm_w, norm_h], 1
    )
    return target


def yolo_target_to_xyxy(target, config=config):
    """
    Convert from YOLO format to xyxy format
    """
    S = config.S
    canvas_size_x = config.IMAGE_SIZE[0]
    canvas_size_y = config.IMAGE_SIZE[1]

    cell_size_x = canvas_size_x / S
    cell_size_y = canvas_size_y / S

    cell_index_x, cell_index_y = torch.where(target[..., 20] == 1)
    norm_cx, norm_cy, norm_w, norm_h = target[cell_index_x, cell_index_y, 21:25].unbind(
        1
    )

    cx = (cell_index_x + norm_cx) * cell_size_x
    cy = (cell_index_y + norm_cy) * cell_size_y
    w = norm_w * canvas_size_x
    h = norm_h * canvas_size_y

    x1, y1, x2, y2 = box_convert(
        torch.stack([cx, cy, w, h], 1), in_fmt="cxcywh", out_fmt="xyxy"
    ).unbind(1)

    boxes = torch.stack([x1, y1, x2, y2], 1)
    labels = torch.argmax(target[cell_index_x, cell_index_y, :20], 1) + 1
    return boxes, labels


if __name__ == "__main__":
    torch.manual_seed(0)
    coord1 = torch.randint(0, 100, (1, 2))
    coord2 = coord1 + torch.randint(0, 100, (1, 2))
    boxes = torch.cat([coord1, coord2], 1)
    print(boxes)
    target = xyxy_to_yolo_target(boxes, torch.tensor([19]))

    print(target.shape)
    print(target[*torch.where(target[..., 20] == 1), :])
    print(yolo_target_to_xyxy(target))
