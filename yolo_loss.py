import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from config_parser import config

from yolo_utils import yolo_resp_bbox

S = config.S
B = config.B
C = config.C


class YOLOLoss(nn.Module):
    def __init__(self, config=config):
        super().__init__()
        self.S = config.S
        self.B = config.B
        self.C = config.C
        self.L_coord = config.L_coord
        self.L_noobj = config.L_noobj

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        output: (batch_size, S, S, B * 5 + C)
        target: (batch_size, S, S, 5 + C)

        output format: [c1, c2, ..., C, B1[conf, cx, cy, w, h], B2[conf, cx, cy, w, h], B]
        target format: [c1, c2, ..., C, conf, cx, cy, w, h]

        loss =
            L_coord * obj * [(x - x_hat)^2 + (y - y_hat)^2]
            L_coord * obj * [w - sqrt(w_hat))^2 + (h - sqrt(h_hat))^2]
            obj * [(conf - conf_hat)^2]
            L_noobj * noobj * [(conf - conf_hat)^2]
            obj * [(c - c_hat)^2]

        where:
            obj = 1 if object exists in cell
            noobj = 1 if no object exists in cell
        """
        S = self.S
        B = self.B
        C = self.C
        BATCH_SIZE = output.size(0)

        obj_mask = target[..., C] == 1
        noobj_mask = ~obj_mask

        # Which boxes in each cell is responsible for the prediction
        output_boxes = output[..., C:].contiguous().view(-1, S, S, B, 5)
        target_boxes = target[..., C:].contiguous().view(-1, S, S, 5)

        best_bbox = yolo_resp_bbox(output_boxes[..., 1:], target_boxes[..., 1:], config)

        # Use advanced indexing
        resp_boxes = output_boxes[
            torch.arange(BATCH_SIZE)[:, None, None],
            torch.arange(S)[None, :, None],
            torch.arange(S)[None, None, :],
            best_bbox,
        ]

        resp_coords = resp_boxes[..., 1:]
        target_coords = target_boxes[..., 1:]

        # Box Loss

        center_loss = F.mse_loss(
            target_coords[obj_mask][..., :2],
            resp_coords[obj_mask][..., :2],
            reduction="sum",
        )

        wh_loss = F.mse_loss(
            target_coords[obj_mask][..., 2:],
            (resp_coords[obj_mask][..., 2:] ** 2).sqrt(),
            reduction="sum",
        )

        box_loss = self.L_coord * (center_loss + wh_loss)

        # Object Loss

        conf_loss = F.mse_loss(
            resp_boxes[obj_mask][..., 0],
            target_boxes[obj_mask][..., 0],
            reduction="sum",
        )

        # No Object Loss
        noobj_loss = self.L_noobj * F.mse_loss(
            resp_boxes[noobj_mask][..., 0],
            target_boxes[noobj_mask][..., 0],
            reduction="sum",
        )

        # Class Loss

        class_loss = F.mse_loss(
            output[..., :C][obj_mask],
            target[..., :C][obj_mask],
            reduction="sum",
        )

        loss = box_loss + conf_loss + noobj_loss + class_loss
        loss = loss / BATCH_SIZE

        return loss


if __name__ == "__main__":
    output = torch.rand(2, S, S, B * 5 + C)
    target = torch.rand(2, S, S, 5 + C)

    loss = YOLOLoss()
    print(loss(output, target))