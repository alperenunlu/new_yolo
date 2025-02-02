import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_utils import yolo_resp_bbox

from typing import Tuple
from torch import Tensor
from config_parser import YOLOConfig


class YOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.config = config

    def forward(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        pred: (batch_size, S, S, B * 5 + C)
        target: (batch_size, S, S, 5 + C)

        pred format: [c1, c2, ..., C, B1[conf, cx, cy, w, h], B2[conf, cx, cy, w, h], B]
        target format: [c1, c2, ..., C, conf, cx, cy, w, h]

        loss =
            L_coord * obj * [(x - x_hat)^2 + (y - y_hat)^2]
            L_coord * obj * [sqrt(w) - sqrt(sqrt(w_hat))^2) + (sqrt(h) - sqrt(sqrt(h_hat))^2)]
            obj * [(conf - conf_hat)^2]
            L_noobj * noobj * [(conf - conf_hat)^2]
            obj * [(c - c_hat)^2]

        where:
            obj = 1 if object exists in cell
            noobj = 1 if no object exists in cell
        """
        S = self.config.S
        B = self.config.B
        C = self.config.C
        BATCH_SIZE = pred.size(0)

        obj_mask = (target[..., C] == 1).view(BATCH_SIZE, S, S, 1)
        noobj_mask = ~obj_mask

        pred_boxes = pred[..., C:].view(-1, S, S, B, 5)
        target_boxes = target[..., C:].view(-1, S, S, 5)

        best_bbox, ious = yolo_resp_bbox(
            pred_boxes[..., 1:],
            target_boxes[..., 1:],
            self.config,
        )

        pred_boxes = pred_boxes[
            torch.arange(pred_boxes.size(0)).view(-1, 1, 1).expand_as(best_bbox),
            torch.arange(pred_boxes.size(1)).view(1, -1, 1).expand_as(best_bbox),
            torch.arange(pred_boxes.size(2)).view(1, 1, -1).expand_as(best_bbox), 
            best_bbox
        ]

        # Box Loss
        center_loss = torch.sum(
            ((target_boxes[..., 1:3] - pred_boxes[..., 1:3]) ** 2) * obj_mask
        )

        wh_loss = torch.sum(
            ((
                target_boxes[..., 3:].sqrt()
                - (pred_boxes[..., 3:] ** 2).sqrt().sqrt()
            ) ** 2) * obj_mask
        )

        box_loss = self.config.L_coord * (center_loss + wh_loss)

        # Object Loss
        if self.config.Rescore:
            conf_loss = self.config.L_obj * torch.sum(
                ((ious - pred_boxes[..., 0]) ** 2) * obj_mask
            )
        else:
            conf_loss = self.config.L_obj * torch.sum(
                ((target_boxes[..., 0] - pred_boxes[..., 0]) ** 2) * obj_mask.squeeze(-1)
            )

        # No Object Loss
        noobj_loss = self.config.L_noobj * torch.sum(
            ((target_boxes[..., 0] - pred_boxes[..., 0]) ** 2) * noobj_mask.squeeze(-1)
        )

        # Class Loss
        class_loss = self.config.L_class * torch.sum(
            ((pred[..., :C] - target[..., :C]) ** 2) * obj_mask
        )

        loss = box_loss + conf_loss + noobj_loss + class_loss

        avg_iou = ious[obj_mask.squeeze(-1)].mean()

        return loss, avg_iou


if __name__ == "__main__":

    def random_pred_and_target(BATCH_SIZE=1, S=7, B=2, C=20):
        torch.manual_seed(0)
        classes = F.one_hot(
            torch.randint(0, C, (S, S)),
            num_classes=C,
        )
        coords = torch.rand(S, S, B * 5)
        pred = torch.cat((classes, coords), dim=-1)
        pred.unsqueeze_(0)
        pred = torch.cat([pred] * BATCH_SIZE, dim=0)

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

        return pred, target

    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    loss = YOLOLoss(config)
    pred, target = random_pred_and_target()
    print(loss(pred, target))
