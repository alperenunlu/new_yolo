import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from config_parser import YOLOConfig

from yolo_utils import yolo_resp_bbox


class YOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.config = config

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
        S = self.config.S
        B = self.config.B
        C = self.config.C
        BATCH_SIZE = output.size(0)

        obj_mask = target[..., C] == 1
        noobj_mask = ~obj_mask

        output_boxes = output[..., C:].view(-1, S, S, B, 5)
        target_boxes = target[..., C:].view(-1, S, S, 5)

        best_bbox, ious = yolo_resp_bbox(
            output_boxes[..., 1:],
            target_boxes[..., 1:],
            self.config,
        )

        resp_boxes = output_boxes.gather(
            -2,
            best_bbox.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, output_boxes.size(-1)),
        ).squeeze(-2)

        resp_coords = resp_boxes[..., 1:]
        target_coords = target_boxes[..., 1:]

        # Box Loss
        obj_resp_coords = resp_coords[obj_mask]
        obj_target_coords = target_coords[obj_mask]

        center_loss = torch.sum(
            (obj_target_coords[..., :2] - obj_resp_coords[..., :2]) ** 2
        )

        wh_loss = torch.sum(
            (obj_target_coords[..., 2:] - (obj_resp_coords[..., 2:] ** 2).sqrt()) ** 2
        )

        box_loss = self.config.L_coord * (center_loss + wh_loss)

        # Object Loss
        conf_loss = self.config.L_obj * torch.sum(
            (resp_boxes[obj_mask][..., 0] - target_boxes[obj_mask][..., 0]) ** 2
        )

        # No Object Loss
        noobj_loss = self.config.L_noobj * torch.sum(
            (resp_boxes[noobj_mask][..., 0] - target_boxes[noobj_mask][..., 0]) ** 2
        )

        # Class Loss
        class_loss = self.config.L_class * torch.sum(
            (output[obj_mask][..., :C] - target[obj_mask][..., :C]) ** 2
        )

        loss = box_loss + conf_loss + noobj_loss + class_loss
        loss = loss / BATCH_SIZE

        avg_iou = ious[obj_mask].mean()

        return loss, avg_iou


if __name__ == "__main__":

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

    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    loss = YOLOLoss(config)
    output, target = random_output_and_target()
    print(loss(output, target))
