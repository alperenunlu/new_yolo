import torch

from torchvision.ops import box_iou

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy
from yolo_viz_utils import draw_yolo_grid_from_dict

from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing import Tuple, Optional
from config_parser import YOLOConfig
from torch.utils.tensorboard import SummaryWriter


def log_progress(
    writer: SummaryWriter,
    metric: MeanAveragePrecision,
    inputs: Tensor,
    outputs: Tensor,
    targets: Tensor,
    loss: Tensor,
    avg_iou: Tensor,
    global_step: int,
    batch_idx: int,
    prefix: str,
    config: YOLOConfig,
    lr: Optional[float] = None,
) -> None:
    threshold = 0.5

    pred_bboxes, labels, confidences = yolo_output_to_xyxy(
        outputs, config=config, threshold=threshold
    )
    target_bboxes, target_labels, _ = yolo_target_to_xyxy(
        targets, config=config, threshold=threshold
    )

    prediction_boxes_list = []
    target_boxes_list = []
    for pred_bbox, label, confidence, target_bbox, target_label in zip(
        pred_bboxes, labels, confidences, target_bboxes, target_labels
    ):
        scores = box_iou(pred_bbox, target_bbox).max(dim=1).values
        prediction_boxes_list.append(
            {
                "boxes": pred_bbox,
                "labels": label,
                "scores": scores,
                "confidences": confidence,
            }
        )
        target_boxes_list.append({"boxes": target_bbox, "labels": target_label})

    metric_forward = metric(prediction_boxes_list, target_boxes_list)

    writer.add_scalar(f"{prefix}/Loss", loss.item(), global_step)
    writer.add_scalar(f"{prefix}/IoU", avg_iou.item(), global_step)
    writer.add_scalar(f"{prefix}/mAP", metric_forward["map"], global_step)
    writer.add_scalar(f"{prefix}/mAP50", metric_forward["map_50"], global_step)
    writer.add_scalar(f"{prefix}/mAP75", metric_forward["map_75"], global_step)
    writer.add_scalar(f"{prefix}/mAP_small", metric_forward["map_small"], global_step)
    writer.add_scalar(f"{prefix}/mAP_medium", metric_forward["map_medium"], global_step)
    writer.add_scalar(f"{prefix}/mAP_large", metric_forward["map_large"], global_step)
    writer.add_scalar(f"{prefix}/mAR-1", metric_forward["mar_1"], global_step)
    writer.add_scalar(f"{prefix}/mAR-10", metric_forward["mar_10"], global_step)
    writer.add_scalar(f"{prefix}/mAR-100", metric_forward["mar_100"], global_step)

    if lr is not None:
        writer.add_scalar(f"{prefix}/Learning Rate", lr, global_step)

    if batch_idx % 25 == 0:
        writer.add_image(
            f"{prefix}/SampleDetections",
            draw_yolo_grid_from_dict(
                images=inputs,
                outputs_list=prediction_boxes_list,
                targets_list=target_boxes_list,
                config=config,
            ),
            global_step,
        )


def log_epoch_summary(
    writer: SummaryWriter,
    metric: MeanAveragePrecision,
    running_loss: float,
    running_iou: float,
    batch_idx: int,
    epoch: int,
    prefix: str,
) -> Tuple[float, float, dict, float]:
    metric_compute = metric.compute()
    map_value = metric_compute["map"]
    map50 = metric_compute["map_50"]

    writer.add_scalar(f"{prefix}/mAP", map_value, epoch)
    writer.add_scalar(f"{prefix}/mAP50", map50, epoch)
    writer.add_scalar(f"{prefix}/Loss", running_loss / (batch_idx + 1), epoch)
    writer.add_scalar(f"{prefix}/IoU", running_iou / (batch_idx + 1), epoch)

    return map_value, map50, metric_compute, running_loss / (batch_idx + 1)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: YOLOConfig,
    map_value: float,
    map50: float,
    metric_compute: dict,
    loss: float,
    path: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": config,
            "mAP": map_value,
            "mAP50": map50,
            "metric_compute": metric_compute,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, path: str
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int
]:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler, checkpoint["epoch"]
