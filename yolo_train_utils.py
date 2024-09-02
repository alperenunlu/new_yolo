import torch

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy
from yolo_viz_utils import draw_yolo_grid

from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing import Tuple, Optional
from config_parser import YOLOCONFIG
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
    config: YOLOCONFIG,
    lr: Optional[float] = None,
) -> None:
    pred_boxes = [
        {"boxes": box, "labels": label, "scores": confidence}
        for box, label, confidence in zip(*yolo_output_to_xyxy(outputs, config=config))
    ]
    target_boxes = [
        {"boxes": box, "labels": label}
        for box, label, _ in zip(*yolo_target_to_xyxy(targets, config=config))
    ]
    metric_forward = metric(pred_boxes, target_boxes)

    if batch_idx % 50 == 0:
        writer.add_image(
            f"{prefix}/SampleDetections",
            draw_yolo_grid(inputs, outputs, targets, config, threshold=0.5),
            global_step,
        )

    writer.add_scalar(f"{prefix}/Loss", loss.item(), global_step)
    writer.add_scalar(f"{prefix}/IoU", avg_iou.item(), global_step)
    writer.add_scalar(f"{prefix}/mAP", metric_forward["map"], global_step)
    writer.add_scalar(f"{prefix}/mAP50", metric_forward["map_50"], global_step)
    writer.add_scalar(f"{prefix}/mAP75", metric_forward["map_75"], global_step)
    writer.add_scalar(f"{prefix}/mAP_small", metric_forward["map_small"], global_step)
    writer.add_scalar(f"{prefix}/mAP_medium", metric_forward["map_medium"], global_step)
    writer.add_scalar(f"{prefix}/mAP_large", metric_forward["map_large"], global_step)

    if lr is not None:
        writer.add_scalar(f"{prefix}/Learning Rate", lr, global_step)


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
    config: YOLOCONFIG,
    map_value: float,
    map50: float,
    metric_compute: dict,
    loss: float,
    path: str,
):
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
):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["config"]
