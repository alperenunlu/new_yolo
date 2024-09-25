import torch

from safetensors.torch import load_model

from yolo_utils import yolo_pred_to_xyxy, yolo_target_to_xyxy
from yolo_viz_utils import draw_yolo_grid_from_dict

from typing import Tuple, Optional
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from config_parser import YOLOConfig
from accelerate import Accelerator
from pathlib import Path


@torch.no_grad()
def log_progress(
    writer: SummaryWriter,
    metric: MeanAveragePrecision,
    inputs: Tensor,
    preds: Tensor,
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

    pred_bboxes_list = yolo_pred_to_xyxy(preds, config=config, threshold=threshold)
    target_bboxes_list = yolo_target_to_xyxy(
        targets, config=config, threshold=threshold
    )

    metric_forward = metric(pred_bboxes_list, target_bboxes_list)

    metrics = {
        "Loss": loss,
        "IoU": avg_iou,
        "mAP": metric_forward["map"],
        "mAP50": metric_forward["map_50"],
        "mAP75": metric_forward["map_75"],
        "mAP_small": metric_forward["map_small"],
        "mAP_medium": metric_forward["map_medium"],
        "mAP_large": metric_forward["map_large"],
        "mAR_1": metric_forward["mar_1"],
        "mAR_10": metric_forward["mar_10"],
        "mAR_100": metric_forward["mar_100"],
    }

    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value.mean(), global_step)

    if lr:
        writer.add_scalar(f"{prefix}/LearningRate", lr, global_step)

    if batch_idx % 25 == 0:
        writer.add_image(
            f"{prefix}/SampleDetections",
            draw_yolo_grid_from_dict(
                images=inputs,
                preds_list=pred_bboxes_list,
                targets_list=target_bboxes_list,
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

    map_value = metric_compute["map"].mean().item()
    map50 = metric_compute["map_50"].mean().item()

    metrics = {
        "mAP": map_value,
        "mAP50": map50,
        "Loss": running_loss / (batch_idx + 1),
        "IoU": running_iou / (batch_idx + 1),
    }

    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, epoch)

    return map_value, map50, metric_compute, running_loss / (batch_idx + 1)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
    epoch: int,
    config: YOLOConfig,
    map_value: float,
    map50: float,
    metric_compute: dict,
    loss: float,
    path: Path,
) -> None:
    accelerator.wait_for_everyone()
    accelerator.save_model(
        accelerator.unwrap_model(model),
        path,
    )
    accelerator.save(optimizer.state_dict(), path / "optimizer.pt")
    accelerator.save(scheduler.state_dict(), path / "scheduler.pt")

    torch.save(
        {
            "epoch": epoch,
            "mAP": map_value,
            "mAP50": map50,
            "metric_compute": metric_compute,
            "loss": loss,
        },
        path / "metrics.pt",
    )

    torch.save(
        config,
        path / "config.pt",
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
) -> int:
    missing, unexpected = load_model(model, path / "model.safetensors")
    assert (
        not missing and not unexpected
    ), f"Missing: {missing}, Unexpected: {unexpected}"

    optimizer.load_state_dict(
        torch.load(path / "optimizer.pt", weights_only=True, map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(path / "scheduler.pt", weights_only=True, map_location="cpu")
    )

    start_epoch = torch.load(
        path / "metrics.pt", map_location="cpu", weights_only=True
    )["epoch"]
    return start_epoch
