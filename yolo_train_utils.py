from itertools import chain

import torch

from torchvision.ops import box_iou

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy
from yolo_viz_utils import draw_yolo_grid_from_dict

from typing import Tuple, Optional
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from config_parser import YOLOConfig
from accelerate import Accelerator


def log_progress(
    accelerator: Accelerator,
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

    pred_bboxes_list = yolo_output_to_xyxy(outputs, config=config, threshold=threshold)
    target_bboxes_list = yolo_target_to_xyxy(
        targets, config=config, threshold=threshold
    )

    for pred_bboxes, target_bboxes in zip(pred_bboxes_list, target_bboxes_list):
        scores = box_iou(pred_bboxes["boxes"], target_bboxes["boxes"]).max(dim=1).values
        pred_bboxes["scores"] = scores

    pred_bboxes_list, target_bboxes_list = accelerator.gather_for_metrics(
        (pred_bboxes_list, target_bboxes_list)
    )

    if isinstance(type(pred_bboxes_list), list):
        pred_bboxes_list = list(chain(*pred_bboxes_list))
        target_bboxes_list = list(chain(*target_bboxes_list))

    metric_forward = metric(pred_bboxes_list, target_bboxes_list)

    metrics = {
        "Loss": loss.item(),
        "IoU": avg_iou.item(),
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
        writer.add_scalar(f"{prefix}/{name}", value, global_step)

    if batch_idx % 25 == 0:
        writer.add_image(
            f"{prefix}/SampleDetections",
            draw_yolo_grid_from_dict(
                images=inputs,
                outputs_list=pred_bboxes_list,
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
    map_value = metric_compute["map"]
    map50 = metric_compute["map_50"]

    # writer.add_scalar(f"{prefix}/mAP", map_value, epoch)
    # writer.add_scalar(f"{prefix}/mAP50", map50, epoch)
    # writer.add_scalar(f"{prefix}/Loss", running_loss / (batch_idx + 1), epoch)
    # writer.add_scalar(f"{prefix}/IoU", running_iou / (batch_idx + 1), epoch)

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
    accelerator: Accelerator,
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
            "epoch": epoch,
            "config": config,
            "mAP": map_value,
            "mAP50": map50,
            "metric_compute": metric_compute,
            "loss": loss,
        },
        path,
    )
    accelerator.save_state(path)


def load_checkpoint(accelerator: Accelerator, path: str) -> int:
    start_epoch = torch.load(path)["epoch"]
    accelerator.load_state(path)
    return start_epoch
