import torch

from torchvision.ops import box_iou

from safetensors import safe_open

from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy
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

    metric_forward = metric(pred_bboxes_list, target_bboxes_list)

    # for key, value in metric_forward.items():
    #     metric_forward[key] = accelerator.gather(value.to(accelerator.device))

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
    accelerator: Accelerator,
    writer: SummaryWriter,
    metric: MeanAveragePrecision,
    running_loss: float,
    running_iou: float,
    batch_idx: int,
    epoch: int,
    prefix: str,
) -> Tuple[float, float, dict, float]:
    metric_compute = metric.compute()

    # for key, value in metric_compute.items():
    #     metric_compute[key] = accelerator.gather(value.to(accelerator.device))

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

    torch.save(
        {
            "epoch": epoch,
            "config": str(config),
            "mAP": map_value,
            "mAP50": map50,
            "metric_compute": metric_compute,
            "loss": loss,
        },
        path / "metrics.pt",
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
) -> int:
    with safe_open(path / "model.safetensors", framework="pt") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})

    start_epoch = torch.load(
        path / "metrics.pt", map_location="cpu", weights_only=True
    )["epoch"]
    return start_epoch
