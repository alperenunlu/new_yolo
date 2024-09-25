import torch
from accelerate.utils.tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from yolo_train_utils import log_progress, log_epoch_summary

from typing import Tuple, Optional
from config_parser import YOLOConfig
from accelerate import Accelerator


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    accelerator: Accelerator,
    metric: MeanAveragePrecision,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    config: YOLOConfig,
) -> Tuple[float, float, dict, float]:
    model.train()
    running_map50 = running_loss = running_iou = 0.0

    loop = tqdm(loader, total=len(loader), desc=f"Training Epoch {epoch}", leave=False)
    for batch_idx, (inputs, targets) in enumerate(loop):
        with accelerator.accumulate(model):
            global_step = epoch * len(loader) + batch_idx

            preds = model(inputs)
            loss, avg_iou = criterion(preds, targets)

            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()

            loss, avg_iou = accelerator.gather_for_metrics((loss, avg_iou))

            map_50 = log_progress(
                writer=writer,
                metric=metric,
                inputs=inputs,
                preds=preds,
                targets=targets,
                loss=loss,
                avg_iou=avg_iou,
                global_step=global_step,
                batch_idx=batch_idx,
                prefix="Train",
                config=config,
                lr=optimizer.param_groups[0]["lr"],
            )

            running_loss += loss.mean().item()
            running_iou += avg_iou.mean().item()
            running_map50 += map_50

            loop.set_postfix(
                {
                    "loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "iou": f"{running_iou / (batch_idx + 1):.4f}",
                    "map50": f"{running_map50 / (batch_idx + 1):.4f}",
                }
            )

    summary = log_epoch_summary(
        writer,
        metric,
        running_loss,
        running_iou,
        batch_idx,
        epoch,
        "Epoch/Train",
    )

    metric.reset()

    return summary


@torch.no_grad()
def valid_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    accelerator: Accelerator,
    metric: MeanAveragePrecision,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    config: YOLOConfig,
) -> Tuple[float, float, dict, float]:
    model.eval()
    running_loss = running_iou = 0.0

    loop = tqdm(
        loader, total=len(loader), desc=f"Validating Epoch {epoch}", leave=False
    )
    for batch_idx, (inputs, targets) in enumerate(loop):
        global_step = epoch * len(loader) + batch_idx

        preds = model(inputs)
        loss, avg_iou = criterion(preds, targets)

        loss, avg_iou = accelerator.gather_for_metrics((loss, avg_iou))
        running_loss += loss.mean().item()
        running_iou += avg_iou.mean().item()

        log_progress(
            writer=writer,
            metric=metric,
            inputs=inputs,
            preds=preds,
            targets=targets,
            loss=loss,
            avg_iou=avg_iou,
            global_step=global_step,
            batch_idx=batch_idx,
            config=config,
            prefix="Valid",
        )

        loop.set_postfix(
            {
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "iou": f"{running_iou / (batch_idx + 1):.4f}",
            }
        )

    summary = log_epoch_summary(
        writer,
        metric,
        running_loss,
        running_iou,
        batch_idx,
        epoch,
        "Epoch/Valid",
    )

    metric.reset()

    return summary
