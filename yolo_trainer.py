import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from yolo_train_utils import log_progress, log_epoch_summary

from typing import Tuple
from config_parser import YOLOCONFIG


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    config: YOLOCONFIG,
) -> Tuple[float, float, dict, float]:
    model.train()
    running_loss = running_iou = 0.0
    metric = MeanAveragePrecision().to(device)

    loop = tqdm(loader, total=len(loader), desc=f"Training Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(loop):
        global_step = epoch * len(loader) + batch_idx
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss, avg_iou = criterion(outputs, targets)
        loss.backward()

        if (batch_idx + 1) % (config.BATCH_SIZE // config.SUBDIVISION) == 0:
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()
        running_iou += avg_iou.item()

        log_progress(
            writer=writer,
            metric=metric,
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            loss=loss,
            avg_iou=avg_iou,
            global_step=global_step,
            batch_idx=batch_idx,
            prefix="Training",
            config=config,
            lr=optimizer.param_groups[0]["lr"],
        )

        loop.set_postfix(
            {
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "iou": f"{running_iou / (batch_idx + 1):.4f}",
            }
        )

    if (batch_idx + 1) % (config.BATCH_SIZE // config.SUBDIVISION) != 0:
        optimizer.step()
        optimizer.zero_grad()

    return log_epoch_summary(
        writer, metric, running_loss, running_iou, batch_idx, epoch, "Train"
    )


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    config: YOLOCONFIG,
) -> Tuple[float, float, dict, float]:
    model.eval()
    running_loss = running_iou = 0.0
    metric = MeanAveragePrecision().to(device)

    loop = tqdm(loader, total=len(loader), desc=f"Validating Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(loop):
        global_step = epoch * len(loader) + batch_idx
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss, avg_iou = criterion(outputs, targets)

        running_loss += loss.item()
        running_iou += avg_iou.item()

        log_progress(
            writer=writer,
            metric=metric,
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            loss=loss,
            avg_iou=avg_iou,
            global_step=global_step,
            batch_idx=batch_idx,
            config=config,
            prefix="Validation",
        )

        loop.set_postfix(
            {
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "iou": f"{running_iou / (batch_idx + 1):.4f}",
            }
        )

    return log_epoch_summary(
        writer, metric, running_loss, running_iou, batch_idx, epoch, "Validation"
    )
