import torch
from accelerate.utils.tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from yolo_train_utils import log_progress, log_epoch_summary

from typing import Tuple, Optional
from config_parser import YOLOConfig
from accelerate import Accelerator

def check_model_params(model):
    for name, param in model.named_parameters():
        if param is None:
            print(f"[WARNING] Parameter {name} is None!")
            return False  # indicate a problem
        if param.data is None:
            print(f"[WARNING] Parameter data for {name} is None!")
            return False
    return True  # all parameters are okay


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

    loop = tqdm(
        loader,
        total=len(loader),
        desc=f"Training Epoch {epoch}:",
        bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )
    for batch_idx, (inputs, targets) in enumerate(loop):
        with accelerator.accumulate(model):
            global_step = epoch * len(loader) + batch_idx

            preds = model(inputs)
            loss, avg_iou = criterion(preds, targets)

            optimizer.zero_grad()
            accelerator.backward(loss)

            optimizer.step()
            if scheduler:
                scheduler.step()

            if batch_idx in [85, 86]:
                # Create a dictionary of gradients
                grads = {}
                for name, param in model.named_parameters():
                    # Check if gradient exists and detach it from the graph
                    if param.grad is not None:
                        grads[name] = param.grad.detach().cpu()

                # Save the checkpoint including gradients
                torch.save({
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss.item(),
                    "preds": preds.detach().cpu(),
                    "inputs": inputs.detach().cpu(),
                    "targets": targets.detach().cpu(),
                    "gradients": grads,  # added gradients
                }, f"checkpoint_{batch_idx}.pth")
                if batch_idx == 86:
                    __import__('sys').exit(0)

            print(loss)
            loss, avg_iou = accelerator.gather_for_metrics((loss, avg_iou))
            print(loss)

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
                log_img=(batch_idx % 50 == 0),
                lr=optimizer.param_groups[0]["lr"],
            )

            running_loss += loss.mean().item()
            running_iou += avg_iou.mean().item()
            running_map50 += map_50

            loop.set_postfix(
                loss=f"{running_loss / (batch_idx + 1):.4f}",
                iou=f"{running_iou / (batch_idx + 1):.4f}",
                map50=f"{running_map50 / (batch_idx + 1):.4f}",
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
    running_map50 = running_loss = running_iou = 0.0

    loop = tqdm(
        loader,
        total=len(loader),
        desc=f"Validating Epoch {epoch}:",
        bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )
    for batch_idx, (inputs, targets) in enumerate(loop):
        global_step = epoch * len(loader) + batch_idx

        preds = model(inputs)
        loss, avg_iou = criterion(preds, targets)

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
            config=config,
            log_img=(batch_idx % 50 == 0),
            prefix="Valid",
        )

        running_loss += loss.mean().item()
        running_iou += avg_iou.mean().item()
        running_map50 += map_50

        loop.set_postfix(
            loss=f"{running_loss / (batch_idx + 1):.4f}",
            iou=f"{running_iou / (batch_idx + 1):.4f}",
            map50=f"{running_map50 / (batch_idx + 1):.4f}",
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
