from pprint import pprint
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from safetensors.torch import load_model

from yolo_model import YOLOv1ResNet
from yolo_loss import YOLOLoss
from yolo_utils import yolo_pred_to_xyxy, yolo_target_to_xyxy, filter_boxes
from voc_data import get_dataloaders

from typing import Tuple, Dict, Any
from config_parser import YOLOConfig


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-cp", type=Path, required=True)
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metric: MeanAveragePrecision,
    config: YOLOConfig,
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    running_loss = running_iou = 0.0
    loop = tqdm(loader, desc="Evaluating")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        preds = model(inputs)
        loss, avg_iou = criterion(preds, targets)

        running_loss += loss.item()
        running_iou += avg_iou.item()

        pred_bboxes_list = yolo_pred_to_xyxy(preds, config, threshold=0.6)
        target_bboxes_list = yolo_target_to_xyxy(targets, config, threshold=0.6)

        pred_bboxes_list = filter_boxes(pred_bboxes_list, threshold=0.15)

        for pred_bboxes, target_bboxes in zip(pred_bboxes_list, target_bboxes_list):
            scores = (
                box_iou(pred_bboxes["boxes"], target_bboxes["boxes"]).max(dim=1).values
            )
            pred_bboxes["scores"] = scores

        metrics = metric(pred_bboxes_list, target_bboxes_list)

        loop.set_postfix(
            loss=f"{running_loss / len(loader):.4f}",
            avg_iou=f"{running_iou / len(loader):.4f}",
            map=f"{metrics['map'].item():.4f}",
            map50=f"{metrics['map_50'].item():.4f}",
        )

    final_metrics = metric.compute()
    avg_loss = running_loss / len(loader)

    return avg_loss, final_metrics


def main():
    args = parse_args()
    config = torch.load(args.checkpoint / "config.pt")
    _, valid_loader = get_dataloaders(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLOv1ResNet(config)
    criterion = YOLOLoss(config)
    metric = MeanAveragePrecision(dist_sync_on_step=True)

    missing, unexpected = load_model(model, args.checkpoint / "model.safetensors")
    assert (
        not missing and not unexpected
    ), f"Missing: {missing}, Unexpected: {unexpected}"

    model = model.to(device)
    criterion = criterion.to(device)
    metric = metric.to(device)

    avg_loss, final_metrics = evaluate_model(
        model=model,
        loader=valid_loader,
        criterion=criterion,
        metric=metric,
        config=config,
        device=device,
    )

    pprint(final_metrics)
    print(f"Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
