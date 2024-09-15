```python
from glob import glob
from pprint import pprint

import torch
from tqdm import tqdm
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from safetensors import safe_open

from yolo_model import YOLOv1ResNet
from yolo_loss import YOLOLoss
from yolo_utils import yolo_pred_to_xyxy, yolo_target_to_xyxy, filter_boxes
from voc_data import get_dataloaders
from config_parser import load_config


def load_checkpoint(model, checkpoint_path):
    with safe_open(checkpoint_path + "/model.safetensors", framework="pt") as f:
        model.load_state_dict({k: f.get_tensor(k) for k in f.keys()})
    return model


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, metric, config, device):
    model.eval()
    running_loss = 0
    idx = 0
    loop = tqdm(dataloader, desc="Evaluating", leave=False)

    for inputs, targets in loop:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss, avg_iou = criterion(outputs, targets)

        running_loss += loss.item()
        idx += 1

        pred_bboxes_list = yolo_output_to_xyxy(outputs, config)
        target_bboxes_list = yolo_target_to_xyxy(targets, config)

        pred_bboxes_list = filter_boxes(pred_bboxes_list)

        for pred_bboxes, target_bboxes in zip(pred_bboxes_list, target_bboxes_list):
            scores = (
                box_iou(pred_bboxes["boxes"], target_bboxes["boxes"]).max(dim=1).values
            )
            pred_bboxes["scores"] = scores

        metric.update(pred_bboxes_list, target_bboxes_list)

        if idx % 10 == 0:
            metrics = metric.compute()
            loop.set_postfix(
                loss=running_loss / idx,
                avg_iou=avg_iou.item(),
                map=metrics["map"].item(),
                map50=metrics["map_50"].item(),
            )

    final_metrics = metric.compute()
    avg_loss = running_loss / idx
    return avg_loss, final_metrics


def main():
    # Load configuration and initialize model
    config = load_config("yolo_config.yaml")
    _, valid_loader = get_dataloaders(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLOv1ResNet(config).to(device)
    criterion = YOLOLoss(config)
    metric = MeanAveragePrecision(dist_sync_on_step=True).to(device)

    # Load the latest checkpoint
    checkpoints = glob("checkpoints/yolo_v1_resnet_*")
    if not checkpoints:
        print("No checkpoints found")
        return

    last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]
    print(f"Loading checkpoint: {last_checkpoint}")
    model = load_checkpoint(model, last_checkpoint)

    # Evaluate model
    avg_loss, final_metrics = evaluate_model(
        model, valid_loader, criterion, metric, config, device
    )

    # Print final results
    pprint(final_metrics)
    print(f"Final Loss: {avg_loss}")


if __name__ == "__main__":
    main()
```