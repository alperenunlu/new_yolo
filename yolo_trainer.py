from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from config_parser import config
from yolo_utils import yolo_output_to_xyxy, yolo_target_to_xyxy


def train_one_epoch(
    model, optimizer, loader, criterion, scheduler, device, config=config
):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_idx = 0
    metric = MeanAveragePrecision()

    BATCH_SIZE = config.BATCH_SIZE
    SUBDIVISION = config.SUBDIVISION

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss, avg_iou = criterion(outputs, targets)
        loss.backward()

        if (running_idx + 1) % BATCH_SIZE // SUBDIVISION == 0:
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_iou += avg_iou.item()
        running_idx += 1

        # Compute mAP
        pred_boxes = [
            {"boxes": box, "labels": label, "scores": confidence}
            for box, label, confidence in zip(*yolo_output_to_xyxy(outputs))
        ]
        target_boxes = [
            {"boxes": box, "labels": label}
            for box, label, _ in zip(*yolo_target_to_xyxy(targets))
        ]

        metric_values = metric(pred_boxes, target_boxes)

        postfix = {
            "loss": f"{running_loss / running_idx:.4f}",
            "iou": f"{running_iou / running_idx:.4f}",
            "map": f"{metric_values['map'].item():.4f}",
            "map_50": f"{metric_values['map_50'].item():.4f}",
        }
        loop.set_postfix(postfix)

    if (running_idx + 1) % BATCH_SIZE // SUBDIVISION != 0:
        optimizer.step()
        optimizer.zero_grad()

    metric_compute = metric.compute()
    map = metric_compute["map"]
    map50 = metric_compute["map_50"]

    return (
        map,
        map50,
        metric_compute,
        running_loss / running_idx,
    )


def validate_one_epoch(model, loader, criterion, device, config=config):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_idx = 0
    metric = MeanAveragePrecision()

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss, avg_iou = criterion(outputs, targets)

        running_loss += loss.item()
        running_iou += avg_iou.item()
        running_idx += 1

        # Compute mAP
        pred_boxes = [
            {"boxes": box, "labels": label, "scores": confidence}
            for box, label, confidence in zip(*yolo_output_to_xyxy(outputs))
        ]
        target_boxes = [
            {"boxes": box, "labels": label}
            for box, label, _ in zip(*yolo_target_to_xyxy(targets))
        ]

        metric_values = metric(pred_boxes, target_boxes)

        postfix = {
            "loss": f"{running_loss / running_idx:.4f}",
            "iou": f"{running_iou / running_idx:.4f}",
            "map": f"{metric_values['map'].item():.4f}",
            "map_50": f"{metric_values['map_50'].item():.4f}",
        }
        loop.set_postfix(postfix)

    metric_compute = metric.compute()
    map = metric_compute["map"]
    map50 = metric_compute["map_50"]

    return (
        map,
        map50,
        metric_compute,
        running_loss / running_idx,
    )


if __name__ == "__main__":
    from yolo_model import YOLOv1ResNet
    from yolo_loss import YOLOLoss
    from voc_data import train_loader

    import torch
    import torch.optim as optim
    from torch.optim.lr_scheduler import OneCycleLR

    device = torch.device("mps")
    model = YOLOv1ResNet().to(device)
    criterion = YOLOLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = OneCycleLR(optimizer, 0.01, steps_per_epoch=len(train_loader), epochs=config.NUM_EPOCHS)

    for epoch in range(config.NUM_EPOCHS):
        map, map50, metric, loss = train_one_epoch(
            model, optimizer, train_loader, criterion, scheduler, device
        )
        print(
            f"Epoch {epoch + 1}: Loss: {loss:.4f}, mAP: {map:.4f}, mAP50: {map50:.4f}"
        )
        print(metric)
        print()
