from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision


def train_one_epoch(model, optimizer, loader, criterion, scheduler, device):
    model.train()
    running_loss = 0.0
    running_idx = 0
    metric = MeanAveragePrecision()

    loop = tqdm(loader)
    for data in loop:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if (running_idx + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_idx += 1

        # pred_boxes = []
        # target_boxes = []
        # for out, tar in zip(outputs, targets):
            # coords, labels, confidences = yolo_output_to_xyxy(out, 0.5)
        #     target_coords, target_labels, _ = yolo_target_to_xyxy(tar)
        #     pred_boxes.append({"boxes": coords, "labels": labels, "scores": confidences})
        #     target_boxes.append({"boxes": target_coords, "labels": target_labels})
        #
        # metric_values = metric(pred_boxes, target_boxes)
        #
        postfix = {
            "loss": f"{running_loss / running_idx:.4f}",
            # "map": metric_values["map"],
            # "map50": metric_values["map_50"],
        }
        loop.set_postfix(postfix)

    if (running_idx + 1) % 8 != 0:
        optimizer.step()
        optimizer.zero_grad()

    metric_compute = metric.compute()
    map = metric_compute["map"]
    map50 = metric_compute["map_50"]
    metric.reset()
    return map, map50, running_loss / running_idx
