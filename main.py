import argparse
from pathlib import Path

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.detection import MeanAveragePrecision
from accelerate import Accelerator

from config_parser import load_config
from voc_data import get_dataloaders
from yolo_model import YOLOv1ResNet
from yolo_loss import YOLOLoss
from yolo_trainer import train_one_epoch, valid_one_epoch
from yolo_train_utils import save_checkpoint, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", "-c", type=str, default="yolo_config.yaml", help="Path to config file"
)
parser.add_argument(
    "--checkpoint", "-cp", type=Path, default=None, help="Path to checkpoint file"
)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_GRAD_BATCHES,
    )

    train_loader, valid_loader = get_dataloaders(config)
    model = YOLOv1ResNet(config)
    criterion = YOLOLoss(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
    )
    metric = MeanAveragePrecision(dist_sync_on_step=True, backend='faster_coco_eval')

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint) + 1

    model, criterion, optimizer, scheduler, metric, train_loader, valid_loader = (
        accelerator.prepare(
            model, criterion, optimizer, scheduler, metric, train_loader, valid_loader
        )
    )

    curr_map_50 = float("-inf")

    writer = SummaryWriter()
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_map, train_map50, train_metric_compute, train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            criterion=criterion,
            scheduler=scheduler,
            accelerator=accelerator,
            metric=metric,
            writer=writer,
            epoch=epoch,
            config=config,
        )
        valid_map, valid_map50, valid_metric_compute, valid_loss = valid_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            accelerator=accelerator,
            metric=metric,
            writer=writer,
            epoch=epoch,
            config=config,
        )

        if valid_map50 > curr_map_50:
            curr_map_50 = valid_map50

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=accelerator,
                epoch=epoch,
                config=config,
                map_value=valid_map,
                map50=valid_map50,
                metric_compute=valid_metric_compute,
                loss=valid_loss,
                path=Path("checkpoints/yolo_v1_resnet"),
            )

    writer.close()
