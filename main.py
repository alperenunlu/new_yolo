import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import argparse
from config_parser import load_config

from voc_data import get_dataloaders
from yolo_model import YOLOv1ResNet
from yolo_loss import YOLOLoss
from yolo_trainer import train_one_epoch, valid_one_epoch
from yolo_train_utils import save_checkpoint, load_checkpoint

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", "-c", type=str, default="yolo_config.yaml", help="Path to config file"
)
parser.add_argument(
    "--checkpoint", "-cp", type=str, default=None, help="Path to checkpoint file"
)

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    train_loader, valid_loader = get_dataloaders(config)
    model = YOLOv1ResNet(config).to(device)
    criterion = YOLOLoss(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=50*len(train_loader), gamma=0.1)
    start_epoch = 0
    if args.checkpoint:
        model, optimizer, scheduler, start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint
        )
        start_epoch += 1
    writer = SummaryWriter()
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_map, train_map50, train_metric_compute, train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            writer=writer,
            epoch=epoch,
            config=config,
        )
        valid_map, valid_map50, valid_metric_compute, valid_loss = valid_one_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch,
            config=config,
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            map_value=valid_map,
            map50=valid_map50,
            metric_compute=valid_metric_compute,
            loss=valid_loss,
            path=f"checkpoints/yolo_v1_resnet_{epoch}.pth",
        )

    writer.close()
