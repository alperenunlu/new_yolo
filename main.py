import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR

import argparse
from config_parser import load_config

from voc_data import get_dataloaders
from yolo_model import YOLOv1ResNet
from yolo_loss import YOLOLoss
from yolo_trainer import train_one_epoch, validate_one_epoch

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

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    train_loader, test_loader = get_dataloaders(config)
    model = YOLOv1ResNet(config).to(device)
    criterion = YOLOLoss(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = OneCycleLR(
        optimizer,
        0.001,
        steps_per_epoch=len(train_loader) // (config.BATCH_SIZE // config.SUBDIVISION),
        epochs=config.NUM_EPOCHS,
        pct_start=0.5,
    )
    writer = SummaryWriter()
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_iou, train_map, train_lr = train_one_epoch(
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
        test_loss, test_iou, test_map = validate_one_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            writer=writer,
            epoch=epoch,
            config=config,
        )
    writer.close()
