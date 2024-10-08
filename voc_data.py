from torchvision.datasets import VOCDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import ConcatDataset
from voc_utils import get_transforms_func, collate_fn

from typing import Tuple
from config_parser import YOLOConfig

download = not __import__("os").path.exists("./data/VOCdevkit")

def get_dataloaders(config: YOLOConfig) -> Tuple[DataLoader, DataLoader]:
    train_datasets = [
        wrap_dataset_for_transforms_v2(
            VOCDetection(
                root="./data",
                year=year,
                image_set=split,
                download=download,
                transforms=get_transforms_func(config, "train"),
            )
        )
        for year in ["2007", "2012"]
        for split in ["train", "val"]
    ]

    valid_datasets = wrap_dataset_for_transforms_v2(
        VOCDetection(
            root="./data",
            year="2007",
            image_set="test",
            download=download,
            transforms=get_transforms_func(config, "valid"),
        )
    )

    train_datasets_concat: Dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_datasets_concat,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    valid_loader = DataLoader(
        valid_datasets,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    train_loader, valid_loader = get_dataloaders(config)
    print(len(train_loader), len(valid_loader))
    print(train_loader.dataset[0][0].shape, train_loader.dataset[0][1]["target"].shape)
