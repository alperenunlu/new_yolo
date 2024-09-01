from torchvision.datasets import VOCDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from voc_utils import get_transforms_func, collate_fn

from typing import Tuple
from config_parser import YOLOCONFIG

def get_dataloaders(config: YOLOCONFIG) -> Tuple[DataLoader, DataLoader]:


    train_datasets = [
        wrap_dataset_for_transforms_v2(
            VOCDetection(
                root="./data",
                year=year,
                image_set=split,
                download=False,
                transforms=get_transforms_func(config, "train"),
            )
        )
        for year in ["2007", "2012"]
        for split in ["train", "val"]
    ]

    test_datasets = wrap_dataset_for_transforms_v2(
        VOCDetection(
            root="./data",
            year="2007",
            image_set="test",
            download=False,
            transforms=get_transforms_func(config, "validation")
        )
    )

    train_datasets_concat= ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_datasets_concat,
        batch_size=config.SUBDIVISION,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_datasets,
        batch_size=config.SUBDIVISION,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, test_loader

if __name__ == "__main__":
    from config_parser import load_config
    config = load_config("yolo_config.yaml")
    train_loader, test_loader = get_dataloaders(config)
    print(len(train_loader), len(test_loader))
    print(train_loader.dataset[0][0].shape, train_loader.dataset[0][1]["target"].shape)