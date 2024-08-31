from voc_utils import train_transforms_func, test_transforms_func, collate_fn
from torchvision.datasets import VOCDetection
from torchvision.datasets import wrap_dataset_for_transforms_v2
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from config_parser import config

batch_size = config.SUBDIVISION
num_workers = config.NUM_WORKERS


train_datasets = [
    wrap_dataset_for_transforms_v2(
        VOCDetection(
            root="./data",
            year=year,
            image_set=split,
            download=False,
            transforms=train_transforms_func,
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
        transforms=test_transforms_func,
    )
)

train_datasets_concat = ConcatDataset(train_datasets)

train_loader = DataLoader(
    train_datasets_concat,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
)

test_loader = DataLoader(
    test_datasets,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
)

if __name__ == "__main__":
    for i, (images, targets) in enumerate(test_loader):
        print(images.shape)
        print(targets.shape)
        break