import torch
from torchvision.transforms import v2 as T

from yolo_utils import xyxy_to_yolo_target

from config_parser import config

train_transforms = T.Compose(
    [
        T.ToImage(),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.Resize(config.IMAGE_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.RandomAffine(
            degrees=10,
            scale=(0.8, 1.2),
            translate=(0.2, 0.2),
            shear=10,
        ),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = T.Compose(
    [
        T.ToImage(),
        T.Resize(config.IMAGE_SIZE),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train_transforms_func(*args):
    transformed = train_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def test_transforms_func(*args):
    transformed = test_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def collate_fn(batch):
    images, annotations = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack([annotation["target"] for annotation in annotations])
    return images, targets