from typing import Tuple, Dict, Any, Union

import torch
from torchvision.transforms import v2

from torchvision.tv_tensors import TVTensor
from torch import Tensor

from yolo_utils import xyxy_to_yolo_target

from config_parser import config


# Since i use torchvision.transforms.v2 we can define better transforms with minimal effort
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        v2.Resize(config.IMAGE_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomAffine(
            degrees=10,
            scale=(0.8, 1.2),
            translate=(0.2, 0.2),
            shear=10,
        ),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(config.IMAGE_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

sample_type = Tuple[torch.Tensor, Dict[str, Union[Tensor, TVTensor, Dict[str, Any]]]]


def train_transforms_func(*args: sample_type) -> sample_type:
    """
    args: Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    """
    transformed = train_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def test_transforms_func(*args: sample_type) -> sample_type:
    transformed = test_transforms(*args)
    transformed[1]["target"] = xyxy_to_yolo_target(
        transformed[1]["boxes"], transformed[1]["labels"]
    )
    return transformed


def collate_fn(batch: sample_type) -> Tuple[Tensor, Tensor]:
    images, annotations = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack([annotation["target"] for annotation in annotations])
    return images, targets
