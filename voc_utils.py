import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import TVTensor

from typing import Tuple, Dict, Any, Union, Callable
from torch import Tensor
from config_parser import YOLOCONFIG

from yolo_utils import xyxy_to_yolo_target

sample_type = Tuple[torch.Tensor, Dict[str, Union[Tensor, TVTensor, Dict[str, Any]]]]


class TransformWrapper:
    def __init__(self, transforms, config):
        self.transforms = transforms
        self.config = config

    def __call__(self, *args: sample_type) -> sample_type:
        transformed = self.transforms(*args)
        transformed[1]["target"] = xyxy_to_yolo_target(
            transformed[1]["boxes"], transformed[1]["labels"], self.config
        )
        return transformed


def get_transforms_func(config: YOLOCONFIG, mode: str) -> Callable[..., sample_type]:
    if mode == "train":
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                v2.RandomAffine(
                    degrees=10,
                    scale=(0.8, 1.2),
                    translate=(0.2, 0.2),
                    shear=10,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                v2.Resize(config.IMAGE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif mode == "valid":
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(config.IMAGE_SIZE),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'valid'.")

    return TransformWrapper(transforms, config)


def collate_fn(batch: sample_type) -> Tuple[Tensor, Tensor]:
    images, annotations = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack([annotation["target"] for annotation in annotations])
    return images, targets


if __name__ == "__main__":
    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    print(get_transforms_func(config, "train"))
    print(get_transforms_func(config, "valid"))
