from typing import List

import yaml
from dataclasses import dataclass


@dataclass
class YOLOConfig:
    S: int
    B: int
    C: int
    IMAGE_SIZE: int
    VOC_DETECTION_CATEGORIES: List[str]
    L_coord: float
    L_noobj: float

    MODEL: str

    BATCH_SIZE: int
    SUBDIVISION: int
    NUM_WORKERS: int
    NUM_EPOCHS: int

    LEARNING_RATE: float
    WEIGHT_DECAY: float


def load_config(path: str) -> YOLOConfig:
    config = yaml.safe_load(open(path, "r"))
    return YOLOConfig(**config)


if __name__ == "__main__":
    config = load_config("yolo_config.yaml")
    print(config)
