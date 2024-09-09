from typing import List, Tuple

import yaml
from dataclasses import dataclass


@dataclass
class YOLOConfig:
    S: int
    B: int
    C: int
    IMAGE_SIZE: Tuple[int, int]
    VOC_DETECTION_CATEGORIES: List[str]
    L_coord: float
    L_obj: float
    L_noobj: float
    L_class: float

    MODEL: str

    BATCH_SIZE: int
    NUM_EPOCHS: int
    ACCUMULATE_GRAD_BATCHES: int

    LEARNING_RATE: float
    WEIGHT_DECAY: float


def load_config(path: str) -> YOLOConfig:
    config = yaml.safe_load(open(path, "r"))
    return YOLOConfig(**config)


if __name__ == "__main__":
    config = load_config("yolo_config.yaml")
    print(config)
