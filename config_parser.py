from typing import List

import yaml
from dataclasses import dataclass


@dataclass
class YOLOCONFIG:
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


def load_config(path: str) -> YOLOCONFIG:
    config = yaml.safe_load(open(path, "r"))
    return YOLOCONFIG(**config)


if __name__ == "__main__":
    config = load_config("yolo_config.yaml")
    print(config)
