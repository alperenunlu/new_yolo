from torch import nn
from torchvision.models import resnet34, resnet50

from config_parser import YOLOConfig


class YOLOV1(nn.Module):
    def __init__(self, config=YOLOConfig):
        super().__init__()

        if config.backbone == "resnet34":
            self.backbone = nn.Sequential(
                *list(resnet34(weights="DEFAULT").children())[:-2]
            )
        elif config.backbone == "resnet50":
            self.backbone = nn.Sequential(
                *list(resnet50(weights="DEFAULT").children())[:-2]
            )

        self.neck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 14 * 14, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, config.S * config.S * (config.B * 5 + config.C)),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    import torch
    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    model = YOLOV1(config)
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
