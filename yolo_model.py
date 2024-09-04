from torch import nn
from torchvision.models import resnet34, resnet50
from config_parser import YOLOConfig


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DetectionHead(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels, config: YOLOConfig):
        super().__init__()
        self.S = config.S
        B = config.B
        C = config.C

        inner_channels = 1024
        self.depth = 5 * B + C
        stride = 2 if self.S == 7 else 1 if self.S == 14 else 0
        self.model = nn.Sequential(
            Block(in_channels, inner_channels, 3, 1, 1),
            Block(inner_channels, inner_channels, 3, stride, 1),
            Block(inner_channels, inner_channels, 3, 1, 1),
            Block(inner_channels, self.depth, 3, 1, 1),
        )

    def forward(self, x):
        return self.model(x).permute(0, 2, 3, 1).contiguous()


class YOLOv1ResNet(nn.Module):
    def __init__(
        self,
        config: YOLOConfig,
        mode="detection",
    ):
        backbone = config.MODEL
        super().__init__()
        self.mode = mode
        if backbone == "resnet34":
            self.resnet = resnet34(weights="DEFAULT")
        elif backbone == "resnet50":
            self.resnet = resnet50(weights="DEFAULT")

        in_features = self.resnet.fc.in_features
        if mode == "detection":
            self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
            self.detection_head = DetectionHead(in_features, config)
            self.backbone.get_submodule("7").requires_grad_(True)

    def forward(self, x):
        if self.mode == "detection":
            x = self.backbone(x)
            x = self.detection_head(x)
        elif self.mode == "classification":
            x = self.resnet(x)
        return x


if __name__ == "__main__":
    import torch
    from config_parser import load_config

    config = load_config("yolo_config.yaml")
    model = YOLOv1ResNet(config)
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
    model = YOLOv1ResNet(config, mode="classification")
    print(model)
    print(model(torch.empty(1, 3, 448, 448)).shape)
