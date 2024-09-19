from argparse import ArgumentParser

import torch

from torchvision.io import read_image
from torchvision.transforms import v2

from safetensors.torch import load_model

from yolo_model import YOLOv1ResNet
from yolo_utils import yolo_pred_to_xyxy, filter_boxes
from yolo_viz_utils import draw_yolo_from_dict, tensor_to_image

from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", "-cp", type=Path, required=True)
    parser.add_argument("--image", "-i", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = torch.load(args.checkpoint / "config.pt")
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(config.IMAGE_SIZE),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLOv1ResNet(config).to(device)
    missing, unexpected = load_model(model, args.checkpoint / "model.safetensors")
    assert (
        not missing and not unexpected
    ), f"Missing: {missing}, Unexpected: {unexpected}"

    model.eval()

    original_image = read_image(str(args.image))

    image = transforms(original_image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        pred = model(image)

    bboxes = yolo_pred_to_xyxy(pred, config)
    bboxes = filter_boxes(
        bboxes,
        threshold=0.5,
    )[0]

    image_with_bbox = draw_yolo_from_dict(
        image.squeeze(0),
        bboxes,
        config,
    )
    image_with_bbox = tensor_to_image(image_with_bbox)

    # add image path "_bbox" suffix then save
    image_path = args.image.with_name(args.image.stem + "_bbox" + args.image.suffix)

    image_with_bbox.save(image_path)


if __name__ == "__main__":
    main()
