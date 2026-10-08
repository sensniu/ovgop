"""Run OVGOP inference on one image with a known head box."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.disg import make_disg_transforms
from main import build_model_main
from util.misc import clean_state_dict, nested_tensor_from_tensor_list
from util.slconfig import DictAction, SLConfig


DIRECTION_NAMES = [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "UPRIGHT",
    "UPLEFT",
    "DOWNLEFT",
    "DOWNRIGHT",
]


def apply_config(args):
    config = SLConfig.fromfile(args.config_file)
    if args.options:
        config.merge_from_dict(args.options)
    for key, value in config._cfg_dict.to_dict().items():
        if hasattr(args, key):
            raise ValueError(f"Config key {key!r} conflicts with a command-line argument")
        setattr(args, key, value)


def load_categories(annotation_path):
    with Path(annotation_path).open() as handle:
        annotations = json.load(handle)
    return {
        int(category["id"]): category["name"]
        for category in annotations["categories"]
    }


def prepare_input(image, head_box):
    width, height = image.size
    x, y, box_width, box_height = head_box
    if box_width <= 0 or box_height <= 0:
        raise ValueError("Head-box width and height must be positive")
    if x >= width or y >= height or x + box_width <= 0 or y + box_height <= 0:
        raise ValueError("Head box does not overlap the image")

    expanded_x = x - box_width * 0.3
    expanded_y = y - box_height * 0.3
    expanded_head_box = torch.tensor(
        [
            max(expanded_x, 0),
            max(expanded_y, 0),
            min(expanded_x + box_width * 1.6, width),
            min(expanded_y + box_height * 1.6, height),
        ],
        dtype=torch.float32,
    )
    eye = torch.tensor(
        [x + box_width / 2, y + box_height / 2], dtype=torch.float32
    )
    size = torch.tensor([height, width])
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty(0, dtype=torch.int64),
        "area": torch.empty(0, dtype=torch.float32),
        "iscrowd": torch.empty(0, dtype=torch.int64),
        "image_id": torch.tensor([0]),
        "orig_size": size,
        "size": size,
    }
    target_gaze = {
        "head_box": expanded_head_box,
        "eye": eye,
        "gaze_box": expanded_head_box.clone(),
        "gaze_point": eye.clone(),
        "orig_size": size,
        "size": size,
        "labels": torch.empty(0, dtype=torch.int64),
        "image_id": torch.tensor([0]),
    }
    transform = make_disg_transforms(training=False)
    image, target, _, face, head_info, _ = transform(image, target, target_gaze)
    return image, target, face, head_info["head_channel"]


@torch.no_grad()
def run_inference(args):
    args.coco_path = str(Path(args.coco_path).expanduser())
    if args.coco_val_path is None:
        args.coco_val_path = str(Path(args.coco_path) / "annotations" / "ovgop_val.json")
    apply_config(args)

    device = torch.device(args.device)
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    model.to(device).eval()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image, target, face, head_channel = prepare_input(image, args.head_box)

    categories_by_id = load_categories(args.coco_val_path)
    category_names = [categories_by_id[key] for key in sorted(categories_by_id)]
    caption = " . ".join(category_names) + " ."
    samples = nested_tensor_from_tensor_list([image]).to(device)
    faces = face.unsqueeze(0).to(device)
    head_channels = head_channel.unsqueeze(0).to(device)

    with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
        outputs, gaze_outputs, direction_logits, _, _ = model(
            samples,
            category_names,
            faces,
            head_channels,
            [target],
            captions=[caption],
        )

    original_size = torch.tensor([[original_height, original_width]], device=device)
    detections = postprocessors["bbox"](outputs, original_size)[0]
    keep = detections["scores"] >= args.score_threshold
    scores = detections["scores"][keep][: args.top_k].cpu()
    labels = detections["labels"][keep][: args.top_k].cpu()
    boxes = detections["boxes"][keep][: args.top_k].cpu()

    heatmap = gaze_outputs[0, 0].float().cpu().numpy()
    gaze_y, gaze_x = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    direction_index = int(direction_logits[0].argmax().item())
    result = {
        "image": str(image_path),
        "head_box_xywh": list(map(float, args.head_box)),
        "gaze_point_normalized": [
            float(gaze_x / heatmap.shape[1]),
            float(gaze_y / heatmap.shape[0]),
        ],
        "gaze_point_pixels": [
            float(gaze_x / heatmap.shape[1] * original_width),
            float(gaze_y / heatmap.shape[0] * original_height),
        ],
        "head_direction": DIRECTION_NAMES[direction_index],
        "detections": [
            {
                "category_id": int(label),
                "category": categories_by_id.get(int(label), "unknown"),
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for score, label, box in zip(scores, labels, boxes)
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    heatmap_path = output_path.with_suffix(".heatmap.npy")
    np.save(str(heatmap_path), heatmap)
    print(f"Predictions: {output_path}")
    print(f"Gaze heatmap: {heatmap_path}")


def get_inference_parser():
    parser = argparse.ArgumentParser("OVGOP single-image inference")
    parser.add_argument("--config-file", "-c", default="configs/disg_ovgop.py")
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument(
        "--data-root",
        dest="coco_path",
        default=os.environ.get("DISG_ROOT", "data/DiSG"),
    )
    parser.add_argument("--val-annotations", dest="coco_val_path")
    parser.add_argument(
        "--bert-path",
        dest="text_encoder_type",
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--clip-model-path",
        dest="model_path",
        default="",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.set_defaults(dataset_file="coco", train_mode="ov")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--head-box",
        nargs=4,
        type=float,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        required=True,
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="prediction.json")
    return parser


if __name__ == "__main__":
    run_inference(get_inference_parser().parse_args())
