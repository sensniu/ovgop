"""DiSG dataset used by the OVGOP training and evaluation pipeline."""

import json
import random
from pathlib import Path

import torch
import torchvision

import datasets.transforms as T

__all__ = ["build"]


class ConvertDiSGAnnotations:
    def __init__(self, category_id_to_index, training):
        self.category_id_to_index = category_id_to_index
        self.training = training

    def __call__(self, image, object_annotations, gaze_annotation):
        width, height = image.size
        image_id = torch.tensor([gaze_annotation["image_id"]])

        gaze_box = torch.as_tensor(gaze_annotation["gaze_bbox"], dtype=torch.float32)
        gaze_box[2:] += gaze_box[:2]
        gaze_box[0::2].clamp_(min=0, max=width)
        gaze_box[1::2].clamp_(min=0, max=height)

        head_x, head_y, head_width, head_height = gaze_annotation["head_bbox"]
        expanded_x = head_x - head_width * 0.3
        expanded_y = head_y - head_height * 0.3
        head_box = torch.tensor(
            [
                max(expanded_x, 0),
                max(expanded_y, 0),
                min(expanded_x + head_width * 1.6, width),
                min(expanded_y + head_height * 1.6, height),
            ],
            dtype=torch.float32,
        )

        gaze_category = gaze_annotation["category_id"]
        if self.training:
            gaze_category = self.category_id_to_index[gaze_category]

        target_gaze = {
            "head_box": head_box,
            "eye": torch.as_tensor(gaze_annotation["head_point"], dtype=torch.float32),
            "gaze_box": gaze_box,
            "gaze_point": torch.as_tensor(gaze_annotation["gaze_point"], dtype=torch.float32),
            "orig_size": torch.tensor([height, width]),
            "labels": [gaze_category],
            "image_id": image_id,
            "size": torch.tensor([height, width]),
        }

        annotations = [ann for ann in object_annotations if not ann.get("iscrowd", 0)]
        boxes = torch.as_tensor([ann["bbox"] for ann in annotations], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        category_ids = [ann["category_id"] for ann in annotations]
        if self.training:
            category_ids = [self.category_id_to_index[category_id] for category_id in category_ids]
        labels = torch.tensor(category_ids, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        target = {
            "boxes": boxes[keep],
            "labels": labels[keep].tolist(),
            "image_id": image_id,
            "area": torch.tensor([ann["area"] for ann in annotations])[keep],
            "iscrowd": torch.tensor([ann.get("iscrowd", 0) for ann in annotations])[keep],
            "orig_size": torch.tensor([height, width]),
            "size": torch.tensor([height, width]),
        }

        if annotations and "keypoints" in annotations[0]:
            keypoints = torch.as_tensor(
                [ann["keypoints"] for ann in annotations], dtype=torch.float32
            ).reshape(len(annotations), -1, 3)
            target["keypoints"] = keypoints[keep]

        return image, target, target_gaze


class DiSGDetection(torchvision.datasets.CocoDetection):
    """One sample corresponds to one person-level gaze annotation."""

    def __init__(self, image_folder, annotation_file, transforms, training):
        super().__init__(image_folder, annotation_file)
        self._transforms = transforms
        self.training_split = training

        with Path(annotation_file).open() as handle:
            annotation_data = json.load(handle)

        categories = sorted(annotation_data["categories"], key=lambda item: item["id"])
        self.all_categories = {item["id"]: item["name"] for item in categories}
        self.category_list = [item["name"] for item in categories]

        if training:
            selected_categories = [item for item in categories if item["ov_setting"] == "seen"]
        else:
            selected_categories = categories

        self.category_id_to_index = {
            item["id"]: index for index, item in enumerate(selected_categories)
        }
        self.index_to_category_name = {
            index: item["name"] for index, item in enumerate(selected_categories)
        }
        self.label_index = set(self.index_to_category_name)

        gaze_annotations = annotation_data["annotations_gaze"]
        if training:
            selected_ids = set(self.category_id_to_index)
            gaze_annotations = [
                annotation
                for annotation in gaze_annotations
                if annotation["category_id"] in selected_ids
            ]
        self.gaze_data = gaze_annotations
        self.ids = list(range(len(self.gaze_data)))
        self.prepare = ConvertDiSGAnnotations(self.category_id_to_index, training)

    def __getitem__(self, index):
        gaze_annotation = self.gaze_data[index]
        image_id = gaze_annotation["image_id"]
        image = self._load_image(image_id)
        object_annotations = self._load_target(image_id)

        if self.training_split:
            selected_ids = set(self.category_id_to_index)
            object_annotations = [
                annotation
                for annotation in object_annotations
                if annotation["category_id"] in selected_ids
            ]

        image, target, target_gaze = self.prepare(
            image, object_annotations, gaze_annotation
        )

        if self.training_split:
            positive_labels = set(target["labels"])
            caption_labels = list(positive_labels)
            negative_labels = list(self.label_index.difference(positive_labels))
            caption_labels.extend(random.sample(negative_labels, len(negative_labels)))
            random.shuffle(caption_labels)

            caption_list = [self.index_to_category_name[label] for label in caption_labels]
            caption_index = {name: index for index, name in enumerate(caption_list)}
            target["labels"] = torch.tensor(
                [
                    caption_index[self.index_to_category_name[label]]
                    for label in target["labels"]
                ],
                dtype=torch.int64,
            )
            target_gaze["labels"] = torch.tensor(
                [
                    caption_index[self.index_to_category_name[label]]
                    for label in target_gaze["labels"]
                ],
                dtype=torch.int64,
            )
        else:
            caption_list = self.category_list
            target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)
            target["categories"] = list(self.coco.dataset["categories"])
            target_gaze["labels"] = torch.tensor(target_gaze["labels"], dtype=torch.int64)

        target["cap_list"] = caption_list
        target["caption"] = " . ".join(caption_list) + " ."

        image, target, target_gaze, face, head_info, gaze_heatmap = self._transforms(
            image, target, target_gaze
        )
        return image, target, target_gaze, face, head_info["head_channel"], gaze_heatmap


def make_disg_transforms(training):
    input_size = 224
    max_size = 352
    output_size = 64
    face_size = 224
    gaze_postprocess = T.gaze_postprocess(input_size, output_size, face_size)

    transforms = []
    if training:
        transforms.extend([T.RandomHorizontalFlip(), T.RandomSizeCrop(384, 600)])
    transforms.extend(
        [
            T.RandomResize([input_size], max_size=max_size),
            gaze_postprocess,
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(transforms)


def build(image_set, args):
    if image_set not in {"ov_train", "ov_val"}:
        raise ValueError(f"DiSG only supports 'ov_train' and 'ov_val', got {image_set!r}")

    root = Path(args.coco_path)
    training = image_set == "ov_train"
    image_folder = root / ("train" if training else "val")
    annotation_file = (
        root / "annotations" / "ovgop_train.json"
        if training
        else Path(args.coco_val_path)
    )
    if not image_folder.is_dir():
        raise FileNotFoundError(f"DiSG image directory not found: {image_folder}")
    if not annotation_file.is_file():
        raise FileNotFoundError(f"DiSG annotation file not found: {annotation_file}")

    return DiSGDetection(
        image_folder,
        annotation_file,
        transforms=make_disg_transforms(training),
        training=training,
    )
