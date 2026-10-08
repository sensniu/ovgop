# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import io
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from util.misc import all_gather
from tensorboard_manager import TensorboardLogger


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True, train_mode='norm', cat_id={}, epoch_=0):
        if tuple(iou_types) != ('bbox',):
            raise ValueError('DiSG evaluation only supports bbox AP')
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats
        if train_mode == 'ov':
            self.unseen_list = cat_id['unseen']
        self.train_mode = train_mode
        self.epoch_ = epoch_

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare_for_coco_detection(predictions)

            with contextlib.redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
            img_ids = coco_eval.params.imgIds
            category_count = len(coco_eval.params.catIds) if self.useCats else 1
            eval_imgs = np.asarray(coco_eval.evalImgs).reshape(
                category_count, len(coco_eval.params.areaRng), len(img_ids))

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        metrics = {}
        for iou_type, coco_eval in self.coco_eval.items():
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.summarize()
            for label, value in zip(("AP", "AP50", "AP75"), coco_eval.stats[:3]):
                print(f"{iou_type} {label}: {value:.3f}")
            if iou_type == "bbox":
                metrics["ap"] = float(coco_eval.stats[0])

            if self.train_mode != "ov":
                continue

            tb_writer = TensorboardLogger.get_writer()
            for threshold, label in ((None, "AP"), (0.50, "AP50"), (0.75, "AP75")):
                split_metrics = self.open_vocabulary_ap(coco_eval, threshold)
                for split, value in split_metrics.items():
                    display_value = value * 100
                    print(f"{iou_type} {label} {split}: {display_value}")
                    tb_writer.add_scalar(
                        f"val_OV/{label}_{split}", display_value, global_step=self.epoch_)
                if iou_type == "bbox" and label == "AP":
                    metrics["ap_base"] = split_metrics["base"]
                    metrics["ap_novel"] = split_metrics["novel"]
        return metrics

    def open_vocabulary_ap(self, coco_eval, threshold):
        precision = coco_eval.eval["precision"]
        if threshold is not None:
            index = np.flatnonzero(np.isclose(coco_eval.params.iouThrs, threshold))
            if not index.size:
                raise ValueError(f"IoU threshold {threshold} is missing")
            precision = precision[index[0]:index[0] + 1]

        base, novel = [], []
        for index, category_id in enumerate(coco_eval.params.catIds):
            values = precision[:, :, index, 0, -1]
            values = values[values > -1]
            if values.size:
                (novel if category_id in self.unseen_list else base).append(
                    float(np.mean(values)))
        return {
            "base": float(np.mean(base)) if base else float("nan"),
            "novel": float(np.mean(novel)) if novel else float("nan"),
        }

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]


            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
