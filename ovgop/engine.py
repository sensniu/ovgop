# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import time
import math
import sys
from pathlib import Path
from typing import Iterable
import torch.nn as nn
import numpy as np
import torch
import csv
import cv2
from sklearn.metrics import roc_auc_score
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from torch.nn import functional as F
from tensorboard_manager import TensorboardLogger
from training.gist import select_sensitive_parameters, validate_selection
from training.losses import gaze_box_energy_loss

EVALUATION_CSV_FIELDS = ("epoch", "ap", "ap_base", "ap_novel", "auc", "dist", "ang")


def append_evaluation_csv(output_dir, epoch, metrics):
    if not utils.is_main_process():
        return
    score_path = Path(output_dir) / "score.csv"
    if score_path.exists() and score_path.stat().st_size > 0:
        with score_path.open(encoding="utf8", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header != list(EVALUATION_CSV_FIELDS):
            raise ValueError(
                f"{score_path} has an incompatible header. Use a new output directory "
                "or remove the existing score.csv before resuming training."
            )
    with score_path.open("a", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_CSV_FIELDS)
        if score_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: metrics[field] for field in EVALUATION_CSV_FIELDS})


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,
                    collect_gist=False):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    _cnt = 0
    for samples, targets, target_gaze, face, head_channel, gaze_heatmap in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        global_iter = epoch * len(data_loader) + _cnt
        samples = samples.to(device)
        categories = data_loader.dataset.category_list

        faces = []
        head_channels = []
        gaze_heatmaps = []
        gaze_boxes = []
        head_direction_clses = []

        for f in face:
            faces.append(f)
        faces = torch.stack(faces, 0)
        faces = faces.to(device)

        for h in head_channel:
            head_channels.append(h)
        head_channels = torch.stack(head_channels, 0)
        head_channels = head_channels.to(device)

        for g in gaze_heatmap:
            gaze_heatmaps.append(g)
        gaze_heatmaps = torch.stack(gaze_heatmaps, 0)
        gaze_heatmaps = gaze_heatmaps.to(device)

        for b in target_gaze:
            gaze_box = b['gaze_box']
            gaze_boxes.append(gaze_box)

        for c in target_gaze:
            cls = c['head_direction_cls']
            head_direction_clses.append(cls)
        head_direction_clses = torch.tensor(head_direction_clses)
        head_direction_clses = head_direction_clses.to(device)

        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs, gaze_outputs, logits, _, _ = model(
                samples, categories, faces, head_channels, targets,
                head_direction_clses, iter=global_iter, captions=captions
            )
        
            loss_dict = criterion(outputs, targets, cap_list, captions)
            weight_dict = criterion.weight_dict

            head_direction_cls_loss = F.cross_entropy(logits, head_direction_clses)
            loss_dict.update({'loss_head_direction_cls': head_direction_cls_loss})
            weight_dict.update({'loss_head_direction_cls': 1.0})

            gaze_heatmap_pred = gaze_outputs.squeeze(1)

            mse_loss = nn.MSELoss(reduction='none')
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmaps)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_loss = torch.mean(l2_loss)
            loss_amp_factor = args.loss_amp_factor
            loss_dict.update({'loss_gaze': gaze_loss})
            weight_dict.update({'loss_gaze': loss_amp_factor})

            box_energy_loss = gaze_box_energy_loss(gaze_boxes, gaze_heatmap_pred)
            loss_energy_factor = args.loss_energy_factor
            loss_dict.update({'loss_energy': box_energy_loss})
            weight_dict.update({'loss_energy': loss_energy_factor})

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        if args.use_ema and epoch >= args.ema_epoch:
            ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if collect_gist:
        scoring_model = model.module if hasattr(model, "module") else model
        tuned_parameter_names, sorted_scores = select_sensitive_parameters(
            scoring_model, optimizer, args.gist_selection_ratio)
        validate_selection(tuned_parameter_names, args.gist_selection_ratio)

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})

    if collect_gist:
        return resstat, tuned_parameter_names, sorted_scores
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None, epoch_=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    from pycocotools.coco import COCO
    coco = COCO(args.coco_val_path)
    category_dict = coco.loadCats(coco.getCatIds())
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)
    if args.train_mode == "ov":
        unseen_id = [item['id'] for item in category_dict if item['ov_setting'] == 'unseen']

    iou_types = ('bbox',)
    useCats = getattr(args, 'useCats', True)
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args.train_mode == "ov":
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats, train_mode=args.train_mode, cat_id={'unseen':unseen_id}, epoch_=epoch_)
    else:
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)

    _cnt = 0
    all_predmap = []
    all_gtmap = []
    total_error = []

    for samples, targets, target_gaze, face, head_channel, _ in metric_logger.log_every(data_loader, 200, header, logger=logger):
        samples = samples.to(device)

        categories = data_loader.dataset.category_list

        faces = []
        head_channels = []
        head_direction_clses = []

        for f in face:
            faces.append(f)
        faces = torch.stack(faces, 0)
        faces = faces.to(device)

        for h in head_channel:
            head_channels.append(h)
        head_channels = torch.stack(head_channels, 0)
        head_channels = head_channels.to(device)

        for c in target_gaze:
            cls = c['head_direction_cls']
            head_direction_clses.append(cls)
        head_direction_clses = torch.tensor(head_direction_clses)
        head_direction_clses = head_direction_clses.to(device)

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs, gaze_outputs, _, _, _ = model(
                samples, categories, faces, head_channels, targets,
                head_direction_clses, captions=input_captions
            )

        gaze_heatmap_pred = gaze_outputs.squeeze(1).float()

        final_output = [g.cpu().numpy() for g in gaze_heatmap_pred]
        target_gaze_point = [g['gaze_point'].cpu().numpy() for g in target_gaze]
        eye_position = [g['eye'].cpu().numpy() for g in target_gaze]

        for f_point, gt_point, eye_point in \
                zip(final_output, target_gaze_point, eye_position):
            out_size = 64  # Size of heatmap
            heatmap = np.copy(f_point)
            f_point = f_point.reshape([out_size, out_size])

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape)
            f_point = np.array([w_index / out_size, h_index / out_size])
            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
            norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

            f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            heatmap = np.squeeze(heatmap)
            heatmap = cv2.resize(heatmap, (5, 5))
            gt_heatmap = np.zeros((5, 5))
            x, y = list(map(int, gt_point * 5))
            gt_heatmap[y, x] = 1.0

            all_predmap.append(heatmap)
            all_gtmap.append(gt_heatmap)
            total_error.append([f_dist, f_angle])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        coco_evaluator.update(res)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    l2, ang = np.mean(np.array(total_error), axis=0)
    all_predmap = np.stack(all_predmap).reshape([-1])
    all_gtmap = np.stack(all_gtmap).reshape([-1])
    auc = roc_auc_score(all_gtmap, all_predmap)
    tb_writer = TensorboardLogger.get_writer()
    tb_writer.add_scalar("val_GE/AUC", auc, global_step=epoch_)
    tb_writer.add_scalar("val_GE/Dist", l2, global_step=epoch_)
    tb_writer.add_scalar("val_GE/Angle", ang, global_step=epoch_)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    detection_metrics = coco_evaluator.summarize()
    print(auc, l2, ang)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats[:3].tolist()
    stats['coco_eval_bbox_base'] = detection_metrics.get('ap_base', float('nan'))
    stats['coco_eval_bbox_novel'] = detection_metrics.get('ap_novel', float('nan'))
    tb_writer.add_scalar("val_OD/AP", stats['coco_eval_bbox'][0], global_step=epoch_)
    tb_writer.add_scalar("val_OD/AP50", stats['coco_eval_bbox'][1], global_step=epoch_)
    tb_writer.add_scalar("val_OD/AP75", stats['coco_eval_bbox'][2], global_step=epoch_)
    append_evaluation_csv(
        args.output_dir,
        epoch_,
        {
            "epoch": epoch_,
            "ap": detection_metrics["ap"],
            "ap_base": detection_metrics.get("ap_base", float("nan")),
            "ap_novel": detection_metrics.get("ap_novel", float("nan")),
            "auc": auc,
            "dist": l2,
            "ang": ang,
        },
    )

    return stats, coco_evaluator
