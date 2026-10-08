# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys
import numpy as np
import gc
import torch

from torch.utils.data import DataLoader

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from training.gist import (
    apply_selection,
    validate_selection,
)
from training.state import restore_training_loss_factors

GROUNDING_DINO_WEIGHTS_URL = (
    'https://github.com/IDEA-Research/GroundingDINO/releases/download/'
    'v0.1.0-alpha/groundingdino_swint_ogc.pth'
)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    parser.add_argument('--tensorboard-dir', dest='tb_path', type=str)
    parser.add_argument('--param_train_epoch', default=4, type=int)
    parser.add_argument('--ovd_frozen_epoch', default=20, type=int)
    parser.add_argument('--loss_amp_factor', default=1000, type=float)
    parser.add_argument('--loss_energy_factor', default=10, type=float)

    # DiSG data and pretrained model sources.
    parser.add_argument('--data-root', dest='coco_path', type=str,
                        default=os.environ.get('DISG_ROOT', 'data/DiSG'))
    parser.add_argument('--val-annotations', dest='coco_val_path', type=str)
    parser.add_argument('--init-checkpoint', dest='init_ckpt', type=str,
                        default=GROUNDING_DINO_WEIGHTS_URL)
    parser.add_argument('--bert-path', dest='text_encoder_type', type=str,
                        default='bert-base-uncased')
    parser.add_argument('--clip-model-path', dest='model_path', type=str,
                        default='')
    parser.set_defaults(dataset_file='coco', train_mode='ov',
                        dataset_train_set='ov_train', dataset_val_set='ov_val',
                        fix_size=False)

    # training parameters
    parser.add_argument('--output-dir', dest='output_dir', default='outputs/disg_ovgop',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    seen_id2name = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'train', 5: 'truck', 6: 'boat', 7: 'bench',
                    8: 'bird', 9: 'horse', 10: 'sheep', 11: 'bear', 12: 'zebra', 13: 'giraffe', 14: 'backpack',
                    15: 'handbag', 16: 'suitcase', 17: 'frisbee', 18: 'skis', 19: 'sports ball', 20: 'kite',
                    21: 'baseball bat', 22: 'baseball glove', 23: 'surfboard', 24: 'tennis racket', 25: 'bottle',
                    26: 'wine glass', 27: 'fork', 28: 'spoon', 29: 'bowl', 30: 'banana', 31: 'apple', 32: 'sandwich',
                    33: 'orange', 34: 'broccoli', 35: 'carrot', 36: 'hot dog', 37: 'pizza', 38: 'donut', 39: 'chair',
                    40: 'bed', 41: 'dining table', 42: 'toilet', 43: 'tv', 44: 'laptop', 45: 'mouse', 46: 'remote',
                    47: 'cell phone', 48: 'microwave', 49: 'oven', 50: 'toaster', 51: 'refrigerator', 52: 'book',
                    53: 'clock', 54: 'vase', 55: 'teddy bear', 56: 'toothbrush', 57: 'head', 58: 'hand', 59: 'leg'}
    classnames_object = [v for k, v in seen_id2name.items()]
    classnames_direction = ["RIGHT", "LEFT", "UP", "DOWN", "UPRIGHT", "UPLEFT", "DOWNLEFT", "DOWNRIGHT"]
    classnames = {'object': classnames_object, 'direction': classnames_direction}
    model, criterion, postprocessors = build_func(args, classnames)
    return model, criterion, postprocessors


def reset_seed(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_initial_checkpoint(source):
    if source.startswith(('https://', 'http://')):
        return torch.hub.load_state_dict_from_url(source, map_location='cpu')
    return torch.load(source, map_location='cpu')


def build_training_state(args, device, logger, selection=None, checkpoint=None):
    """Build a fresh model and optimizer for either training stage."""
    reset_seed(args)
    model, criterion, postprocessors = build_model_main(args)
    model.to(device)
    ema_m = ModelEma(model, args.ema_decay) if args.use_ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    if selection is not None:
        validate_selection(selection, args.gist_selection_ratio)
        apply_selection(model_without_ddp, selection)
    for name, parameter in model_without_ddp.named_parameters():
        if name.split('.')[0] == 'bert':
            parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        get_param_dict(args, model_without_ddp), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    if checkpoint is not None:
        model_without_ddp.load_state_dict(checkpoint['model'])
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                ema_m = ModelEma(model, args.ema_decay)
        if not args.eval and all(key in checkpoint for key in ('optimizer', 'lr_scheduler', 'epoch')):
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            restore_training_loss_factors(args, checkpoint)
    elif not args.eval:
        if args.init_ckpt:
            initial = load_initial_checkpoint(args.init_ckpt)['model']
            model_state = model_without_ddp.state_dict()
            matching = {}
            for key, value in initial.items():
                clean_key = key.replace('module.', '')
                if clean_key in model_state and model_state[clean_key].shape == value.shape:
                    matching[clean_key] = value
                else:
                    print(f"❌ Skipping key {key} in pretrained state dict: not found or shape mismatch in DINO model.")
            if not matching:
                raise RuntimeError(f"No model weights match the initialization checkpoint: {args.init_ckpt}")
            model_without_ddp.load_state_dict(matching, strict=False)
            logger.info("Loaded %s matching tensors from %s", len(matching), args.init_ckpt)
            del initial, model_state, matching

        if args.pretrain_model_path:
            from collections import OrderedDict
            initial = torch.load(args.pretrain_model_path, map_location='cpu')['model']
            ignored = args.finetune_ignore or []
            filtered = OrderedDict(
                (key, value) for key, value in utils.clean_state_dict(initial).items()
                if not any(word in key for word in ignored))
            logger.info("Ignore keys: %s", json.dumps(ignored))
            logger.info(str(model_without_ddp.load_state_dict(filtered, strict=False)))

    n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info('number of params: %s', n_parameters)
    return model, model_without_ddp, criterion, postprocessors, optimizer, lr_scheduler, ema_m, n_parameters


def run_selection_stage(args, device):
    """Collect GIST statistics without creating experiment output files."""
    logger = setup_logger(distributed_rank=args.rank, color=False, name='detr_stage1')
    logger.info('Stage I: collecting GIST optimizer statistics (no output files).')
    model, model_without_ddp, criterion, postprocessors, optimizer, lr_scheduler, ema_m, _ = (
        build_training_state(args, device, logger))
    dataset_train = build_dataset(image_set=args.dataset_train_set, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler_train, args.param_select_batch_size, drop_last=True)
    data_loader = DataLoader(dataset_train, batch_sampler=batch_sampler,
                             collate_fn=utils.collate_fn, num_workers=args.num_workers)
    selection = None
    for epoch in range(args.param_train_epoch + 1):
        if args.distributed and hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)
        result = train_one_epoch(
            model, criterion, postprocessors, data_loader, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=False, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), ema_m=ema_m,
            collect_gist=(epoch == args.param_train_epoch))
        if epoch == args.param_train_epoch:
            _, selection, _ = result
    validate_selection(selection, args.gist_selection_ratio)
    if args.distributed:
        chosen = [selection if utils.is_main_process() else None]
        torch.distributed.broadcast_object_list(chosen, src=0)
        selection = chosen[0]
    logger.info('Stage I complete; rebuilding Stage II from the initialization checkpoint.')
    del model, model_without_ddp, criterion, postprocessors, optimizer, lr_scheduler
    del ema_m, dataset_train, sampler_train, batch_sampler, data_loader, result
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return selection

def main(args):
    utils.init_distributed_mode(args)
    args.coco_path = str(Path(args.coco_path).expanduser())
    if args.coco_val_path is None:
        args.coco_val_path = str(Path(args.coco_path) / 'annotations' / 'ovgop_val.json')
    if args.tb_path is None:
        args.tb_path = str(Path(args.output_dir) / 'tensorboard')

    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    raw_args = vars(args).copy()
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    output_dir = Path(args.output_dir)
    if not args.resume and (output_dir / 'checkpoint.pth').is_file():
        args.resume = str(output_dir / 'checkpoint.pth')
    if args.eval and not args.resume:
        raise ValueError("Evaluation requires --resume or checkpoint.pth in --output-dir")

    checkpoint = None
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

    device = torch.device(args.device)
    if args.eval:
        selection = None
    elif checkpoint is not None:
        selection = checkpoint.get('gist_selection')
        if selection is None:
            raise RuntimeError(
                "Training checkpoint lacks GIST selection metadata; use a checkpoint "
                "created by this version. Legacy checkpoints remain usable for evaluation.")
        validate_selection(selection, args.gist_selection_ratio)
    else:
        selection = run_selection_stage(args, device)

    # Stage II (or evaluation) is the first point that creates experiment files.
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        cfg.dump(str(output_dir / 'config_cfg.py'))
        with (output_dir / 'config_args_raw.json').open('w') as handle:
            json.dump(raw_args, handle, indent=2)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    from tensorboard_manager import TensorboardLogger
    TensorboardLogger.initialize(args.tb_path)
    model, model_without_ddp, criterion, postprocessors, optimizer, lr_scheduler, ema_m, n_parameters = (
        build_training_state(args, device, logger, selection=selection, checkpoint=checkpoint))
    del checkpoint
    gc.collect()
    wo_class_error = False

    dataset_train = None
    if not args.eval:
        dataset_train = build_dataset(image_set=args.dataset_train_set, args=args)
    dataset_val = build_dataset(image_set=args.dataset_val_set, args=args)
    sampler_train = None
    if dataset_train is not None:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    base_ds = get_coco_api_from_dataset(dataset_val)
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")

    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)


    Freeze_batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    Freeze_data_loader_train = DataLoader(dataset_train, batch_sampler=Freeze_batch_sampler_train,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers)



    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed and hasattr(sampler_train, 'set_epoch'):
            sampler_train.set_epoch(epoch)

        if args.ovd_frozen_epoch == epoch:
            print("-----------------------------------------")
            print(f"Reducing object-detection learning rates at epoch {args.ovd_frozen_epoch}.")
            print("-----------------------------------------")
            optimizer.param_groups[0]['lr'] = args.lr_frozen_od
            optimizer.param_groups[2]['lr'] = args.lr_frozen_od_backbone
            optimizer.param_groups[3]['lr'] = args.lr_frozen_od_linear_proj_mult
            lr_scheduler.base_lrs[0] = args.lr_frozen_od*1.2
            lr_scheduler.base_lrs[2] = args.lr_frozen_od_backbone*1.2
            lr_scheduler.base_lrs[3] = args.lr_frozen_od_linear_proj_mult*1.2

            args.loss_amp_factor = 6000
            args.loss_energy_factor = 60

        train_stats = train_one_epoch(
            model, criterion, postprocessors, Freeze_data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None), ema_m=ema_m)
        # Only Stage II reaches this loop; its checkpoint and evaluation cadence is unchanged.
        tb_writer = TensorboardLogger.get_writer()
        tb_writer.add_scalar("LearningRate/lr_param_group_0", optimizer.param_groups[0]['lr'], global_step=epoch)
        tb_writer.add_scalar("LearningRate/lr_param_group_1", optimizer.param_groups[1]['lr'], global_step=epoch)
        tb_writer.add_scalar("LearningRate/lr_param_group_2", optimizer.param_groups[2]['lr'], global_step=epoch)
        tb_writer.add_scalar("LearningRate/lr_param_group_3", optimizer.param_groups[3]['lr'], global_step=epoch)
        tb_writer.add_scalar("LearningRate/lr_param_group_4", optimizer.param_groups[4]['lr'], global_step=epoch)

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'gist_selection': selection,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)

        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None), epoch_=epoch
        )
        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'gist_selection': selection,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        if args.use_ema:
            ema_test_stats, ema_coco_evaluator = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'gist_selection': selection,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
            'epoch': epoch,
            'n_parameters': n_parameters,
        }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
