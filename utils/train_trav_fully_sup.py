"""
1. Make Dataset loader from labeled_rgbd_pairs.csv
2. Fully-supervised train on labeled dataset
3. infer masks on unlabeled data as ground truth labels
"""

import os
import sys
import pprint
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from importlib import import_module
import datetime
from tqdm import tqdm
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader.dataloader import get_4fold_loaders
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import TravRGBDLabeledDataset

from utils.init_func import group_weight
# from utils.init_func import configure_optimizers
from utils.lr_policy import WarmUpPolyLR
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.val_mm import evaluate, evaluate_msf
# from local_configs.NYUDepthv2.DFormer_Base import C
# from eval import evaluate_mid


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=f'local_configs.Trav.DFormer_Base', type=str, help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
parser.add_argument('--device', type=str, default='cuda:0')  # Change to 'cuda:1' for GPU 1
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
# parser.add_argument("--save_path", type=str, default='')
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0, type=int)
parser.add_argument('--save_path', default=f'trained/trav/', type=str)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def is_eval(epoch, config):
    return epoch > int(config.checkpoint_start_epoch) or epoch == 1 or epoch % config.checkpoint_step == 0


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            logger.info("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = (
        True  # train speed is slower after enabling this opts.
    )

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == '__main__':
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        config = getattr(import_module(args.config), "C")
        logger = get_logger(config.log_dir, config.log_file)  # , rank=engine.local_rank
        if args.use_seed:
            set_seed(config.seed)
            logger.info(f"set seed {config.seed}")
        else:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True 
            logger.info("use random seed")

        # assert not (args.compile and args.syncbn), "syncbn is not supported in compile mode"
        if not args.compile and args.compile_mode != "default":
            logger.warning(
                "compile_mode is only valid when compile is enabled, ignoring compile_mode"
            )
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb_dir = config.tb_dir + "/{}".format(
                time.strftime("%b%d_%d-%H-%M", time.localtime())
            )
            generate_tb_dir = config.tb_dir + "/tb"
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)
            pp = pprint.PrettyPrinter(indent=4)
            logger.info(f"config: \n{pp.pformat(config)}")

        logger.info("args parsed:")
        for k in args.__dict__:
            logger.info(f'{k}: {args.__dict__[k]}')

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)

        if args.syncbn:
            BatchNorm2d = nn.SyncBatchNorm
            logger.info("using syncbn")
        else:
            BatchNorm2d = nn.BatchNorm2d
            logger.info("using regular bn")

        model = segmodel(
            cfg=config,
            criterion=criterion,
            norm_layer=BatchNorm2d,
            syncbn=args.syncbn,
        )

        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr

        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

        if config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_list,
                lr=base_lr,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "SGDM":
            optimizer = torch.optim.SGD(
                params_list,
                lr=base_lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise NotImplementedError

        if engine.distributed:
            logger.info(".............distributed training.............")
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(
                    model,
                    device_ids=[engine.local_rank],
                    output_device=engine.local_rank,
                    find_unused_parameters=False,
                )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        val_data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": config.train_source,
            "eval_source": config.eval_source,
            "class_names": config.class_names,
        }
        if args.compile:
            compiled_model = torch.compile(
                model, backend="inductor", mode=args.compile_mode
            )
        else:
            compiled_model = model
        miou, best_miou = 0.0, 0.0
        train_timer = gpu_timer()
        eval_timer = gpu_timer()

        if args.amp:
            scaler = torch.amp.GradScaler()
        model = compiled_model
        # train_loader, train_sampler = get_train_loader(engine, TravRGBDLabeledDataset, config)
        folds = get_4fold_loaders(engine, TravRGBDLabeledDataset, config, 
                                  csv_file="/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv")
        wandb.init(project="MM-FSS", config=config)
        for fold_idx, (train_loader, val_loader, train_sampler, val_sampler) in enumerate(folds):
            logger.info(f"Fold: {fold_idx}")
            logger.info(f"Val dataset len:{len(val_loader)*int(args.gpus)}")
            niters_per_epoch = len(train_loader)
            total_iteration = config.epochs * niters_per_epoch
            lr_policy = WarmUpPolyLR(
                base_lr,
                config.lr_power,
                total_iteration,
                len(train_loader) * config.warm_up_epoch,
            )
            engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
            if engine.continue_state_object:
                engine.restore_checkpoint()

            for epoch in range(engine.state.epoch, config.epochs + 1):
                model.train()
                i = 0
                pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}/{config.epochs}")
                train_timer.start()
                for idx, batch in pbar:
                    if engine.distributed:
                        train_sampler.set_epoch(epoch)
                    sum_loss = 0

                    engine.update_iteration(epoch, idx)

                    rgb = batch["rgb"]
                    gt = batch["gt"]
                    laser = batch["laser"]

                    rgb = rgb.cuda(non_blocking=True)
                    gt = gt.cuda(non_blocking=True)
                    laser = laser.cuda(non_blocking=True)

                    if args.amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss, out = model(rgb, laser, gt)
                    else:
                        loss, out = model(rgb, laser, gt)

                    # reduce the whole loss over multi-gpu
                    if engine.distributed:
                        reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

                    if args.amp:
                        # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                        scaler.scale(loss).backward()
                        # otherwise, optimizer.step() is skipped.
                        scaler.step(optimizer)
                        # Updates the scale for next iteration.
                        scaler.update()
                        optimizer.zero_grad(
                            set_to_none=True
                        )  # TODO: check if set_to_none=True impact the performance
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    if not args.amp:
                        if epoch == 1:
                            for name, param in model.named_parameters():
                                if param.grad is None:
                                    logger.warning(f"{name} has no grad, please check")

                    current_idx = (epoch - 1) * len(train_loader) + idx
                    lr = lr_policy.get_lr(current_idx)

                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]["lr"] = lr

                    if engine.distributed:
                        sum_loss += reduce_loss.item()
                        print_str = (
                            "Epoch {}/{}".format(epoch, config.epochs)
                            + " lr=%.4e" % lr
                            + " loss=%.4f total_loss=%.4f"
                            % (reduce_loss.item(), (sum_loss / (idx + 1)))
                        )

                    else:
                        sum_loss += loss
                        print_str = (
                            f"Epoch {epoch}/{config.epochs} "
                            + f"lr={lr:.4e} loss={loss:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                        )

                    pbar.set_postfix({
                        'lr': f"{lr:.4e}",
                        'loss': f"{loss.item():.4f}",
                        'total_loss': f"{(sum_loss / (idx + 1)):.4f}"
                    })
                    del loss
                # logger.info(print_str)
                train_timer.stop()

                if is_eval(epoch, config):
                    eval_timer.start()
                    torch.cuda.empty_cache()
                    # if args.compile and args.mst and (not args.sliding):
                    #     model = uncompiled_model
                    # TODO: FIX this
                    if engine.distributed:
                        with torch.no_grad():
                            model.eval()
                            device = torch.device("cuda")
                            if args.val_amp:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    if args.mst:
                                        all_metrics = evaluate_msf(
                                            model,
                                            val_loader,
                                            config,
                                            device,
                                            [0.5, 0.75, 1.0, 1.25, 1.5],
                                            True,
                                            engine,
                                            sliding=args.sliding,
                                        )
                                    else:
                                        all_metrics = evaluate(
                                            model,
                                            val_loader,
                                            config,
                                            device,
                                            engine,
                                            sliding=args.sliding,
                                        )
                            else:
                                if args.mst:
                                    all_metrics = evaluate_msf(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5],
                                        True,
                                        engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    all_metrics = evaluate(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        engine,
                                        sliding=args.sliding,
                                    )
                            if engine.local_rank == 0:
                                metric = all_metrics[0]
                                for other_metric in all_metrics[1:]:
                                    metric.update_hist(other_metric.hist)
                                ious, miou = metric.compute_iou()
                                acc, macc = metric.compute_pixel_acc()
                                f1, mf1 = metric.compute_f1()
                                if miou > best_miou:
                                    best_miou = miou
                                    engine.save_and_link_checkpoint(
                                        config.log_dir,
                                        config.log_dir,
                                        config.log_dir_link,
                                        infor="_miou_" + str(miou),
                                        metric=miou,
                                    )
                                logger.info(f"miou: {miou}; best: {best_miou}")
                    elif not engine.distributed:
                        with torch.no_grad():
                            model.eval()
                            device = torch.device("cuda")
                            if args.val_amp:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    if args.mst:
                                        metric = evaluate_msf(
                                            model,
                                            val_loader,
                                            config,
                                            device,
                                            [0.5, 0.75, 1.0, 1.25, 1.5],
                                            True,
                                            engine,
                                            sliding=args.sliding,
                                        )
                                    else:
                                        metric = evaluate(
                                            model,
                                            val_loader,
                                            config,
                                            device,
                                            engine,
                                            sliding=args.sliding,
                                        )
                            else:
                                if args.mst:
                                    metric = evaluate_msf(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5],
                                        True,
                                        engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    metric = evaluate(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        engine,
                                        sliding=args.sliding,
                                    )
                            ious, miou = metric.compute_iou()
                            acc, macc = metric.compute_pixel_acc()
                            f1, mf1 = metric.compute_f1()
                            wandb.log({
                                "Fold": fold_idx,
                                "epoch": epoch,
                                "mIoU": miou,
                                "mean Acc": macc,
                                "mean F1": mf1,
                                "pixel Acc": acc,
                                "F1_cls_0": f1[0],
                                "F1_cls_1": f1[1],
                                "IoU_cls_0": ious[0],
                                "IoU_cls_1": ious[1],
                                "Acc_cls_0": acc[0],
                                "Acc_cls_1": acc[1],
                            })

                        if miou > best_miou:
                            best_miou = miou
                            engine.save_and_link_checkpoint(
                                config.log_dir,
                                config.log_dir,
                                config.log_dir_link,
                                infor="_miou_" + str(miou),
                                metric=miou,
                            )
                        logger.info(f"miou: {miou}; best: {best_miou}")
                    logger.info(
                        f"Epoch {epoch} validation result: mIoU {miou}, best mIoU {best_miou}"
                    )
                    eval_timer.stop()

                eval_count = 0
                for i in range(engine.state.epoch + 1, config.epochs + 1):
                    if is_eval(i, config):
                        eval_count += 1
                left_time = (
                    train_timer.mean_time * (config.epochs - engine.state.epoch)
                    + eval_timer.mean_time * eval_count
                )
                eta = (
                    datetime.datetime.now() + datetime.timedelta(seconds=left_time)
                ).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"Avg train time: {train_timer.mean_time:.2f}s, avg eval time: {eval_timer.mean_time:.2f}s, left eval count: {eval_count}, ETA: {eta}"
                )
