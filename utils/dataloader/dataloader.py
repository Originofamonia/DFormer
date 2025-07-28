import cv2
import torch
import numpy as np
from torch.utils import data
import random
import pandas as pd

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from utils.dataloader.RGBXDataset import TravRGBDLabeledDataset
from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
    normalize_depth,
)


def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale


class TrainPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if self.config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(
                rgb, gt, modal_x, self.config.train_scale_array
            )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize(
                modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28]
            )  # [0.5,0.5,0.5]
        else:
            modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        # return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)
        # p_rgb = p_rgb
        # p_modal_x = p_modal_x

        return p_rgb, p_gt, p_modal_x


class TravTransform(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None, is_train=True):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign
        self.is_train = is_train

    def __call__(self, rgb, gt, modal_x):
        if self.is_train:
            rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        if self.sign:
            modal_x = normalize_depth(
                modal_x, 3.7124, 1.4213
            )  # [0.5,0.5,0.5]
        else:
            modal_x = normalize_depth(modal_x, 3.7124, 1.4213)

        return rgb.transpose(2, 0, 1), gt, modal_x.transpose(1, 0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class ValPre(object):
    def __init__(self, norm_mean, norm_std, sign=False, config=None):
        self.config = config
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sign = sign

    def __call__(self, rgb, gt, modal_x):
        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, [0.48, 0.48, 0.48], [0.28, 0.28, 0.28])
        return rgb.transpose(2, 0, 1), gt, modal_x.transpose(2, 0, 1)
        # return rgb, gt, modal_x


def get_train_loader(engine, dataset, config):
    data_setting = {
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
    if config.get('dataset', False) == 'Trav':
        train_transform = TravTransform(config.norm_mean, config.norm_std, 
                                        config.x_is_single_channel, config)
    else:
        train_transform = TrainPre(
            config.norm_mean, config.norm_std, config.x_is_single_channel, config
        )

    train_dataset = dataset(
        data_setting,
        "train",
        train_transform,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return train_loader, train_sampler


def get_val_loader(engine, dataset, config, val_batch_size=1):
    data_setting = {
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
    if config.get('dataset', False) == 'Trav':
        val_transform = TravTransform(config.norm_mean, config.norm_std, 
                                        config.x_is_single_channel, config)
    else:
        val_transform = ValPre(
            config.norm_mean, config.norm_std, config.x_is_single_channel, config
        )

    val_dataset = dataset(data_setting, "val", val_transform)

    val_sampler = None
    is_shuffle = False
    batch_size = val_batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = val_batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=val_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return val_loader, val_sampler


def get_fs_train_loader(engine, dataset, config):
    data_setting = {
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
    
    train_transform = TravTransform(config.norm_mean, config.norm_std, 
                                    config.x_is_single_channel, config)
    # __init__(self, setting, split_name, transform=None, K=5, Q=1)
    train_dataset = dataset(
        data_setting,
        "train",
        train_transform,
        config.shots,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return train_loader, train_sampler


def get_fs_val_loader(engine, dataset, config, val_batch_size=1):
    setting = {
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
    
    val_transform = TravTransform(config.norm_mean, config.norm_std, 
                                    config.x_is_single_channel, config)
    # __init__(self, setting, split_name, transform=None, K=5, Q=1)
    val_dataset = dataset(setting, "val", val_transform, config.shots,)

    val_sampler = None
    is_shuffle = False
    batch_size = val_batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = val_batch_size // engine.world_size
        is_shuffle = False

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=val_sampler,
        # worker_init_fn=seed_worker,
        # generator=g,
    )

    return val_loader, val_sampler


def get_kfold_loaders(engine, dataset_class, config, csv_file):
    # Load full dataframe
    full_df = pd.read_csv(csv_file)
    full_df = full_df[full_df['label'].notna() & (full_df['label'] != '')]
    # Setup KFold
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        train_df = full_df.iloc[train_idx]
        val_df = full_df.iloc[val_idx]

        if config.get('dataset', False) == 'Trav':
            transform = TravTransform(config.norm_mean, config.norm_std, config.x_is_single_channel, config)
        else:
            transform = TrainPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)

        # Build data_setting
        data_setting = {
            "rgb_root": config.rgb_root_folder,
            "rgb_format": config.rgb_format,
            "gt_root": config.gt_root_folder,
            "gt_format": config.gt_format,
            "transform_gt": config.gt_transform,
            "x_root": config.x_root_folder,
            "x_format": config.x_format,
            "x_single_channel": config.x_is_single_channel,
            "class_names": config.class_names,
            "train_source": None,  # not used
            "eval_source": None,   # not used
        }

        # Instantiate dataset class using overridden df
        train_dataset = dataset_class(data_setting, train_df, transform=transform)
        train_dataset.df = train_df

        val_dataset = dataset_class(data_setting, val_df, transform=transform)
        val_dataset.df = val_df

        # Sampler setup
        train_sampler = None
        val_sampler = None
        is_train_shuffle = True
        is_val_shuffle = False
        train_batch_size = config.batch_size
        val_batch_size = config.val_batch_size if hasattr(config, "val_batch_size") else 1

        if engine.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            train_batch_size = config.batch_size // engine.world_size
            val_batch_size = val_batch_size // engine.world_size
            is_train_shuffle = False

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=is_train_shuffle,
            pin_memory=True,
            sampler=train_sampler,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            num_workers=config.num_workers,
            drop_last=False,
            shuffle=is_val_shuffle,
            pin_memory=True,
            sampler=val_sampler,
        )

        folds.append((train_loader, val_loader, train_sampler, val_sampler))

    return folds


def get_unlabeled_loaders(engine, dataset_class, config, labeled_csv, unlabeled_csv):
    """
    Fully supervised train on all labeled RGB-D pairs
    Test on all unlabeled pairs
    """
    labeled_df = pd.read_csv(labeled_csv)
    train_df = labeled_df[labeled_df['label'].notna() & (labeled_df['label'] != '')]
    val_df = pd.read_csv(unlabeled_csv)
    val_df.rename(columns={'depth_path': 'depth', 'img_path': 'image'}, inplace=True)

    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": None,  # not used
        "eval_source": None,   # not used
    }

    train_transform = TravTransform(config.norm_mean, config.norm_std, config.x_is_single_channel, config, True)
    train_dataset = dataset_class(data_setting, train_df, transform=train_transform)
    train_dataset.df = train_df

    val_transform = TravTransform(config.norm_mean, config.norm_std, config.x_is_single_channel, config, False)
    val_dataset = dataset_class(data_setting, val_df, transform=val_transform)
    val_dataset.df = val_df

    # Sampler setup
    train_sampler = None
    val_sampler = None
    is_train_shuffle = True
    is_val_shuffle = False
    train_batch_size = config.batch_size
    val_batch_size = config.val_batch_size if hasattr(config, "val_batch_size") else 1

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_batch_size = config.batch_size // engine.world_size
        val_batch_size = val_batch_size // engine.world_size
        is_train_shuffle = False

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_train_shuffle,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_val_shuffle,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def get_fewshot_loaders(engine, dataset_class, config, s_csv, q_csv):
    """
    Fully supervised train on all labeled RGB-D pairs
    Test on all unlabeled pairs
    """
    s_df = pd.read_csv(s_csv)
    s_df = s_df[s_df['label'].notna() & (s_df['label'] != '')]
    q_df = pd.read_csv(q_csv)
    # q_df.rename(columns={'depth_path': 'depth', 'img_path': 'image'}, inplace=True)

    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": None,  # not used
        "eval_source": None,   # not used
    }

    train_transform = TravTransform(config.norm_mean, config.norm_std, config.x_is_single_channel, config, True)
    train_dataset = dataset_class(
        df_support=s_df,
        df_query=q_df,
        setting=data_setting,
        transform=train_transform,
        n_shots=config.shots,
        n_queries=1,
        max_iters=config.episodes_per_epoch
    )
    train_dataset.df = s_df

    val_transform = TravTransform(config.norm_mean, config.norm_std, config.x_is_single_channel, config, False)
    val_dataset = dataset_class(
        df_support=s_df,       # still use training as support set
        df_query=q_df,           # evaluate on validation set
        setting=data_setting,
        transform=val_transform,
        n_shots=config.shots,
        n_queries=1,
        max_iters=config.eval_iterations
    )

    # Sampler setup
    train_sampler = None
    val_sampler = None
    is_train_shuffle = True
    is_val_shuffle = False
    train_batch_size = config.batch_size
    val_batch_size = config.val_batch_size if hasattr(config, "val_batch_size") else 1

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_batch_size = config.batch_size // engine.world_size
        val_batch_size = val_batch_size // engine.world_size
        is_train_shuffle = False

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=is_train_shuffle,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=is_val_shuffle,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler, val_sampler
