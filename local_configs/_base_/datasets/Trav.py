# from .. import *
import sys
import os
import os.path as osp
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from local_configs._base_ import *

# Dataset config
"""Dataset Path"""
C.root_dir = f'/home/qiyuan/2024fall/DFormer/datasets/trav'
C.scenes = ['erb', 'uc', 'wh']
C.dataset_path = osp.join(C.root_dir)  #  "NYUDepthv2"
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "labels")
C.gt_format = ".png"
C.gt_transform = False
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, False for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = (
    True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
)
C.train_source = osp.join(C.dataset_path, "df1.csv")
C.eval_source = osp.join(C.dataset_path, "df2.csv")
C.is_test = True
C.num_train_imgs = 421
C.num_eval_imgs = 423
C.num_classes = 2
C.class_names = [
    "obstacle",
    "freespace",
]

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.5174, 0.4857, 0.5054])
C.norm_std = np.array([0.2726, 0.2778, 0.2861])


def rename_path():
    """
    rename from path from 10.33.48.70 to 10.33.48.66
    """
    scenes = ['erb', 'uc', 'wh']
    curr_path = os.getcwd()
    df_list = []
    for sc in scenes:
        file = osp.join(curr_path, f'datasets/trav/{sc}_laser_mapping.csv')
        df = pd.read_csv(file, index_col=0)
        df['img'] = df['img'].str.replace('/mnt/hdd', '/home/qiyuan/2023spring')
        df['laser'] = df['laser'].str.replace('/mnt/hdd/zak_extracted', '/data/zak/robot/extracted')
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    df_shuffled = merged_df.sample(frac=1, random_state=444).reset_index(drop=True)
    split_index = len(df_shuffled) // 2
    df1 = df_shuffled.iloc[:split_index]
    df2 = df_shuffled.iloc[split_index:]
    df1.to_csv(f'datasets/trav/df1.csv')
    df2.to_csv(f'datasets/trav/df2.csv')
    # merged_df.to_csv(f'datasets/trav/merged_rgbd.csv')


if __name__ == '__main__':
    rename_path()
