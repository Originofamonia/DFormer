# from .. import *
import sys
import os
import os.path as osp
import pickle
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from local_configs._base_ import *

# Dataset config
"""Dataset Path"""
C.dataset = f'Trav'
C.root_dir = '/home/edward/data/trav'
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
C.x_is_single_channel = True
# True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input

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


# Function to read pkl file and convert to ndarray
def read_pkl_to_array(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        laser = np.array(data['ranges'][::-1])[540:900]
    # Convert the loaded data to a NumPy ndarray if it isn't one already
    return laser


def calc_depth_mean_std():
    df = pd.read_csv(f'datasets/trav/merged_rgbd.csv', index_col=0)
    # Apply the function to the 'file_paths' column to create a new column with the ndarrays
    df['depths'] = df['laser'].apply(read_pkl_to_array)
    all_elements = np.concatenate(df['depths'].values)

    # Calculate mean and standard deviation of the concatenated array
    overall_mean = np.mean(all_elements)
    overall_std = np.std(all_elements)
    print(f'mean: {overall_mean}, std: {overall_std}')  # 3.712411900604355, 1.4213359933145486


def update_path(path, new_base, level):
    parts = path.split(os.path.sep)
    last_parts = os.path.join(*parts[-level:])
    return os.path.join(new_base, last_parts)


def rename_path_csv():
    """
    rename df1 and df2's filepaths
    old:
    img: /home/qiyuan/2023spring/segmentation_indoor_images/uc/positive/images/1661556423196969025.jpg
    depth: /data/zak/robot/extracted/uc/8_26/front_laser/1661556423178211711.pkl
    new:
    img: /home/edward/data/segmentation_indoor_images
    depth: /home/edward/data/extracted
    """
    img_base = "/home/edward/data/segmentation_indoor_images"
    depth_base = '/home/edward/data/extracted'
    dfs = ['df1.csv', 'df2.csv']
    for df_file in dfs:
        df_path = os.path.join(C.dataset_path, df_file)
        df = pd.read_csv(df_path, index_col=0)
        df['laser'] = df['laser'].apply(lambda x: update_path(x, depth_base, 4))
        df['img'] = df['img'].apply(lambda x: update_path(x, img_base, 4))
        df = df.rename(columns={'laser': 'depth'})
        df.to_csv(df_path)


if __name__ == '__main__':
    # rename_path()
    # calc_depth_mean_std()
    rename_path_csv()
