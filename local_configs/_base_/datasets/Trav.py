from .. import *

# Dataset config
"""Dataset Path"""
C.root_dir = f'/home/qiyuan/2023spring/segmentation_indoor_images'
C.scenes = ['erb', 'uc', 'wh']
C.dataset_path = osp.join(C.root_dir, "NYUDepthv2")
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".jpg"
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, False for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, "Depth")
C.x_format = ".png"
C.x_is_single_channel = (
    True  # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
)
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = True
C.num_train_imgs = 795
C.num_eval_imgs = 654
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
    print(curr_path)


if __name__ == '__main__':
    rename_path()
