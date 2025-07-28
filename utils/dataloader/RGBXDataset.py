import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
# from torchvision import io

class RGBXDataset(Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        # temp_index=item_name.find("/")
        # item_name=item_name[temp_index+1:].split(self._rgb_format)[0]

        # trav    
        # rgb_path = os.path.join(self._rgb_path, item_name.replace('.jpg','').replace('.png','') + self._rgb_format)
        # x_path = os.path.join(self._x_path, item_name.replace('.jpg','').replace('.png','')  + self._x_format)
        # gt_path = os.path.join(self._gt_path, item_name.replace('.jpg','').replace('.png','')  + self._gt_format)
        
        # NYUv2
        item_name = item_name.split("\t")
        item_name[-1]=item_name[-1].replace("\n","")  # in case trailing \n
        rgb_path = os.path.join(self._rgb_path, item_name[0].replace("RGB/",""))
        x_path = os.path.join(self._x_path, item_name[1].replace("Label/",""))
        gt_path = os.path.join(self._gt_path, item_name[1].replace("Label/",""))

        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x =  self._open_image(x_path, cv2.COLOR_BGR2RGB)
        
        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()  # [3, 480, 640]
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()  # [480, 640]
        x = torch.from_numpy(np.ascontiguousarray(x)).float()  # [3, 480, 640]

        output_dict = dict(rgb=rgb, gt=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        if split_name == 'train':
            source = self._train_source
        else:
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors


class TravRGBDDataset(Dataset):
    def __init__(self, setting, transform=None):
        super(TravRGBDDataset, self).__init__()
        self._transform_gt = setting['transform_gt']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self.transform = transform
        self.df = None  # passed in, not loaded here
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        rgb_path = row['img']
        gt_path = rgb_path.replace('/images/', '/labels/')
        gt_file = os.path.splitext(gt_path)[0] + '.npy'
        laser_file = row['depth']

        with open(laser_file, 'rb') as f:
            data = pickle.load(f)
            laser = np.array(data['ranges'][::-1])[540:900]
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = np.load(gt_file)

        if len(laser.shape) == 1:
            laser = np.expand_dims(laser, axis=1)

        if self.transform is not None:
            rgb, gt, laser = self.transform(rgb, gt, laser)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()  # [3, 480, 640]
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()  # [480, 640]
        laser = torch.from_numpy(np.ascontiguousarray(laser)).float()
        output_dict = dict(rgb=rgb, gt=gt, laser=laser, rgb_path=rgb_path, 
                           gt_path=gt_path, laser_path=laser_file, n=len(self.df))

        return output_dict
    
    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img


class TravRGBDLabeledDataset(TravRGBDDataset):
    def __init__(self, setting, df, transform=None):
        super().__init__(setting, transform=transform)
        self.df = df
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        with open(row['depth'], 'rb') as f:
            data = pickle.load(f)
            laser = np.array(data['ranges'][::-1])[540:900]
        rgb = self._open_image(row['image'], cv2.COLOR_BGR2RGB)
        if 'label' in row and type(row['label']) is str:
            gt = np.load(row['label'])
        else:
            gt = None

        if len(laser.shape) == 1:
            laser = np.expand_dims(laser, axis=1)

        if self.transform is not None:
            rgb, gt, laser = self.transform(rgb, gt, laser)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()  # [3, 480, 640]
        laser = torch.from_numpy(np.ascontiguousarray(laser)).float()
        if gt is not None:
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()  # [480, 640]
            output_dict = dict(rgb=rgb, gt=gt, laser=laser, rgb_path=row['image'], 
                           gt_path=row['label'], depth_path=row['depth'], n=len(self.df))
        else:
            output_dict = dict(rgb=rgb, laser=laser, rgb_path=row['image'], 
                            depth_path=row['depth'], n=len(self.df))
        return output_dict


class FewShotTravRGBDDataset(Dataset):
    def __init__(self, setting, split_name, transform=None, K=5, Q=1):
        super(FewShotTravRGBDDataset, self).__init__()
        self._split_name = split_name
        self._transform_gt = setting['transform_gt']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self.transform = transform
        self.K = K  # Number of support samples
        self.Q = Q  # Number of query samples

        # Load dataset and organize by class
        self.df = self._get_file_names(split_name)
        self.class_to_images = self._group_by_class()

    def _get_file_names(self, split_name):
        """ Load file names from CSV """
        file_source = self._train_source if split_name == 'train' else self._eval_source
        return pd.read_csv(file_source, index_col=0)

    def _group_by_class(self):
        """ Group file paths by class """
        class_to_images = {cls: [] for cls in self.class_names}
        for _, row in self.df.iterrows():
            gt_path = row['img'].replace('/images/', '/labels/')
            gt_file = os.path.splitext(gt_path)[0] + '.npy'
            class_label = self._get_class_from_mask(gt_file)  # Extract class info from mask
            if class_label in class_to_images:
                class_to_images[class_label].append({
                    'rgb': row['img'],
                    'gt': gt_file,
                    'depth': row['depth']
                })

        return class_to_images

    def _get_class_from_mask(self, gt_file):
        """ Extract dominant class from mask (excluding 255) """
        gt = np.load(gt_file)
        
        # Get unique classes, excluding background (255)
        unique_classes = np.unique(gt)
        unique_classes = unique_classes[(unique_classes != 255)]  # Remove 255
        
        # Keep only valid classes (0 or 1)
        valid_classes = unique_classes[np.isin(unique_classes, [0, 1])]

        return int(np.random.choice(valid_classes)) if len(valid_classes) > 0 else -1

    def __len__(self):
        return len(self.class_names)  # Number of episodes = number of classes

    def __getitem__(self, index):
        """ Returns an episode: K support samples + Q query samples for a class """
        class_label = self.class_names[index]
        images_list = self.class_to_images[class_label]

        if len(images_list) < self.K + self.Q:
            raise ValueError(f"Not enough samples for class {class_label}!")

        # Randomly sample K support and Q query examples using numpy
        sampled_indices = np.random.choice(len(images_list), self.K + self.Q, replace=False)
        sampled_images = [images_list[i] for i in sampled_indices]
        support_set = sampled_images[:self.K]
        query_set = sampled_images[self.K:]

        def load_sample(sample):
            """ Load and transform an image, mask, and depth file """
            with open(sample['depth'], 'rb') as f:
                laser = np.array(pickle.load(f)['ranges'][::-1])[540:900]
            laser = np.expand_dims(laser, axis=1) if len(laser.shape) == 1 else laser

            rgb = self._open_image(sample['rgb'], cv2.COLOR_BGR2RGB)
            gt = np.load(sample['gt'])

            if self.transform:
                rgb, gt, laser = self.transform(rgb, gt, laser)

            return (
                torch.tensor(rgb).float(),
                torch.tensor(gt).long(),
                torch.tensor(laser).float()
            )

        # Load support and query sets
        support_images, support_masks, support_lasers = zip(*[load_sample(s) for s in support_set])
        query_images, query_masks, query_lasers = zip(*[load_sample(q) for q in query_set])

        return {
            "s_imgs": torch.stack(support_images),  # [K, C, H, W]
            "s_masks": torch.stack(support_masks),  # [K, H, W]
            "s_depths": torch.stack(support_lasers),  # [K, L]
            "q_imgs": torch.stack(query_images),  # [Q, C, H, W]
            "q_masks": torch.stack(query_masks),  # [Q, H, W]
            "q_depths": torch.stack(query_lasers),  # [Q, L]
            "class": class_label
        }

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        """ Open an image file """
        return np.array(cv2.imread(filepath, mode), dtype=dtype)


class FewShotTravDatasetBinary(Dataset):
    def __init__(self, df_support, df_query, setting, transform=None,
                 n_shots=1, n_queries=1, max_iters=1000):
        super().__init__()
        self.df_support = df_support.reset_index(drop=True)
        self.df_query = df_query.reset_index(drop=True)
        self.setting = setting
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.max_iters = max_iters

        # Reuse your dataset class
        self.support_dataset = TravRGBDLabeledDataset(setting, df=self.df_support, transform=transform)
        self.query_dataset = TravRGBDLabeledDataset(setting, df=self.df_query, transform=transform)

    def __len__(self):
        return self.max_iters

    def __getitem__(self, index):
        # Sample support and query indices
        support_indices = np.random.choice(len(self.df_support), size=self.n_shots, replace=False)
        query_indices = np.random.choice(len(self.df_query), size=self.n_queries, replace=False)

        # Load support data
        support_images = []
        support_labels = []
        support_lasers = []
        for i in support_indices:
            sample = self.support_dataset[i]
            support_images.append(sample['rgb'])   # [3, H, W]
            support_labels.append(sample['gt'])    # [H, W]
            support_lasers.append(sample['laser']) # [360, 1] assumed

        # Load query data
        query_images = []
        query_labels = []
        query_lasers = []
        for i in query_indices:
            sample = self.query_dataset[i]
            query_images.append(sample['rgb'])
            query_labels.append(sample['gt'])
            query_lasers.append(sample['laser'])

        # Stack tensors
        support_images = torch.stack(support_images, dim=0)   # [n_shots, 3, H, W]
        support_labels = torch.stack(support_labels, dim=0)   # [n_shots, H, W]
        support_lasers = torch.stack(support_lasers, dim=0)   # [n_shots, 360, 1]
        query_images = torch.stack(query_images, dim=0)       # [n_queries, 3, H, W]
        query_labels = torch.stack(query_labels, dim=0)       # [n_queries, H, W]
        query_lasers = torch.stack(query_lasers, dim=0)       # [n_queries, 360, 1]

        return {
            's_img': support_images,
            's_gt': support_labels,
            's_depth': support_lasers,
            'q_img': query_images,
            'q_gt': query_labels,
            'q_depth': query_lasers
        }
