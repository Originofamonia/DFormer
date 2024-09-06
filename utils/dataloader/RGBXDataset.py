import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import io

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

        temp_index=item_name.find("/")

        item_name=item_name[temp_index+1:].split(self._rgb_format)[0]
        # item_name = item_name.split(" ")
        # item_name[-1]=item_name[-1].replace("\n","")
            
        rgb_path = os.path.join(self._rgb_path, item_name.replace('.jpg','').replace('.png','') + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name.replace('.jpg','').replace('.png','')  + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name.replace('.jpg','').replace('.png','')  + self._gt_format)

        # rgb_path = os.path.join(self._rgb_path, item_name[0].replace("rgb/",""))
        # x_path = os.path.join(self._x_path, item_name[1].replace("depth/",""))
        # gt_path = os.path.join(self._gt_path, item_name[2].replace("label/",""))

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

        # if self._split_name == 'train':
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()  # [3, 480, 640]
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()  # [480, 640]
        x = torch.from_numpy(np.ascontiguousarray(x)).float()  # [3, 480, 640]
        # else:
        #     rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        #     gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
        #     x = torch.from_numpy(np.ascontiguousarray(x)).float()
        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
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
    def __init__(self, setting, split_name, transform=None, file_length=None):
        super(TravRGBDDataset, self).__init__()
        self._split_name = split_name  # train, eval
        self._transform_gt = setting['transform_gt']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self.df = self._get_file_names(split_name)
        # self._file_length = file_length
        self.transform = transform
    
    def _get_file_names(self, split_name):
        if split_name == 'train':
            df = pd.read_csv(self._train_source, index_col=0)
        else:
            df = pd.read_csv(self._eval_source, index_col=0)
        return df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        rgb_path = row['img']
        gt_path = rgb_path.replace('/images/', '/labels/')
        gt_file = os.path.splitext(gt_path)[0] + '.png'
        laser_file = row['laser']

        # rgb = io.read_image(rgb_path)
        # gt = io.read_image(gt_file)
        with open(laser_file, 'rb') as f:
            data = pickle.load(f)
            laser = np.array(data['ranges'][::-1])[540:900]
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_file, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        # print(laser.shape)
        if len(laser.shape) == 1:
            laser = np.expand_dims(laser, axis=1)

        if self.transform is not None:
            rgb, gt, laser = self.transform(rgb, gt, laser)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()  # [3, 480, 640]
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()  # [480, 640]
        laser = torch.from_numpy(np.ascontiguousarray(laser)).float()
        output_dict = dict(rgb=rgb, gt=gt, laser=laser, f=str(rgb_path), n=len(self.df))

        return output_dict
    
    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img
        