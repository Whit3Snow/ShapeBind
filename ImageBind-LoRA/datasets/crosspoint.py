import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from models.imagebind_model import ModalityType

import pandas as pd
import torch
import numpy as np

import data

import glob
import random
import math

from PIL import Image
from .plyfile import load_ply
# from . import data_utils as d_utils
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ShapeTalkDataset
# 
# Output of CrosspointDataset is a image-shape pair


def load_shapenet_data():
    BASE_DIR = '/root/volume/CrossPoint/'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        all_filepath += pcs
        
    return all_filepath

import pdb 
def get_render_imgs(pcd_path):

    path_lst = pcd_path.split('/')
    path_lst[5] = 'ShapeNetRendering'
    path_lst[-1] = path_lst[-1][:-4]
    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))
    
    return img_path_list
        
import pickle

class CrossPointDataset(Dataset):
    def __init__(self, split: str = 'train', sample_points_num: int = 1024, transform: Optional[Callable] = None,
                 train_size : float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        
        self.device = device    
        self.data = load_shapenet_data()
        self.transform = transform

        # Random seed 고정
        np.random.seed(random_seed)
        
        # ShapeTalk default point number 16384
        self.npoints = 2048
        self.permutation = np.arange(self.npoints)

        self.sample_points_num = sample_points_num

        train_paths, test_paths = train_test_split(self.data, train_size=train_size, random_state=random_seed)

        self.split = split
        if split == 'train':
            self.data = train_paths
        elif split == 'test':
            self.data = test_paths
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

        self.classes = self._get_class_indices()


    def _get_class_indices(self):
        # 클래스별로 데이터 인덱스를 분류합니다
        
        class_indices = {}

        """
        for idx, pcd_path in enumerate(self.data):
            label = pcd_path.split("/")[6]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        classes = []
        for idx_list in class_indices.values():
                classes.append(idx_list)
        
        if self.split == "train":
            with open("crosspoint_data_classes.pkl",'wb') as f:
                pickle.dump(classes, f)
        
        if self.split == "test":
            with open("crosspoint_test_data_classes.pkl",'wb') as f:
                pickle.dump(classes, f)
        """

        if self.split == "train":
            with open("crosspoint_data_classes.pkl",'rb') as f:
                 classes = pickle.load(f)
        
        elif self.split == "test":
            with open("crosspoint_test_data_classes.pkl", "rb") as f:
                classes = pickle.load(f)

        breakpoint()
        
        return classes


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.choice(get_render_imgs(pcd_path)) # 하나의 object에 있는 multi view 들에 대해 하나만 random으로 가져오기 

        # to_tensor=False일 때 Tensor로 return 아닐 시 PIL error
        images = data.load_and_transform_vision_data([render_img_path], self.device, to_tensor=True)
        images = images.squeeze(0)
 
        # if self.transform is not None:
        #     image = images[0]
        #     images = self.transform(image)

            # render_img_list.append(render_img)
        
        shape = load_ply(self.data[item])

        # Sample & Normalize
        shape = self.random_sample(shape, self.sample_points_num)
        shape = self.pc_norm(shape)

        # Point-BERT는 여기서 float()
        shape = torch.from_numpy(shape).to(self.device)
        
        # return shape, ModalityType.SHAPE, images, ModalityType.VISION, render_img_path.split("/")[6]
        return shape, ModalityType.SHAPE, images, ModalityType.VISION

    def __len__(self):
        return len(self.data)


from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import pdb

class CustomBatchSampler():
    def __init__(self, classes, batch_size):
        self.batch_size = batch_size
        self.classes = classes
        self.n_batches = sum(len(c) for c in self.classes) // self.batch_size


    def __iter__(self):

        batches = []
        breakpoint()

        for i in range(self.n_batches):
            batch = []
            classes_idx = random.sample(range(len(self.classes)), self.batch_size)
            ext_classes = [self.classes[idx] for idx in classes_idx]

            for class_idx_group in ext_classes:
                batch.append(random.choice(class_idx_group))
                if len(batch) == self.batch_size:
                    break

            batches.append(batch)
        
        # 남은 data들 12에 안 들어가는 것들 추가해 말아???? "Q.jhj"
        
        return iter(batches)
