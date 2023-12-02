import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from models.imagebind_model import ModalityType

import pandas as pd
import torch
import numpy as np

import data
import pdb

from PIL import Image
# ShapeTalkDataset
# 
# Output of ShapeTalkDataset is a text-shape pair



top_img_dir = 'images/full_size/'
shapetalk_file = 'language/shapetalk_raw_public_version_0.csv'
top_shape_dir = 'point_clouds/scaled_to_align_rendering/'

import pickle

class ShapeTalkDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', max_utters: int = 5,
                 sample_points_num: int = 1024, train_size: float = 0.8, 
                 random_seed: int = 42, device: str = 'cpu', pair_type: str='text'):
        self.root_dir = root_dir
        self.device = device
        self.max_utters = max_utters
        self.split = split
        
        self.df = pd.read_csv(os.path.join(root_dir, shapetalk_file))
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        # ShapeTalk default point number 16384
        self.npoints = 16384
        self.pair_type = pair_type
        self.permutation = np.arange(self.npoints)

        shape_dir = os.path.join(root_dir, top_shape_dir)
        cls = [os.path.join(shape_dir, p) for p in os.listdir(shape_dir)]
        model = []
        for p in cls:
            model.extend([os.path.join(p, t) for t in os.listdir(p)])
        self.shape_paths = []
        for p in model:
            self.shape_paths.extend([os.path.join(p, t) for t in os.listdir(p)])

        self.sample_points_num = sample_points_num

        self.idx = []
        if pair_type == 'text' or pair_type == 'edit':
            train_idx, test_idx = train_test_split(range(len(self.df)), train_size=train_size, random_state=random_seed)

            if split == 'train':
                self.idx = train_idx
            elif split == 'test':
                self.idx = test_idx
            else:
                raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")
        else:
            train_idx, test_idx = train_test_split(self.shape_paths, train_size=train_size, random_state=random_seed)
            
            # with open("shapetalk_train.txt",'w') as f:
            #     for string in train_idx:
            #         f.write(string + '\n')
            # with open("shapetalk_test.txt",'w') as f:
            #     for string in test_idx:
            #         f.write(string + '\n')
            
            # breakpoint()
            if split == 'train':
                self.idx = train_idx
            elif split == 'test':
                self.idx = test_idx
            else:
                raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")

            self.classes = self._get_class_indices()


    def _get_class_indices(self):
        # 클래스별로 데이터 인덱스를 분류합니다
        
        class_indices = {}
        
        """
        for idx, pcd_path in enumerate(self.idx):
            label = pcd_path.split("/")[7]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        classes = []
        for idx_list in class_indices.values():
                classes.append(idx_list)
        
        if self.split == "train":
            with open("shapetalk_data_classes.pkl",'wb') as f:
                # classes = pickle.load(f)
                pickle.dump(classes, f)
        elif self.split == "test":
            with open("shapetalk_test_data_classes.pkl",'wb') as f:
                # classes = pickle.load(f)
                pickle.dump(classes, f)
        """

        if self.split == "train":
            with open("shapetalk_data_classes.pkl",'rb') as f:
                 classes = pickle.load(f)
        
        elif self.split == "test":
            with open("shapetalk_test_data_classes.pkl", "rb") as f:
                classes = pickle.load(f)

        # breakpoint()
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
    
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        idx = self.idx[index]

        if self.pair_type == 'text':
            dt = self.df.loc[idx]

            shape_path = os.path.join(self.root_dir, top_shape_dir, dt["target_uid"] + ".npz")
            shape = np.load(shape_path)['pointcloud']

            # Single-line preprocessing
            # shape = data.load_and_transform_3D([shape], self.device)

            # Sample & Normalize
            shape = self.random_sample(shape, self.sample_points_num)
            shape = self.pc_norm(shape)

            # Point-BERT는 여기서 float()
            shape = torch.from_numpy(shape).to(self.device)
        
            text = f'This is a {dt["target_object_class"]}. ' + ' '.join(dt[f'utterance_{i}'] for i in range(self.max_utters) if type(dt[f'utterance_{i}']) is str)
            text = data.load_and_transform_text([text], self.device)

            return shape, ModalityType.SHAPE, text, ModalityType.TEXT
        
        elif self.pair_type == 'edit':
            dt = self.df.loc[idx]

            shape_path = os.path.join(self.root_dir, top_shape_dir, dt["source_uid"] + ".npz")
            shape = np.load(shape_path)['pointcloud']

            target = np.load(os.path.join(self.root_dir, top_shape_dir, dt["target_uid"] + ".npz"))['pointcloud']
            img_path = os.path.join(self.root_dir, top_img_dir, dt["target_uid"] + '.png')

            target_img = data.load_and_transform_vision_data([img_path], self.device, to_tensor=True)
            target_img = target_img.squeeze(0)

            # Single-line preprocessing
            # shape = data.load_and_transform_3D([shape], self.device)

            # Sample & Normalize
            shape = self.random_sample(shape, self.sample_points_num)
            shape = self.pc_norm(shape)

            target = self.random_sample(target, self.sample_points_num)
            target = self.pc_norm(target)

            # Point-BERT는 여기서 float()
            shape = torch.from_numpy(shape).to(self.device)
        
            t = f'This is a {dt["target_object_class"]}. ' + ' '.join(dt[f'utterance_{i}'] for i in range(self.max_utters) if type(dt[f'utterance_{i}']) is str)
            text = data.load_and_transform_text([t], self.device)

            return shape, shape_path, text, t, target, target_img, img_path
        else:

            # breakpoint()
            shape = np.load(idx)['pointcloud']
            img_path = os.path.splitext(idx)[0].split(top_shape_dir)
            img_path = os.path.join(img_path[0], top_img_dir, img_path[1] + '.png')
            images = data.load_and_transform_vision_data([img_path], self.device, to_tensor=True)
            image = images.squeeze(0)
            
            return shape, ModalityType.SHAPE, image, ModalityType.VISION


# ShapeTalkDataset(
#             root_dir=os.path.join("/root/volume/datasets", "shapetalk"), split="train",
#             sample_points_num = 1024,
#             pair_type="vision",
#         )