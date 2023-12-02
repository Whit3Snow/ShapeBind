import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from models.imagebind_model import ModalityType

import pandas as pd
import torch
import numpy as np

# import data
import pdb

# Create Shape-Text embedding pair or Shape Embedding directory & Text Embedding directory

# Inference : batch_size Number of items to do the inference on at once (default 256)
# npy에 담는 batch : write_batch_size Write batch size (default 10**6)


# top_img_dir = 'images/full_size/'
# shapetalk_file = 'language/shapetalk_raw_public_version_0.csv'
# top_shape_dir = 'point_clouds/scaled_to_align_rendering/'

import pickle
import pdb
class EmbeddingDataset(Dataset):
    def __init__(self, emb_file: str, split: str='train', pair_type:str = 'text',
                 train_size: float = 0.9, sample_points_num: int = 4096,
                 random_seed: int = 42, device: str = 'cpu'):
        self.emb_file = emb_file
        self.device = device
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        # ShapeTalk default point number 16384
        self.npoints = 16384
        self.permutation = np.arange(self.npoints)

        self.data = np.load(emb_file)

        self.sample_points_num = sample_points_num
        self.pair_type = pair_type

        if pair_type == 'text':
            train_idx, test_idx = train_test_split(range(len(self.data.files)//2), train_size=train_size, random_state=random_seed)
            if split == 'train':
                self.paths = train_idx
            elif split == 'test':
                self.paths = test_idx
        elif pair_type == 'shape':
            train_idx, test_idx = train_test_split(self.data.files, train_size=train_size, random_state=random_seed) 
            if split == 'train':
                self.paths = train_idx
            elif split == 'test':
                self.paths = test_idx


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
        return len(self.paths)

    def __getitem__(self, index):
        
        if self.pair_type == 'text':
            i = self.paths[index]
            breakpoint()
            shape_emb = self.data[f'{i}-shape']
            text_emb = self.data[f'{i}-text']

            shape_emb = torch.from_numpy(shape_emb).to(self.device)
            text_emb = torch.from_numpy(text_emb).to(self.device)

            return shape_emb, text_emb

        elif self.pair_type == 'shape':
            shape_path = self.paths[index]
            shape = np.load(shape_path)['pointcloud']

            # Sample & Normalize
            shape = self.random_sample(shape, self.sample_points_num)
            shape = self.pc_norm(shape)

            # Point-BERT는 여기서 float()
            shape = torch.from_numpy(shape).to(self.device)
            latent = torch.from_numpy(self.data[shape_path]).to(self.device)
            
            shape=shape.float()
            latent=latent.float()

            return latent, shape
