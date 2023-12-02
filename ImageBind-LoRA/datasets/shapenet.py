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

# ShapeNetDataset
# 
# Output of ShapeNetDataset is a shape, class pair



shapetalk_file = 'ShapeNet-55/'
top_shape_dir = 'shapenet_pc/'

import pickle

# '02858304' (boat) and '02834778': 'bicycle', don't exist in SN-Core V2.
# synth_id_to_category_dict = {
#     '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
#     '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
#     '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
#     '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
#     '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
#     '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
#     '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
#     '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
#     '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
#     '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
#     '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
#     '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
#     '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
#     '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
#     '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
#     '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
#     '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
#     '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
#     '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
# }

id_to_category_dict = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02992529': 'cellphone'
}


id_to_category_dict_shapetalk = {
    '02691156': 'airplane',  '02773838': 'bag',        '04256520': 'sofa',  
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02871439': 'bookshelf',   '02876657': 'bottle',    '02880940': 'bowl',    
    '02933112': 'cabinet',    '03948459': 'pistol',    '03797390': 'mug',
    '02954340': 'cap',        '03001627': 'chair',      '03636649': 'lamp', 
    '03046257': 'clock',     '03325088': 'faucet',     '04379243': 'table',    
    '03467517': 'guitar',    '03513137': 'helmet',    '04225987': 'skateboard',
    '03624134': 'knife',          
    # display, dresser, vase, trashbin, weighted average, plant, scissors, flowerpot, person
}
id_to_index = list(id_to_category_dict.keys())
id_to_index_shapetalk = list(id_to_category_dict_shapetalk.keys())

class ShapeNetDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train',
                 sample_points_num: int = 1024, train_size: float = 0.8, 
                 random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.device = device
        self.split = split
        
        # breakpoint()

        # Read txt file
        with open(os.path.join(root_dir, shapetalk_file, split+".txt"), 'r') as f:
            # self.shape_paths = [line.strip() for line in f.readlines()]

            self.shape_paths = []
            for line in f.readlines():
                dir = line.strip()
                cls = dir[:8]

                if cls in id_to_category_dict_shapetalk.keys():
                    self.shape_paths.append(dir)
        

        # Random seed 고정
        np.random.seed(random_seed)
        
        # ShapeTalk default point number 16384
        self.npoints = 8192
        self.permutation = np.arange(self.npoints)
        self.sample_points_num = sample_points_num


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
        return len(self.shape_paths)

    def __getitem__(self, index):
        shape_path = self.shape_paths[index]

        cls = shape_path[:8]
        # cls_idx = id_to_index.index(cls)
        cls_idx = id_to_index_shapetalk.index(cls)

        shape_path = os.path.join(self.root_dir, top_shape_dir, shape_path)

        # npy file : (8192, 3)
        shape = np.load(shape_path)

        # Single-line preprocessing
        # shape = data.load_and_transform_3D([shape], self.device)

        # Sample & Normalize
        shape = self.random_sample(shape, self.sample_points_num)
        shape = self.pc_norm(shape)

        # Point-BERT는 여기서 float()
        shape = torch.from_numpy(shape).to(self.device)
    
        return shape, cls_idx

# data = ShapeNetDataset(
#             root_dir="/root/volume/Point-BERT/data/ShapeNet55-34/", split="train",
#             sample_points_num = 1024,
#         )

# from torch.utils.data import DataLoader
# bs = 12
# dataloader = DataLoader(data, batch_size=bs, shuffle=False, num_workers=16)
# print(next(iter(dataloader)))


class ShapeNetEmbDataset(Dataset):
    def __init__(self, emb_file: str='/root/volume/embeddings/embeddings-cls-train.npz', split: str='train',
                 random_seed: int = 42, device: str = 'cpu'):
        
        self.emb_file = emb_file
        self.device = device
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        self.data = np.load(emb_file)

    def __len__(self):
        return len(self.data.files)//2

    def __getitem__(self, index):
        
        output_hidden_states = self.data[f'{index}-output']
        cls = self.data[f'{index}-cls']

        output_hidden_states = torch.from_numpy(output_hidden_states).to(self.device)
        cls = torch.from_numpy(cls).to(self.device)

        return output_hidden_states, cls

# /root/volume/embeddings/embeddings-full.npz

class ShapeNetTsneDataset(Dataset):
    def __init__(self, emb_file: str='/root/volume/embeddings/embeddings-cls-train.npz', split: str='train',
                 random_seed: int = 42, device: str = 'cpu'):
        
        self.emb_file = emb_file
        self.device = device
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        self.data = np.load(emb_file)

    def __len__(self):
        return len(self.data.files)//2

    def __getitem__(self, index):
        
        output_hidden_states = self.data[f'{index}-output']
        cls = self.data[f'{index}-cls']

        output_hidden_states = torch.from_numpy(output_hidden_states).to(self.device)
        cls = torch.from_numpy(cls).to(self.device)

        return output_hidden_states, cls