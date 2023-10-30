import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

import data

# ShapeTalkDataset
# 
# Output of ShapeTalkDataset is a text-shape pair



top_img_dir = 'images/full_size/'
shapetalk_file = 'language/shapetalk_raw_public_version_0.csv'
top_shape_dir = 'point_clouds/scaled_to_align_rendering/'


class ShapeTalkDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', max_utters: int = 5,
                 train_size: float = 0.8, random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.device = device
        self.max_utters = max_utters
        
        self.df = pd.read_csv(os.path.join(root_dir, shapetalk_file))

        self.idx = []
        train_idx, test_idx = train_test_split(range(len(self.df)), train_size=train_size, random_state=random_seed)

        if split == 'train':
            self.idx = train_idx
        elif split == 'test':
            self.idx = test_idx
        else:
            raise ValueError(f"Invalid split argument. Expected 'train' or 'test', got {split}")


    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        idx = self.idx[index]
        dt = self.df.loc[idx]

        text = f'This is a {dt["target_object_class"]}. ' + ' '.join(dt[f'utterance_{i}'] for i in range(self.max_utters) if type(dt[f'utterance_{i}']) is str)
        text = data.load_and_transform_text([text], self.device)

        shape_path = os.path.join(self.root_dir, top_shape_dir, dt["target_uid"] + ".npz")
        shape = np.load(shape_path)['pointcloud']
        shape = torch.from_numpy(shape).to(self.device)

        return text, shape