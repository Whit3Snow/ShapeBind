import os
from typing import Optional, Callable

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
# from models.imagebind_model import ModalityType

import pandas as pd
import torch
import numpy as np

# import data

# Create Shape-Text embedding pair or Shape Embedding directory & Text Embedding directory

# Inference : batch_size Number of items to do the inference on at once (default 256)
# npy에 담는 batch : write_batch_size Write batch size (default 10**6)


top_img_dir = 'images/full_size/'
shapetalk_file = 'language/shapetalk_raw_public_version_0.csv'
top_shape_dir = 'point_clouds/scaled_to_align_rendering/'

import pickle

class ShapeTalkEmbDataset(Dataset):
    def __init__(self, root_dir: str,
                 sample_points_num: int = 1024,
                 random_seed: int = 42, device: str = 'cpu'):
        self.root_dir = root_dir
        self.device = device
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        # ShapeTalk default point number 16384
        self.npoints = 16384
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
        return len(self.shape_paths)

    def __getitem__(self, index):
        shape_path = self.shape_paths[index]
        shape = np.load(shape_path)['pointcloud']

        # Sample & Normalize
        shape = self.random_sample(shape, self.sample_points_num)
        shape = self.pc_norm(shape)

        # Point-BERT는 여기서 float()
        shape = torch.from_numpy(shape).to(self.device)
    
        return shape, shape_path



class ShapetalkEmb_ClsDataset(Dataset):
    def __init__(self, emb_file: str='/root/ShapeBind/ImageBind-LoRA/embeddings.npz',
                 random_seed: int = 42, device: str = 'cpu'):
        
        self.emb_file = emb_file
        self.device = device
        
        # Random seed 고정
        np.random.seed(random_seed)
        
        self.data = np.load(emb_file)

    def __len__(self):
        return len(self.data.files)
    
    def __getitem__(self, index):
        
        path = self.data.files[index]
        label = path.split("/")[7]
        shape_emb = self.data[path]

        return shape_emb, label
    # def save_in_chunks(data, chunk_size, folder, base_filename):
    #     num_samples = data.shape[0]
    #     num_chunks = (num_samples + chunk_size - 1) // chunk_size  # 올림으로 청크 개수 계산

    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    #     for i in range(num_chunks):
    #         start = i * chunk_size
    #         end = start + chunk_size
    #         chunk = data[start:end]
    #         filename = os.path.join(folder, f"{base_filename}_{i}.npy")
    #         np.save(filename, chunk)

    # # 예시: 대규모 임베딩 데이터셋
    # large_embedding = np.random.rand(1000000, 512)  # 대용량 데이터 예시

    # # 1.44GB 당 분할 저장 (임의로 결정한 청크 크기)
    # chunk_size = 360000  # 이 크기는 실험을 통해 조정 필요
    # save_in_chunks(large_embedding, chunk_size, 'embeddings_folder', 'embedding')


