import logging
import torch
import data
import numpy as np
from tqdm import tqdm

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from datasets.shapenet import ShapeNetDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, force=True)

device = "cuda:0"  # "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
file_name = '../../volume/embeddings/zero-embeddings30-cls-test.npz'


import pdb

model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/deepspeed_3d_text_full_1024_v1/imagebind-epoch=57-val_loss=1.37.ckpt/checkpoint/mp_rank_00_model_states.pt", deepspeed=True)


model.eval()
model.to(device)


# Change train, test when saving
dataset = ShapeNetDataset(
            root_dir="/root/volume/Point-BERT/data/ShapeNet55-34/", split="test",
            sample_points_num = 1024,
        )

bs = 512
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=16)

e = {}
with torch.no_grad():
    for index, (shape, cls) in enumerate(tqdm(dataloader)):
        # output_hidden_states = model({ModalityType.SHAPE: shape.to(device)})["output_hidden_states"]
        output_hidden_states = model({ModalityType.SHAPE: shape.to(device)})[ModalityType.SHAPE]
        
        # np_cls = np.array(cls)
        for i in range(len(shape)):
            # print(bs*index+i)
            e[f'{bs*index+i}-output'] = output_hidden_states[i].numpy(force=True)
            e[f'{bs*index+i}-cls'] = np.array([cls[i]])



# shape = torch.cat(s, dim=0)
# text = torch.cat(t, dim=0)

# shape = shape.numpy(force=True)
# text = text.numpy(force=True)

np.savez(file_name, **e)
print(f"file saved in {file_name}")
