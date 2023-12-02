import torch.nn as nn
from models import imagebind_model
import torch
from models.imagebind_model import ModalityType, load_module
import data
from tqdm import tqdm

from datasets.shapenet import ShapeNetEmbDataset
from torch.utils.data import DataLoader

import pdb
id_to_category_dict_shapetalk = {
    '02691156': 'airplane',  '02773838': 'bag',       '04256520': 'sofa',  
    '02808440': 'bathtub',   '02818832': 'bed',       '02828884': 'bench',
    '02871439': 'bookshelf', '02876657': 'bottle',    '02880940': 'bowl',    
    '02933112': 'cabinet',   '03948459': 'pistol',    '03797390': 'mug',
    '02954340': 'cap',       '03001627': 'chair',     '03636649': 'lamp', 
    '03046257': 'clock',     '03325088': 'faucet',    '04379243': 'table',    
    '03467517': 'guitar',    '03513137': 'helmet',    '04225987': 'skateboard',
    '03624134': 'knife',          
    # display, dresser, vase, trashbin, weighted average, plant, scissors, flowerpot, person
}

text_list = list(id_to_category_dict_shapetalk.values())
text_list = ["This is a "+text for text in text_list]


device = "cuda:0"  # "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/deepspeed_3d_text_full_1024_v1/imagebind-epoch=57-val_loss=1.37.ckpt/checkpoint/mp_rank_00_model_states.pt", deepspeed=True)

model.eval()
model.to(device)


test_dataset = ShapeNetEmbDataset(
            '/root/volume/embeddings/zero-embeddings30-cls-test.npz', split="test",
            device=device,
            # sample_points_num = args.sample_points_num
            # device=args.device,
            # transform=ContrastiveTransformations(contrast_transforms, n_views=2 if args.self_contrast else 1)
        )


val_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

with torch.no_grad():
    cls_emb = model({ModalityType.TEXT: data.load_and_transform_text(text_list, device)})[ModalityType.TEXT]
    
total_num = 0
total_correct = 0
for index, (shape_emb, cls) in enumerate(tqdm(val_loader)):
    cls=cls.squeeze(1)

    total_num += len(cls)
    total_correct += (cls==torch.argmax(shape_emb @ cls_emb.T, dim=-1)).float().sum()
    print(f"batch accuracy: {(cls==torch.argmax(shape_emb @ cls_emb.T, dim=-1)).float().mean()}")

accuracy = total_correct/total_num

print(f'{accuracy*100}%')






