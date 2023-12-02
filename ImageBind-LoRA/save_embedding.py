import logging
import torch
import data
import numpy as np
from tqdm import tqdm

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA
from datasets.shapetalk_embedding import ShapeTalkEmbDataset
from datasets.shapetalk import ShapeTalkDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, force=True)


lora = False
linear_probing = False
device = "cuda:0"  # "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
load_head_post_proc_finetuned = True
file_name = '../../volume/embeddings/embeddings-edit-test.npz'


import pdb

assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1

model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/deepspeed_3d_text_full_1024_v1/imagebind-epoch=57-val_loss=1.37.ckpt/checkpoint/mp_rank_00_model_states.pt", deepspeed=True)


if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                        #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints/lora/500_epochs_lp", postfix="_dreambooth_last")

model.eval()
model.to(device)

dataset = ShapeTalkDataset('/root/volume/datasets/shapetalk', pair_type='edit', split='test')
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=16)

e = {}
with torch.no_grad():
    for shape, _, text, _, idx in tqdm(iter(dataloader)):
        shape_e = model({ModalityType.SHAPE: shape.to(device)})[ModalityType.SHAPE]
        text_e = model({ModalityType.TEXT: text.to(device).squeeze(1)})[ModalityType.TEXT]
        for i in range(len(shape)):
            e[f'{idx[i]}-shape'] = shape_e[i].numpy(force=True)
            e[f'{idx[i]}-text'] = shape_e[i].numpy(force=True)



# shape = torch.cat(s, dim=0)
# text = torch.cat(t, dim=0)

# shape = shape.numpy(force=True)
# text = text.numpy(force=True)

np.savez(file_name, **e)
print(f"file saved in {file_name}")
