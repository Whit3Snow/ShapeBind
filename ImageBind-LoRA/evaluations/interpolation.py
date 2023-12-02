import logging
import torch
import data
import os

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

from models.pc_ae_net import PointcloudAutoencoder
from utils.latent_ops import slerp

logging.basicConfig(level=logging.INFO, force=True)

lora = False
linear_probing = False
device = "cuda:0"  # "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
load_head_post_proc_finetuned = True

decoder_path = '/root/volume/.checkpoints/decoder/changeit3m-epoch=347-val_loss=0.00.ckpt/checkpoint/mp_rank_00_model_states.pt'
cls = 'chair'
sample = 2

top_img_dir = 'images/full_size/'
shapetalk_file = 'language/shapetalk_raw_public_version_0.csv'
top_shape_dir = 'point_clouds/scaled_to_align_rendering/'


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



# pc_paths=[".assets/chair.ply",
#           ".assets/plane.ply",
#           ".assets/phone.ply"]
          
pc_paths=[
          ".assets/guitar.npz",
          ".assets/chair.npz",
          ".assets/airplane.npz",
          ".assets/airplane.npz",
          ]


# breakpoint()
model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/deepspeed_3d_text_full_1024_v1/imagebind-epoch=57-val_loss=1.37.ckpt/checkpoint/mp_rank_00_model_states.pt", deepspeed=True)
# with open('output_loaded_model.txt', 'w') as f:
#     for name, child in model.named_children():
#         for param in child.parameters():
#             print(name, param, file=f)
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

# get shapes from cls


shape_dir = os.path.join('/root/volume/datasets/shapetalk', top_shape_dir, cls)
model = [os.path.join(shape_dir, t) for t in os.listdir(shape_dir)]
shape_paths = []
for p in model:
    shape_paths.extend([os.path.join(p, t) for t in os.listdir(p)])

shape_paths[:sample]


# Load data
inputs = {
    ModalityType.SHAPE: data.load_and_transform_3D(shape_paths, device),
}


with torch.no_grad():
    embeddings = model(inputs)



checkpoint = torch.load(decoder_path, map_location=device)
decoder = PointcloudAutoencoder(1024)
# If your checkpoint is a dictionary with more than just the state_dict
checkpoint_state_dict = checkpoint['state_dict']  # or whatever the correct key is

# Then modify the keys
new_state_dict = {key.replace('model.', ''): value for key, value in checkpoint_state_dict.items()}
decoder.load_state_dict(new_state_dict, strict=True)

inter_l = slerp(embeddings[0], embeddings[1], 0.5)

points = decoder(inter_l.unsqueeze(0))
