import logging
import torch
import data

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

logging.basicConfig(level=logging.INFO, force=True)


lora = False
linear_probing = False
device = "cuda:0"  # "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
load_head_post_proc_finetuned = True
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

# text_list=["bird",
#            "car",
#            "dog3",
#            "dog5",
#            "dog8",
#            "grey_sloth_plushie"]
text_list=[
           "guitar",
           "car",
           "chair",
           "dog",
           "cat",
           "airplane"]
text_list = ["This is a "+text for text in text_list]

# image_paths=[".assets/bird_image.jpg",
#              ".assets/car_image.jpg",
#              ".assets/dog3.jpg",
#              ".assets/dog5.jpg",
#              ".assets/dog8.jpg",
#              ".assets/grey_sloth_plushie.jpg"]
image_paths=[".assets/chair.png",
            ".assets/guitar.png",
            ".assets/plane.png",
            ".assets/bird_image.jpg",
             ".assets/car_image.jpg",
             ".assets/dog3.jpg",
             ".assets/grey_sloth_plushie.jpg",
             ".assets/phone.jpg"]

# pc_paths=[".assets/chair.ply",
#           ".assets/plane.ply",
#           ".assets/phone.ply"]
          
pc_paths=[
          ".assets/guitar.npz",
          ".assets/chair.npz",
          ".assets/airplane.npz",
          ".assets/airplane.npz",
          ]

# Instantiate model
# model = imagebind_model.imagebind_huge(pretrained=True)
# with open('output_pretrained_model.txt', 'w') as f:
#     for name, child in model.named_children():
#         for param in child.parameters():
#             print(name, param, file=f)
# model = imagebind_model.imagebind_huge(pretrained=True, model_path="/root/volume/.checkpoints/full_1024/last.ckpt")
# model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/3d_2d_full_1024/last.ckpt")

# breakpoint()
model = imagebind_model.imagebind_huge(pretrained=True, device=device, model_path="/root/volume/.checkpoints/full_1024/last.ckpt")
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


# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, to_tensor=True),
    # ModalityType.SHAPE: data.load_and_transform_ply(pc_paths, device),
    ModalityType.SHAPE: data.load_and_transform_3D(pc_paths, device),
}


with torch.no_grad():
    embeddings = model(inputs)

# print("Vision x Text: ")
# print(torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1), dim=-1))
# print(torch.argmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * (lora_factor if lora else 1), dim=-1))

print("Shape x Text: ")
print(torch.softmax(embeddings[ModalityType.SHAPE] @ embeddings[ModalityType.TEXT].T, dim=-1))
print(torch.argmax(embeddings[ModalityType.SHAPE] @ embeddings[ModalityType.TEXT].T, dim=-1))

# print("Shape x Vision: ")
# print(torch.softmax(embeddings[ModalityType.SHAPE] @ embeddings[ModalityType.VISION].T, dim=-1))
# print(torch.argmax(embeddings[ModalityType.SHAPE] @ embeddings[ModalityType.VISION].T, dim=-1))