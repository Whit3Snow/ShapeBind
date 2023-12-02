# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os
import argparse
from datetime import datetime

try:
    import comet_ml
except ImportError:
    comet_ml = None
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning(
        "Matplotlib not installed. This is not needed if you run this script as --headless")

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module

# DeepSpeed Optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam

logging.basicConfig(level=logging.INFO, force=True)

from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import pdb
from changeit3d_model.changeit3d_net import LatentDirectionFinder
# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True

from changeit3d_model.mlp import MLP
import torch.nn as nn



    
class ChangeIt3M(L.LightningModule):
    def __init__(self, lr=5e-4, lr_patience=10, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42,
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95),
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False
                 ):
        super().__init__()
        assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. " \
            "Linear probing stores params in lora_checkpoint_dir"
        self.save_hyperparameters()
        self.editor_model = LatentDirectionFinder()


    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        #                         betas=self.hparams.momentum_betas)
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # DeepSpeed Offload CPU Optimizer  
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        #                         betas=self.hparams.momentum_betas)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )

        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                              patience=self.hparams.lr_patience,
                                                              verbose=True, min_lr=5e-8),
            "monitor": 'val_loss',
        }

        return [optimizer], [lr_scheduler]

    def editor_loss(self, batch, mode="train"):
        feats_a_tensor, feats_b_tensor = batch

        # class_a is always "vision" according to ImageBind !!! < can be any modality according to ShapeBind
        # embedding : (a, b) : (Shape, Text) 순서
        # Editor 모델에 그대로 넣기 : [12, 1024]로 input
        
        edit_loss, batch_accuracy = self.editor_model.single_step_train(feats_a_tensor, feats_b_tensor)


        self.log(mode + "_acc_top1", batch_accuracy, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        self.log(mode + "_loss", edit_loss, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        
        return edit_loss

    def training_step(self, batch, batch_idx):
        return self.editor_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.editor_loss(batch, mode="val")

    def on_validation_epoch_end(self):

        # Editor 저장
        pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the changeit3m editor with PyTorch Lightning")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="/root/volume/datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["shapetalk"], choices=["dreambooth", "shapetalk"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing",
                        action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str,
                        default="./.logs", help="Directory to save the logs")
    
    # Get the current date and time
    current_time = datetime.now()
    # Format it as a string in the format of 'YearMonthDay-HourMinuteSecond', e.g., '20231109-153025'
    default_log_dir = current_time.strftime("%Y%m%d-%H%M%S")

    parser.add_argument("--loggers_name", type=str,
                    default=default_log_dir, help="Name to save the logs")

    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (Don't plot samples on start)")

    # Add sampler 1024:PointBERT 2048:ChangeIt3D
    parser.add_argument("--sample_points_num", type=int, default=1024,
                        help="Number of points to sample")
    parser.add_argument("--max_epochs", type=int, default=500,
                        help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lr_patience", type=int, default=10, help="Learning rate scheduler patience")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float,
                        default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true",
                        help="Use self-contrast on the image modality")

    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "shape"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")

    # parser.add_argument("--dvae_device", type = str, default = "cuda:0", help = "dvae's map location device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create loggers
    loggers = []
    for logger in args.loggers if args.loggers is not None else []:
        if logger == "wandb":
            wandb.init(project="imagebind", config=args)
            wandb_logger = pl_loggers.WandbLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(wandb_logger)
        elif logger == "tensorboard":
            tensorboard_logger = pl_loggers.TensorBoardLogger(
                save_dir=args.loggers_dir,
                name=args.loggers_name)
            loggers.append(tensorboard_logger)
        elif logger == "comet":
            comet_logger = pl_loggers.CometLogger(
                save_dir=args.loggers_dir,
                api_key=os.environ["COMET_API_KEY"],
                workspace=os.environ["COMET_WORKSPACE"],
                project_name=os.environ["COMET_PROJECT_NAME"],
                experiment_name=os.environ.get("COMET_EXPERIMENT_NAME", None),
            )
            loggers.append(comet_logger)
        elif logger == "mlflow":
            mlflow_logger = pl_loggers.MLFlowLogger(
                save_dir=args.loggers_dir,
                experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
                tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                run_name="imagebind"
            )
            loggers.append(mlflow_logger)
        else:
            raise ValueError(f"Unknown logger: {logger}")

    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True

    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device_name)


    train_datasets = []
    test_datasets = []

    # Load datasets
    if "shapetalk" in args.datasets:
        from datasets.embeddings import EmbeddingDataset
        train_datasets.append(EmbeddingDataset(
            '../../volume/embeddings/embeddings-full.npz', split="train",
            sample_points_num = args.sample_points_num
            # device=args.device,
            # transform=ContrastiveTransformations(contrast_transforms, n_views=2 if args.self_contrast else 1)
        ))
        test_datasets.append(EmbeddingDataset(
            '../../volume/embeddings/embeddings-full.npz', split="test",
            sample_points_num = args.sample_points_num
            # device=args.device,
            # transform=ContrastiveTransformations(contrast_transforms, n_views=2 if args.self_contrast else 1)
        ))

    if len(args.datasets) == 1:
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # Visualize some examples
    if not args.headless:
        # TODO: visualize 3d pointcloud
        pass
        # NUM_IMAGES = args.batch_size
        # imgs = [torch.stack(train_dataset[idx][0], dim=0) for idx in range(NUM_IMAGES)]
        # imgs = torch.stack(imgs, dim=0)
        # img_grid = torchvision.utils.make_grid(imgs.reshape(-1, *imgs.shape[2:]), nrow=6, normalize=True, pad_value=0.9)
        # img_grid = img_grid.permute(1, 2, 0)
        # plt.figure(figsize=(10, 5))
        # plt.title(f"Augmented image examples of the available datasets: {args.datasets}")
        # plt.imshow(img_grid.cpu())
        # plt.axis("off")
        # plt.show()
        # plt.close()

    # Parse indices of layers to apply LoRA
    lora_layer_idxs = {}
    lora_modality_names = []
    modalities = ["vision", "text", "shape"]
                  #   "audio", "thermal", "depth", "imu",

    for modality_name in args.lora_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            lora_layer_idxs[modality_type] = getattr(
                args, f'lora_layer_idxs_{modality_name}', None)
            if not lora_layer_idxs[modality_type]:
                lora_layer_idxs[modality_type] = None
            lora_modality_names.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    # Train dataset
    model = ChangeIt3M(max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr, lr_patience=args.lr_patience,
                           weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,
                           temperature=args.temperature,
                           num_workers=args.num_workers, self_contrast=args.self_contrast,
                           lora=args.lora, lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=args.linear_probing,
                        #    dvae_device = args.dvae_device,
                           )

    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_loss", dirpath=args.full_model_checkpoint_dir,
                                                       filename="changeit3m-{epoch:02d}-{val_loss:.2f}",
                                                       save_last=True, mode="min")]}
    else:
        checkpointing = {
            "enable_checkpointing": args.full_model_checkpointing, }

    # DDP
    # from lightning.pytorch.strategies import DDPStrategy
    # trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
    #                   devices=1 if ":" not in device_name else [int(idx) for idx in device_name.split(":")[1:]],
    #                 #   strategy="ddp",
    #                   strategy='ddp_find_unused_parameters_true',
    #                 #   strategy=DDPStrategy(find_unused_parameters=True)
    #                   deterministic=True,
    #                   max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
    #                   logger=loggers if loggers else None, **checkpointing)
    
    # Deepspeed
    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=1 if ":" not in device_name else [int(idx) for idx in device_name.split(":")[1:]],
                      strategy="deepspeed_stage_2",
                    #   strategy="deepspeed_stage_2_offload", 
                    #   precision=16,
                      deterministic=True,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers if loggers else None, **checkpointing)



    trainer.fit(model, train_loader, val_loader)
