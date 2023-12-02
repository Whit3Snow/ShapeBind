import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from changeit3d_model.mlp import MLP
from losses.chamfer import chamfer_loss

import pdb
class PointcloudAutoencoder(nn.Module):
    def __init__(self, encoder_latent_dim):

        super(PointcloudAutoencoder, self).__init__()

        self.encoder_latent_dim = encoder_latent_dim
        self.ae_decoder = MLP(in_feat_dims=encoder_latent_dim,
                         out_channels=[256,256,512] + [4096 * 3],
                         b_norm=False)
        

    def __call__(self, z_shape, bcn_format=True):
        """
        :param z_shape: B x encoder_latent_dim
        :param bcn_format: the AE.encoder works with Batch x Color (xyz) x Number of points format
        """
        b_size, _ = z_shape.shape
        recon = self.ae_decoder(z_shape).view([b_size, 4096, 3])
        return recon



    def single_step_train(self, pc, z_shape, device='cuda', loss_rule="chamfer"):
        """ Train the auto encoder for one epoch based on the Chamfer (or EMD) loss.
        :param loader: (train) pointcloud_dataset loader
        :param optimizer: torch.optimizer
        :param device: cuda? cpu?
        :return: (float), average loss for the epoch.
        """
        self.train()

        recon = self(z_shape)
        loss = chamfer_loss(pc, recon).mean()

        return loss