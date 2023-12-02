import torch
import numpy as np
from torch import nn

import torch.nn.functional as F
# from changeit3d_model.mlp import MLP
from .mlp import MLP
import pdb

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
      
    def forward(self, x):
        return torch.relu(x)
    

class LatentDirectionFinder(nn.Module):
    def __init__(self):
        super().__init__()

        d_lang_model= 1024       
        shape_latent_dim = 1024
        in_dim = d_lang_model + shape_latent_dim
        ablation_version = 'decoupling_mag_direction'
        self_contrast=True
        self.gamma = None
        self.editor_unit = MLP(in_dim, [256, shape_latent_dim, shape_latent_dim, shape_latent_dim], b_norm=True, remove_final_bias=True)
        # self.editor_unit = MLP(in_dim, [256, shape_latent_dim, shape_latent_dim, shape_latent_dim*2, shape_latent_dim*2, shape_latent_dim, shape_latent_dim], b_norm=True, remove_final_bias=True)
        
        closure = ReLU()
            
        if ablation_version == 'decoupling_mag_direction':
            self.magnitude_unit = MLP(in_dim, [256, 128, 64, 1], closure=closure)
            unit_normalize_direction = True
        elif ablation_version == 'coupled':        
            unit_normalize_direction = False
            self.magnitude_unit = None    
        else:
            raise ValueError('ablation version of ChangeIt3D not understood.')
        
        self.context_free = True
        self.unit_normalize_direction = unit_normalize_direction
        self.self_contrast = self_contrast
        
        if not self.context_free:
            raise NotImplementedError('Not in this version.')

    def context_free_edit(self, z_lang, z_shape):
        
        edit_in = torch.cat([z_lang, z_shape], axis = 1) 

        edit_latent = self.editor_unit(edit_in)
        
        if self.unit_normalize_direction:
            edit_latent = F.normalize(edit_latent, dim=1)

        if self.magnitude_unit is not None:
            magnitude = self.magnitude_unit(edit_in)
            edit_latent *= magnitude
        else:
            magnitude = torch.zeros(len(z_lang))

        return edit_latent, magnitude


    def __call__(self, z_lang, z_shape):
        if self.context_free:
            return self.context_free_edit(z_lang, z_shape)
                            
    def single_step_train(self, z_lang, z_shape, gamma=0, adaptive_id_penalty="", device="cuda"):  
        
        
        self.train()
        
        # z_lang = z_lang.cpu()
        # z_shape = z_shape.cpu()
        
        edit_latent, _ = self(F.normalize(z_lang, dim = 1), F.normalize(z_shape, dim = 1))
        transformed_distractor = F.normalize(z_shape, dim = 1) + edit_latent


        distractor_text_sim =  F.cosine_similarity(F.normalize(z_lang, dim = 1),F.normalize(z_shape, dim = 1), dim = -1)
        edit_distractor_text_sim = F.cosine_similarity(F.normalize(z_lang, dim = 1), transformed_distractor, dim = -1)
        
        # combined_tensor = torch.stack((distractor_text_sim, edit_distractor_text_sim), dim=1)
        logits = torch.stack((distractor_text_sim, edit_distractor_text_sim), dim=1)
        # sums = combined_tensor.sum(dim=1, keepdim=True)
        # logits = combined_tensor / sums

        criterion = nn.CrossEntropyLoss()
        
        clip_loss = criterion(logits, torch.tensor([0.0,1.0]).to(device).repeat(logits.shape[0], 1)) 

        # correct_answers = (labels==1).sum()
        # correct_answers.item()

        labels = torch.argmax(logits, dim=1)
        batch_accuracy = (labels==1).float().mean()

        return clip_loss, batch_accuracy

