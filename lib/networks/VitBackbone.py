# utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import timm

import numpy as np
from lib.utils import utils
# ---------------------------------------------------------------------------- #
"""
TODO: Adjust NLST dataset to allow for 224x224 images when applied to non-regional group"""
class ViT_encoder(nn.Module):
    def __init__(self,
                 depth:int=8,
                 num_heads:int=8,
                 embed_dim:int=748,
                 patch_size:int=16,
                 image_size:int=224,
                 channels_in:int = 3,
                 n_classes:int = 2,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio:float=4,
                 ):
        super(ViT_encoder,self).__init__()
        '''
        Args
            maskratio: Experiment Variable defining the amount of the image masked
            imagesize: Original image size typically 64x64 for ROI but can be defined between 64-256
            embed_dim: Embedding Dimension
            Num_channels: Number of input channels. All CT are converted to 1xNxN
            Num_patches: Calculated either by the provided patch_size wanted or user defined
            Num_heads: Number of Attention heads for the ViT backbone
            Depth: Depth of ViT Backbone, defines the number of Blocks created.
            Norm_layer: Normalization layer function to be used
            mlp_ratio: Multilayered perceptron dimension ratio compared to the hidden_dimension or the embedding dimension.
        '''
        self.patch_embed = PatchEmbed(image_size, patch_size, channels_in, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
                                nn.Linear(embed_dim, n_classes),
                                nn.Sigmoid())
        
    def forward(self, x:torch.Tensor):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)    
        x = self.head(x)
        return x

class CustomHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Customize the output dim for your task
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_normal_(self.fc.weight)  # Good initialization for weights
        self.fc.bias.data.fill_(0)  # Initialize biases to 0

    def forward(self, x):
        return self.sigmoid(self.fc(x))
    

class Custom_ViT_B_16(nn.Module):
    def __init__(self,
                 depth:int=8,
                 num_heads:int=12,
                 embed_dim:int=768,
                 patch_size:int=16,
                 image_size:int=64,
                 channels_in:int = 1,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio:float=4,
                 custom:bool=False,
                 pretrain:bool=False
                 ):
        super(Custom_ViT_B_16,self).__init__()
        self.custom = custom
        self.pretrain = pretrain

        if custom:
            self.encoder = ViT_encoder(depth=depth,
                                       num_heads=num_heads,
                                       embed_dim=embed_dim,
                                       patch_size=patch_size,
                                       image_size=image_size,
                                       channels_in=channels_in,
                                       norm_layer=norm_layer,
                                       mlp_ratio=mlp_ratio)
        elif pretrain:
            self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrain, num_classes=2)
            self.encoder.head = CustomHead(embed_dim,2)
        else:
            self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrain,  num_classes=2)


    def forward(self,x):
        '''
        Args:
            x: Torch.Tensor of input image. 
        '''
        x = self.encoder(x)
        return x
    
    