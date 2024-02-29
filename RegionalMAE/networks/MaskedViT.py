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

import numpy as np
from RegionalMAE.utils import utils
# ---------------------------------------------------------------------------- #

def build_2d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    '''
    '''
    h, w = grid_size, grid_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed

class MAEEncoder(nn.Module):
    """
    Masked ViT Encoder Module - Backbone is either a ViT-B-16 or ViT-S-16 
    No pretraining used for MAE implementation!
    """
    def __init__(self,
                 embed_dim:int=748,
                 num_heads:int=8,
                 depth:int=24,
                 norm_layer=nn.LayerNorm,
                 mlp_ratio:float=4.,

                ):
        super(MAEEncoder,self).__init__()
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

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

    def forward(self, x:torch.Tensor):
        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)
class MAEDecoder(nn.Module):
    """
    Masked ViT Decoder Module
    """
    def __init__(self,
                 patch_size:int,
                 embed_dim:int=748,
                 decoder_dim:int=256,
                 decoder_heads:int=16,
                 num_patches:int = 8,
                 block_depth:int=4,
                 mlp_ratio:float=4.,
                 norm_layer=nn.LayerNorm,
                 ):
        super(MAEDecoder,self).__init__()
        '''
        Args
        '''
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_dim))
        self.decoder_pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), require_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) 
            for i in range(block_depth)])
        
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2*1, bias=True)

    def forward(self, x:torch.Tensor, ids:torch.Tensor):
        '''
        '''
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0],ids.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 

        x_ = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embedding

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x
        

class MaskedViT(nn.Module):
    """
    Masked ViT Autoencoder Module
    """
    def __init__(self,
                 maskratio:int=None,
                 region:str=None,
                 patch_size:int = 4, 
                 inputsize:int = 64, 
                 inputheads:int = 8,
                 chn_in:int = 1,
                 embed_dim:int=748,
                 encoder_depth:int=12,
                 decoder_depth:int=8,
                 decoder_heads:int=16,
                 decoder_dim:int=256,
                 loss_norm:bool=False,
                 norm_layer = nn.LayerNorm,
                 maskstyle:str='Random',
                 ) -> None:
        '''
        Args:
            maskratio:
            patch_size: describes the size each patch will have for an image (NxN)
            inputsize: original image input size
            inputheads: number of attention heads present for the model, this variable should be dependent on whether whole-chest or ROI are analyzed
            chn_in: number of channel for a given input image
            num_classes: number of classification classes for the classifer (benign vs. malignant)
            embed_dim: size of the hidden embedding of the model
            depth: number of convolutions applied to the model (DEPRECATED used in MaskedCNNs)
        '''
        super(MaskedViT, self).__init__()

        # Defining Parameters for Masking Image
        self.maskstyle = maskstyle
        self.region = region
        self.maskratio = maskratio or None
        self.region= region or None

        # Defining Patch Embedding
        self.patch_embed = PatchEmbed(inputsize, patch_size, chn_in, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.encoder = MAEEncoder(imagesize=inputsize,
                                  embed_dim=embed_dim,
                                  num_channels=chn_in,
                                  num_patches=self.num_patches,
                                  num_heads=inputheads,
                                  depth=encoder_depth,
                                  maskstyle=self.maskstyle)
        
        self.decoder = MAEDecoder(patch_size=patch_size,
                                  embed_dim=embed_dim,
                                  decoder_embed=decoder_dim,
                                  num_patches=self.num_patches,
                                  decoder_heads=decoder_heads,
                                  block_depth=decoder_depth,
                                  )

    def init_weights(self, m):
        '''
        Initializes Model Weights using Uniform Distribution 
        TODO: Either include a load_weights function or add option to initialize weights based on a provided state_dict. 
        '''
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self,imgs):
        '''
        Args: 
            imgs: [Batch, Channel_in, Height, Width]
        Returns:
            patchx: [Batch, N_Patches, p*p]
        '''
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.chn_in, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)

        return x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.chn_in))
    
    def unpatchify(self, patchx):
        '''
        Converts input from [B,L,p*p] to [b,1,h,w]
        Args:
            patchx: [B, L, p*p]
        Returns:
            imgs: [B,1,H,W]
        '''
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.chn_in))
        x = torch.einsum('nhwpqc->nchpwq', x)

        return x.reshape(shape=(x.shape[0], self.chn_in, h * p, h * p))
    
    def random_mask(self, x: torch.Tensor):
        '''
        Randomly select a list of patches to drop per sample shuffling
        Args:
            x: [B,L,p*p]
        Returns:
            maskx: [B, L] Where L is either 0 or 1 (0 - kept, 1 - masked)
        '''
        B,L,D = x.shape
        
        len_keep = int(L*(1-self.mask_ratio))
        noise = torch.rand(B,L, device = x.device)
        
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:,:len_keep]

        masked = torch.gather(x,dim=1,index=ids_keep.unsqueeze(-1).repeat(1,1,D)) # Generate Binary mask: 0 keep, 1 is remove
        mask = torch.ones([B,L],device = x.device)
        mask[:,:len_keep] = 0
        mask = torch.gather(mask,dim=1,index=ids_restore) # Unshuffle and get binary mask
        
        return masked, mask, ids_restore
    
    def regional_labels(self, x:torch.Tensor, x_mask:torch.Tensor):
        '''
        Removes Patches defined as being part of a specific region. Note, suing regional_mask overights the use of mask-ratio as
        the whole region is removed from the input data. 
        -----------
        Parameters:
            x: torch.Tensor containing the original patches in a flattened shape
            region: string that defines which region should be removed from the image. 
        --------
        Returns:
        '''
        patched_masks = self.patchify(x_mask)
        B_mask, L_mask, D_mask = patched_masks.shape

        label_region = torch.zeros(B_mask,L_mask, device=x.device)
        for b in range(B_mask):
            for l in range(L_mask):
                segment_ratio = sum(patched_masks[b][l][:])/len(D_mask)
                if segment_ratio > 0.5:
                    label_region[B_mask,L_mask] = 1 # Tumor Region
                elif segment_ratio < 0.5 and segment_ratio > 0.01:
                    label_region[B_mask,L_mask] = 2 # Boundary Region
                else:
                    label_region[B_mask,L_mask] = 3 # Parenchyma
        
        return label_region

    def mask_specific_region(self, x, labels, mask_ratio):
        '''
        Removes Patches based on what region they belong to in a given image.
        -----------
        Parameters:
        --------
        Returns:
        '''
        region_to_mask = {'tumor':1, 'boundary':2, 'parenchyma':3}
        B, L, D = x.shape
        
        # Select only the indices of the patches belonging to the specified region
        ids_region_to_mask = torch.nonzero(labels == region_to_mask[self.region], as_tuple=True)[1]
        ids_keep = torch.tensor([idx for idx in range(L) if idx not in ids_region_to_mask], device=x.device)
        
        masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([B, L], device=x.device)
        mask[:, ids_region_to_mask] = 0
        
        return masked, mask, ids_keep

    def forward(self, x:torch.Tensor):
        '''
        '''
        x, x_mask = x['image'], x['nodule_mask']
        region_labels = self.regional_labels(x, x_mask)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # masking: length -> length * mask_ratio
        if self.region == None:
            x, mask, ids = self.random_mask(x, self.maskratio)
        else:
            x, mask, ids = self.mask_specific_region(x, region_labels)

        latent = self.encoder(x)
        pred = self.decoder(latent, ids)
        
        return {'img': x,
                'mask': mask,
                'pred': pred
                }
