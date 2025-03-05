# utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description:
'''

# Libraries
# ---------------------------------------------------------------------------- #
from typing import Callable, List, Optional, Tuple, Union
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import time
import numpy as np
from lib.utils import utils
from lib.networks.pos_embed import get_2d_sincos

# TODO: Learn trace utilities and use torch import _assert more often
# ---------------------------------------------------------------------------- #


class Classifier(nn.Module):
    def __init__(self, dim:int=768, n_classes:int=2, depth:int=1, scaling:float=0.6, flatten:bool=False, pool:bool=False):
        super().__init__()
        
        self.dim = dim
        self.max_depth = depth
        self.scaling = scaling
        self.pool = pool
        self.flatten = flatten
        self.n_classes = n_classes
        
        layers = []
        self.flat = nn.Flatten()

        for i in range(self.max_depth):
            layers.append(self.LinearBlock(i+1, dim, scaling))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights_)

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias,0)
            nn.init.constant_(m.weight,1.0)
            
    def LinearBlock(self, depth:int, dim_in:int, scaling:float):
        """
        """
        if depth == self.max_depth:
            dim_out = self.n_classes
            return nn.Sequential(nn.Linear(dim_in, dim_out),
                                 nn.Sigmoid())
        
        else:
            dim_out = int(dim_in * scaling)
            return nn.Sequential(
                                 nn.Linear(dim_in, dim_out),
                                 nn.Dropout(),
                                 nn.ReLU(inplace=True))
    
    def forward(self,x:torch.Tensor):
        '''
        Forward Function for Classifier Module
        -----
        Args:
            x (torch.Tensor): Input tensor of shape (B, P, D)
        Returns:
            x (torch.Tensor): Output tensor of shape (B, classes)
        '''
        if self.pool:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        if self.flatten:
            x = self.flat(x)

        x = self.mlp(x)
        return x

class PatchEmbedding2D(nn.Module):
    def __init__(self,
                 image_size:int=64,
                 chn_in:int=1,
                 patch_size:int=16,
                 dim:int=768,
                 norm_layer:Optional[Callable]=None,
                 flatten:bool=True,
                 bias:bool=True):
        super().__init__()
        # NOTE: We are assuming that the image is a 2D patch and not a 3D image, and that all sizes are the same. (1x64x64)
        self.img_size = image_size
        self.flatten = flatten
        self.num_patches = (image_size // patch_size) * (image_size // patch_size) 
        
        self.proj = nn.Conv2d(chn_in, dim, kernel_size=patch_size, stride=patch_size, bias=bias)        
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()
        
        self.apply(self._init_weights_)

    def _init_weights_(self,m):
        pass
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
        Args:
            - x (torch.Tensor): [Batch, Channel, Height, Width]
        '''
        B,C,H,W = x.shape

        if self.img_size is not None:
            assert H == self.img_size, f"Input height ({H}) doesn't mathc model ({self.img_size})"
            assert W == self.img_size, f"Input height ({W}) doesn't mathc model ({self.img_size})"
            
        x = self.proj(x)                    # BCHW -> Conv2d -> [B Dim H/patch W/patch]
        
        if self.flatten:
            x = x.flatten(2).transpose(1,2) # [B Dim H/patch W/patch] -> [B Length Channels]

        x = self.norm(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size:int=64, patch_size:int=8, chn_in:int=1,
                 dim:int=768, depth:int=12, num_heads:int=12, mlp_ratio:float=4., qkv_bias:bool=False,
                 attn_drop_rate:float=0., drop_rate:float=0., drop_path_rate:float=0.,
                 norm_layer:Optional[Callable]=None, act_layer:Optional[Callable]=None):
        super().__init__()    
        '''
        Args:
            img_size (int): Size of the input image.
            patch_size (int): Size of each patch.
            chn_in (int): Number of input channels.
            dim (int): Dimension of the embedding.
            depth (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of the MLP.
            qkv_bias (bool): Whether to include bias in the QKV projection.
            attn_drop_rate (float): Dropout rate for attention layers.
            drop_rate (float): Dropout rate
            norm_layer (Optional[Callable]): Normalization layer.
            act_layer (Optional[Callable]): Activation layer.
        '''

        self.chn_in = chn_in
        self.patch_size = patch_size
        self.patch_emb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=chn_in, embed_dim=dim)
        self.num_patches = self.patch_emb.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_emb.num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # TODO: Review Block class in timm to make sure we are using it properly - remove custom ViTBlock we built.
        depth_drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.blocks = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                #   proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=depth_drop_rate[i], norm_layer=norm_layer, act_layer=act_layer)
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights_)
    
    def _init_weights_(self, m):
        '''
        NOTE: Stopped using 2dsin_cos for patch_embedding, but should consider adding it back into the model
        '''
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x:torch.Tensor, maskratio:float):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape                   # Batch, Patch Numbers, Embed Dimensions
        len_keep = int(L * (1 - maskratio))
        
        noise = torch.rand(N, L, device=x.device)
        shuffled_ids = torch.argsort(noise, dim=1) 
        restore_ids = torch.argsort(shuffled_ids, dim=1)

        ids_keep = shuffled_ids[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore_ids)
        
        return x_masked, mask, restore_ids

    def mask_specific_region(self, x,region):
        """
        Masks all patches belonging to a specific region in the input data and returns the indices to restore.
        -----
        Args:
            x (torch.tensor): The input data tensor (e.g., patch embeddings).
            region (str): The name of the region to mask (e.g., 'Tumor').
        --------
        Returns:
            x_masked (torch.tensor): The input data with the specified region masked.
            mask (torch.tensor): The binary mask indicating masked patches (1 for masked, 0 for kept).
            ids_restore (torch.tensor): The indices of the masked patches to be restored.
        """

        N, L, D = x.shape
        if self.patch_size == 16:
            tumor_indices = [5, 6, 9, 10]
        if self.patch_size == 8:
            tumor_indices = [18,19,20,21,26,27,28,29,34,35,36,37,42,43,44,45]

        parenchyma_indices = [i for i in range(L) if i not in tumor_indices]
        
        # Keep Region Subset
        if region == 'Tumor':
            all_indices = parenchyma_indices + tumor_indices
            ids_keep = torch.tensor(parenchyma_indices, device=x.device)
            len_keep = len(parenchyma_indices)       
        elif region == 'Parenchyma':
            all_indices = tumor_indices + parenchyma_indices
            ids_keep = torch.tensor(tumor_indices, device=x.device)
            len_keep = len(tumor_indices)
        else:
            raise ValueError(f"Invalid region: {region}. Choose 'Tumor' or 'Parenchyma'.")

        all_indices = torch.tensor(all_indices, device = x.device)
        all_indices = all_indices[None, :].repeat(N, 1)
        ids_keep = ids_keep[None, :].repeat(N, 1)
    
        ids_restore = torch.argsort(all_indices, dim=1)
    
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, region:str=None, task:str=None, maskratio:float=0):
        B, C, H, W = x.shape
        x = self.patch_emb(x)                           # Input: [B,C,H,W] -> [B, Emb, Patch]
        x = x + self.pos_embed[:,1:,:]    
        cls_token = self.cls_token.expand(B, -1, -1)    # Create class token

        if task != None:
            if region == "None":
                x, mask, ids_restore = self.random_masking(x, maskratio)        # Random Masking
            else:
                x, mask, ids_restore = self.mask_specific_region(x, region)     # Mask Regional Patches

            cls_token = self.cls_token + self.pos_embed[:,:1,:] # 
            cls_tokens = cls_token.expand(x.shape[0],-1,-1)     # Expand to match B size
            x = torch.cat((cls_tokens, x), dim=1)               # Add Class tokens to input
        
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x, mask, ids_restore
        
        else:
            cls_tokens = cls_token.expand(x.shape[0],-1,-1)
            x = torch.cat((cls_tokens, x), dim=1)
        
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

        return x

class ViTDecoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, chn_in=1, dim=768, scaling=1, depth=4, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer:Optional[Callable]=None, act_layer:Optional[Callable]=None):
        super().__init__()

        self.num_patches = (img_size//patch_size)**2
        self.decoder_dim = int(dim * scaling)
        self.num_tokens = 1         
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)
        depth_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        
        # VIT-Pytorch Library Example
        # # self.mask_tokens = nn.Parameter(torch.zeros(1,1,self.decoder_dim))
        # self.mask_tokens = nn.Parameter(torch.randn(1,1,self.decoder_dim))
        # self.decoder_pos_emb = nn.Embedding(self.num_patches, self.decoder_dim)

        # META's MAE Decoder Example
        self.encoder_to_decoder = nn.Linear(dim, self.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1,1, self.decoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_dim))

        self.blocks = nn.ModuleList([
            Block(dim=self.decoder_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=depth_path_rate[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(dim)                                     # Should this be normalized on output?
        self.pixels = nn.Linear(self.decoder_dim, patch_size**2 * chn_in)

        self.apply(self._init_weights_)

    def _init_weights_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    def forward(self, x:torch.Tensor, restore_ids:torch.Tensor):
        x= self.encoder_to_decoder(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], restore_ids.shape[1]+1-x.shape[1],1)
        x_ = torch.cat([x[:,1:,:],mask_tokens], dim=1)
        x_ = torch.gather(x_,dim=1, index=restore_ids.unsqueeze(-1).repeat(1,1,x.shape[2]))
        x = torch.cat([x[:,:1,:], x_],dim=1)
        
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pixels(x)

        return x[:,1:,:]

class CustomMAE(nn.Module):
    def __init__(self, img_size:int=64, patch_size:int=8, chn_in:int=1, heads:int=12, encoder_depth:int=8, decoder_depth:int=4,
                 embed_dim:int=768, mlp_ratio:float=4., drop_rate:float=0, att_drop_rate:float=0., n_classes:int=2,
                 qkv_bias:bool=False, mask_ratio=0.75, tlearn=None, region=None):
        super().__init__()
        """
        -----
        Args:
            img_size (int): Size of the input image.
            patch_size (int): Size of each patch.
            chn_in (int): Number of input channels.
            heads (int): Number of attention heads.
            encoder_depth (int): Number of encoder layers.
            decoder_depth (int): Number of decoder layers.
            embed_dim (int): Dimension of the embedding.
            dec_embedim (int): Dimension of the decoder embedding.
            mlp_dim (int): Dimension of the MLP.
            mlp_ration (int): Ratio of the MLP. [NOTE: Currently not really implemented, used to base mlp of this ratio but not in current version]
            drop_rate (float): Dropout rate.
            att_drop_rate (float): Attention dropout rate.
            n_classes (int): Number of classes.
            qkv_bias (bool): Whether to include bias in the QKV projection. [Disabled]
            mask_ratio (float): Ratio of patches to mask.
            tlearn (str): Task to learn.
            region (str): Region to mask.       
        """
        self.chn_in = chn_in
        self.tlearn = tlearn
        self.region = region
        self.patch_size = patch_size
        self.num_patchs = (img_size // patch_size)**2
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        
        self.encoder = ViTEncoder(img_size=img_size,
                                  patch_size=patch_size,
                                  chn_in=chn_in,
                                  num_heads= heads,
                                  depth=encoder_depth,
                                  dim=embed_dim,
                                  mlp_ratio= mlp_ratio,
                                  drop_rate=drop_rate,
                                  attn_drop_rate=att_drop_rate
                                  )
        
        self.decoder = ViTDecoder(img_size=img_size,
                                  patch_size=patch_size,
                                #   num_patches=self.encoder.num_patches,
                                  chn_in=chn_in,
                                  num_heads=heads,
                                  depth=decoder_depth,
                                  dim=embed_dim,
                                  mlp_ratio= mlp_ratio,
                                  drop_rate=drop_rate,
                                  attn_drop_rate=att_drop_rate
                                  )
        self.classifier = Classifier(dim=embed_dim, n_classes= n_classes, depth=1, scaling=0.6, flatten=False, pool='mean')
    
    def patchify(self,imgs):
        '''
        Args: 
            imgs (torch.Tensor): Input image(s) of shape [B, C, H, W]
        Returns:
            patchx (torch.Tensor): [Batch, N_Patches, p*p]
        '''
        p = self.encoder.patch_emb.patch_size[0] 
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.chn_in, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.chn_in))
        return x 
    
    def unpatchify(self, x):
        '''
        Converts input from [B,L,p*p] to [b,1,h,w]
        Args:
            patchx: [B, L, p*p]
        Returns:
            imgs: [B,1,H,W]
        '''
        p = int(x.shape[2]**(0.5))
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.chn_in))
        x = torch.einsum('nhwpqc->nchpwq', x)

        return x.reshape(shape=(x.shape[0], self.chn_in, h * p, h * p))
    
    def apply_masking(self, x, mask):
        """
        Generates the patched version of the image for visualization
        ------
        Args:
            - x (torch.Tensor): [B, N, P**2*C]
            - mask (torch.Tensor): [B, N] 0 keep patch, 1 remove patch
        ------
        Returns:
            - x (torch.Tensor): [B, N, P**2*C] with patches that were removed containing only zeros 
        """
        B, N, _ = x.shape
        mask = mask.unsqueeze(-1).expand_as(x)
        # print(f"X Shape {x.shape}")
        # print(f"Mask Shape {mask.shape}")

        return x * (1 - mask)
    
    def restoration_loss(self, x, pred, mask, region:str=None, losstype='MSE', pixel_norm=False):
        """
        Calculate the loss for masked patches based on the masking procedure.
        -----
        Args:
            x (torch.Tensor): Original input image or batch of images.
            pred (torch.Tensor): Predicted reconstruction from the model.
            mask (torch.Tensor): Binary mask indicating masked patches (1 for masked, 0 for unmasked).
            losstype (str): Type of loss to use ('MSE' or 'MAE').
            pixel_norm (bool): Whether to normalize the pixels.
        --------
        Returns:
            torch.Tensor: Computed loss value.
        """
        target = self.patchify(x)
        
        if pixel_norm:                                   # Normalize the pixels
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**0.5

        if losstype == 'MSE': 
            loss = F.mse_loss(pred, target, reduction='none')
        elif losstype == 'MAE':
            loss = F.l1_loss(pred, target, reduction='none')
        else:
            raise ValueError(f"Invalid losstype: {losstype}. Choose 'MSE' or 'MAE'.")
        
        loss = loss.mean(dim=-1)*mask
        loss = loss.sum() / mask.sum()
        
        return loss

    def forward(self, x:torch.Tensor, labels:torch.Tensor=None, lossfunc:str=None, region:str=None, task:str=None):
        """
        -----
        Args:
            - x (torch.Tensor):
            - nodule_masks (torch.Tensor):
            - labels (torch.Tensor):
            - region (str):
            - task (str):
        --------
        Returns:
            - loss (torch.Tensor): Loss value. 
            - y (torch.Tensor): Predicted classification. (For 'Dx' task)
        """

        if task !='dx':
            latent, mask, restore_ids = self.encoder(x, region=region, task=task, maskratio=self.mask_ratio)
            pred = self.decoder(latent, restore_ids)
            loss = self.restoration_loss(x, pred, mask, region=region, losstype=lossfunc, pixel_norm=False)
            return loss
        
        else:
            latent = self.encoder(x)
            y = self.classifier(latent[:,1:,:])
            loss = F.binary_cross_entropy(y, labels)
            return y, loss

    def inference(self, x:torch.Tensor, task:str=None, region:str=None):
        """
        Returns the Classification of the given image using the ViT Encoder + Classifier
        Args:
            - x (torch.Tensor): 
        """
        if task != 'dx':
            latent, mask, restore_ids = self.encoder(x, region=region, task=task, maskratio=self.mask_ratio)
            rec = self.decoder(latent, restore_ids)
            rec = self.unpatchify(rec)

            mask_img = self.patchify(x)
            mask_img = self.apply_masking(mask_img, mask)
            mask_img = self.unpatchify(mask_img)
            
            return {'reconstruction': rec,
                    'masked_images': mask_img
                    }
        else:
            latent = self.encoder(x)
            y_hat = self.classifier(latent)
            return {"classification": y_hat}
        