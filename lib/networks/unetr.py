# coding: utf-8 
'''
    Project: RegionalMAE
    Authors:
        1. Axel Masquelin
    Description
    Copyright (c) 2022
'''

# --------- Libraries ---------- #
from typing import Sequence, Tuple, Union
from torch import Tensor

from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from embedding import PatchEmbed3d, PatchEmbed2d

import torch.nn as nn
import torch
# -------------------------------#
def projection(x, hidden_size, grid_size):
    prj_view = (x.size(0), *grid_size, hidden_size)
    x = x.view(prj_view)
    prj_axes = (0,len(x.shape)-1) + tuple(d+1 for d in range(len(grid_size)))
    x = x.permute(prj_axes).contiguous()

    return x
class encoder(nn.Module):
    def __init__(self,
                input_size:int,
                patch_size:int,
                chn_in:int,
                chn_out:int,
                embed_dim:int,
                head_in:int,
                embed_layer:int,
                spatial_dims:int = 3,
                learnabel_emb:bool = True,
                hidden_states:bool = True,
                norm_name: Union[Tuple,str]="instance",
                conv_block:bool = True,
                res_block:bool = True,
                ):
        super().__init__()
        self.enc1 = UnetrBasicBlock(
                        spatial_dims = spatial_dims,
                        in_channels=chn_in,
                        out_channels=chn_out,
                        kernel_size=3,
                        stride=1,
                        norm_name=norm_name,
                        res_block=res_block,
                        )
        self.enc2 = UnetrPrUpBlock(
                        spatial_dims = spatial_dims,
                        in_channels = chn_out,
                        out_channels = embed_dim* 2,
                        num_layer = 2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,)
        self.enc3 = UnetrPrUpBlock(
                        spatial_dims = spatial_dims,
                        in_channels = embed_dim*2,
                        out_channels = embed_dim * 4,
                        num_layer = 2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
        )
        self.enc4 = UnetrPrUpBlock(
                        spatial_dims = spatial_dims,
                        in_channels = embed_dim * 4,
                        out_channels = embed_dim * 8,
                        num_layer = 2,
                        kernel_size=3,
                        stride=1,
                        upsample_kernel=2,
                        norm_name=norm_name,
                        conv_block=conv_block,
                        res_block=res_block,
        )

        self.encoderlist = [self.enc1,self.enc2,self.enc3,self.enc4]
    def forward(self, x:Tensor):
        x_outs = [enc(x) for enc in self.encoderlist]       

        return x_outs
        
class decoder(nn.Module):
    def __init__(self,
                 feature_size:int,
                 chn_out:int,
                 embed_dim:int,
                 spatial_dim:int,
                 conv_block:bool = True,
                 res_block: bool = True,
                 norm_name: Union[Tuple,str] = 'instance'
                 ):
        """
        Args:
            input_size
            patch_size:
            chn_in:
            chn_out:
            feature_size:
            embed_dim: size of embedding_space
            spatial_dim: number of spatial dimensions
        """
        super().__init__()
        self.dec4 = UnetrUpBlock(
            spatial_dim=spatial_dim,
            in_channels= embed_dim,
            out_channels= feature_size*8,
            kernel_size= 3,
            upsample_kernel_size= 2,
            norm_name= norm_name,
            conv_block= conv_block,
            res_block= res_block
            )
        self.dec3 = UnetrUpBlock(
            spatial_dim=spatial_dim,
            in_channels= feature_size*8,
            out_channels= feature_size*4,
            kernel_size= 3,
            upsample_kernel_size= 2,
            norm_name= norm_name,
            res_block= res_block
            )
        self.dec2 = UnetrUpBlock(
            spatial_dim=spatial_dim,
            in_channels= feature_size*4,
            out_channels= feature_size*2,
            kernel_size= 3,
            upsample_kernel_size= 2,
            norm_name= norm_name,
            res_block= res_block
            )
        self.dec1 = UnetrUpBlock(
            spatial_dim= spatial_dim,
            in_channels= feature_size*2,
            out_channels= feature_size,
            kernel_size= 3,
            upsample_kernel_size= 2,
            norm_name= norm_name,
            res_block= res_block
            )
        
        self.outconv = UnetOutBlock(
            spatial_dim=spatial_dim,
            in_channels = feature_size,
            out_channels=chn_out
            )
    
    def forward(self, x_feat, x:Tensor):
        y1 = self.dec4(x_feat, x[-1])
        y2 = self.dec3(x[-2], y1)
        y3 = self.dec2(x[-3], y2)
        y4 = self.dec1(x[-4], y3)
        return y4

    
class unetrND(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size, embed_layer = self.get_dimensionality(args)
        
        self.encoder = encoder(input_size = input_size,
                               patch_size = args.patch_size,
                               chn_in = args.chn_in,
                               chn_out = args.chn_out,
                               embed_dim = args.embed_dim,
                               head_in = args.enncoder_heads,
                               embed_layer = embed_layer,
                               learnabel_emb = True,
                               hidden_states = True,
                               )
        

        self.decoder = decoder(input_size= input_size,
                               patch_size= args.patch_size,
                               chn_in= args.chn_in,
                               chn_out= args.chn_out,
                               feature_size= args.features,
                               embed_dim= args.embed_dim,
                               spatial_dim= args.spatial_dim,
                               )
        
        if self.args.pretrain:
            self.apply(self.get_weights(flag=True))
        else:
            self.apply(self.get_weights)

    def get_dimensionality(self,  args):
        """
        """
        if args.input_dim == 3:
            embed_layer = PatchEmbed3d
            input_size = (args.xdim, args.ydim, args.zdim)
        elif args.input_dim == 2:
            input_size = (args.xdim, args.ydim)
            embed_layer = PatchEmbed2d
        assert len(args.input_dims) == len(args.patch_dim), f'Input dimension {len(args.input_dims)} != patch_dim'
        return (input_size, embed_layer)

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    def get_weights(m, flag=False):
        """
        Allows for UNETR random weight initialization, or pretraining
        """
        if flag:
            pass
        else:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x:Tensor) -> Tensor:
        """
        x - image tensor (Needs to be converted to Patches)
        """
        x = self.encoder(x)
        x_feats = projection(x[-1])
        patchx = self.decoder(x_feats, x)

        return patchx

