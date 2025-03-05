# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
from lib.networks import MaskedViTv2
from lib.networks import VitBackbone
import torch.nn as nn
import pandas as pd
import torch
import timm
import sys
import os
# ---------------------------------------------------------------------------- #

def GPU_init(loc):
    """
    Definition: GPU Initialization function
    Inputs: loc - 0 or 1 depending on which GPU is being utilized
    Outputs: check_gpu - gpu enabled variable
    """
    check_gpu = torch.device("cuda:" + str(loc) if torch.cuda.is_available() else "cpu")
    print("Available Device: " + str(check_gpu))
    
    return check_gpu

def select_model(cfg:dict, tlearn:str=None, region:str=None,):
    """
    Initialize model
    -----
    Args:
        cfg (dict): dictionary containing string to load a set of models
    Returns:
        net:
    """
    available_models = ['self', 'cooc', 'pre', 'scratch']
    assert tlearn in available_models, f"'{tlearn}' is not an available methodology, skipping normalization"
    
    if tlearn == 'cooc' or tlearn == 'self':
        net = MaskedViTv2.CustomMAE(img_size= cfg['model']['input_size'],
                            patch_size= cfg['model']['patch_size'],
                            chn_in= cfg['model']['chn_in'],
                            heads= cfg['model']['heads'],
                            encoder_depth= cfg['model']['enc_depth'],
                            decoder_depth= cfg['model']['dec_depth'],
                            embed_dim= cfg['model']['embed_dim'],
                            mlp_ratio=cfg['model']['mlp_ratio'],
                            drop_rate= cfg['model']['drop_rate'],
                            att_drop_rate= cfg['model']['att_droprate'],
                            n_classes= cfg['model']['n_classes'],
                            mask_ratio= cfg['model']['mask_ratio'],
                            tlearn=tlearn,
                            region=region
                            )

    else:
        if tlearn == 'pre':
            net = VitBackbone.Custom_ViT_B_16(pretrain=True)
            # net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        else:
            # net = VitBackbone.Custom_ViT_B_16(cfg, pretrain=None)
            net = VitBackbone.Custom_ViT_B_16(custom=True, pretrain=False)


    return net.to(cfg['device'])
    
def init_weights(m):
    '''
    Initializes Model Weights using Uniform Distribution 
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def freeze_weights(model):
    for param in model.encoder.parameters():
         param.requires_grad = False
    
    return model

def transfer_weights(config:dict, maskratio:float, model:list, task:str):
    if task == 'Dx':
        tail = select_model(config, maskratio=maskratio, func='Dx')
    
    if task == 'Segment':
        tail = select_model(config, maskratio=maskratio, func='Segment')

    model.decoder = tail

    net = freeze_weights(model)

    return model

def check_parameters(netlist, params=None):
    """
    Parameters:
    -----------
    """
    sys.stdout.write('\n {0}| Number of Parameters in Networks |{0}'.format('-'*6))
    for i, net in enumerate(netlist):
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        sys.stdout.write("\n {0} Number of Parameters: {1}".format(params['model'][i], pytorch_total_params))

    sys.stdout.write('\n {0}'.format('-'*48))

def check_directories(config):
    """
    -----
    Args:
    """
    savepath = os.getcwd() + config['savepath']
    create_directories(savepath)
    for learn in config['learn']:
        if learn == 'cooc' or learn == 'self':
            create_directories(savepath + learn + '/')
            for region in config['experiment']['regions']:
                create_directories(savepath + learn + '/' + region +'/')
        else:
            create_directories(savepath + learn + '/')

def create_directories(folderpath):
    """
    -----
    Args:
    """
    # sys.stdout.write('\n\r {0} | Checking for Result Directories | {1}\n '.format('-'*25, '-'*25))
    # print(folderpath) 
    dir_exists = os.path.exists(folderpath)
    if not dir_exists:
        sys.stdout.write('\n\r {0} | Creating {1} Directories | {0}\n '.format('-'*10, folderpath ))
        os.mkdir(folderpath)