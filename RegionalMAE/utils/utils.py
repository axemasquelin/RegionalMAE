# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
from RegionalMAE.networks import MaskedViT
from RegionalMAE.networks import VitBackbone
import torch.nn as nn
import pandas as pd
import torch
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

def init_weights(m):
    '''
    Initializes Model Weights using Uniform Distribution 
    TODO: Either include a load_weights function or add option to initialize weights based on a provided state_dict. 
    '''
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def freeze_weights(model):
    """
    """

    for param in model.encoder.parameters():
         param.requires_grad = False
    
    return model

def transfer_weights(config:dict, maskratio:float, model:list, task:str):
    """
    """
    if task == 'Dx':
        tail = select_model(config, maskratio=maskratio, func='Dx')
    
    if task == 'Segment':
        tail = select_model(config, maskratio=maskratio, func='Segment')

    model.decoder = tail

    net = freeze_weights(model)

    return model

def select_model(config:dict, region:str=None, tlearn:bool=False, func:str=None):
    """
    Loads Model Architecture based on the user selected model provided by the configuration yaml file
    -----------
    Parameters:
    model - string
        string defining which model to load from RegionalMAE/networks
    --------
    Returns:
    net - nn.Module()
        Initialized neural network model using pytorch library
    """

    if tlearn == 'MAE':
        net = MaskedViT.RegionMAE(region=region)

    if tlearn == 'scratch':
        net = VitBackbone.Custom_ViT_B_16(config, pretrain=None)
    if tlearn == 'pre':
        net = VitBackbone.Custom_ViT_B_16(config, pretrain=True)

    return net.to(config['device'])


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

def csv_save(data:pd.DataFrame,
             savedir:str,
             name:str):
    ''' Save AUCs scores to a csv file '''

    pth_to_save = savedir + name + '.csv'
    data.to_csv(pth_to_save)

def create_directory(savedirectory):
    """
    -----------
    Parameters:
    """
    dir_flag = os.path.exists(savedirectory)
    if not dir_flag:
        sys.stdout.write('\n\r {0} | Creating {1} | {0}\n '.format('-'*30, savedirectory))
        os.mkdir(savedirectory)

def check_directories(config):
    """
    Parameters:
    -----------
    """
    sys.stdout.write('\n\r {0} | Checking for Result Directories | {1}\n '.format('-'*30, '-'*30))
    savepath = os.getcwd() + config['savepath']

    create_directory(savepath)
    for task in config['tasks']:
        create_directory(savepath + task + '/')
        for learn in config['learn']:
            create_directory(savepath + task + '/' + learn + '/')
            if learn == 'MAE':
                for region in config['experiment_params']['regions']:
                    create_directory(savepath + task + '/' + learn + '/' + region +'/')



