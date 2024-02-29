# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
# from RegionalMAE.networks import MaskedCNN as cnns
from RegionalMAE.networks import MaskedViT
from RegionalMAE.networks import VitBackbone
import pandas as pd
import numpy as np
import torch
import sys
import cv2
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

def select_model(config:dict, region:str=None, pretrain:bool=False, func:str=None):
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

    if pretrain == 'MAE':
        net = MaskedViT(region=region,
                            embeddim= config['experiment_params']['embeddim'],
                            n_classes= len(config['experiment_params']['classlabels']))

    if pretrain == 'scratch':
        # if func == 'Segment':
        #     net = vits.load_UNETR(config, weights=None)
        # else: 
        net = VitBackbone.load_ViTB16(config, weights=None)
    if pretrain == 'pre':
        # if func == 'Segment':
        #     net = vits.load_UNETR(config, weights=True)
        # else:
        net = VitBackbone.load_ViTB16(config, weights=True)

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

def csv_save(ms, data, name = ''):
    ''' Save AUCs scores to a csv file '''

    cols = [name +str(i+1) for i in range(data.shape[1])]
    logs = pd.DataFrame(data, columns=cols)    

    pth_to_save = os.getcwd() + "/results/" + str(ms) + 'x/' + name + '.csv'
    logs.to_csv(pth_to_save)

    # print(logs)

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
        create_directory(savepath + str(task) + '/')

        for learn in config['learn']:
            create_directory(savepath + str(task) + '/' + str(learn) + '/')

            if learn == 'MAE':
                for region in config['experiment_params']['regions']:
                    create_directory(savepath + str(task) + '/' + str(learn) + '/' + str(region) +'/')



