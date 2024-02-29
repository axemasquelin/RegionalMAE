# coding: utf-8 
'''
""" MIT License """
    Project: PulmonaryMAE
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''

# ----------------Libaries--------------- #
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import argparse
import logging
import sys
import os

from PulmonaryMAE import eval
from PulmonaryMAE.data import dataloader
from PulmonaryMAE.utils import progress
from PulmonaryMAE.utils import utils
from bin import builder
# ---------------------------------------------------------------------------- #

"""
TODO:

"""

def experiment(dataset:pd.DataFrame,  tlearn:str, config:dict, maskratio:float=None, region:str=None):
    """
    """
    bar =progress.ProgressBar(model= config['project'],
                              maxfold= config['experiment_params']['folds'],
                              maskratio = maskratio,
                              bar_length= 50)

    loss_training = np.zeros(config['experiment_params']['folds'])
    loss_validation = np.zeros(config['experiment_params']['folds'])
    Acc_training = np.zeros(config['experiment_params']['folds'])
    Acc_validation = np.zeros(config['experiment_params']['folds'])

    for k in range(config['experiment_params']['folds']):
        
        df_train, df_test = train_test_split(dataset, test_size = config['training_data']['split'][2], random_state = k)
        df_train, df_val = train_test_split(df_train, test_size = config['training_data']['split'][1], random_state = k)
        
        # Load Training Dataset
        trainset =  dataloader.Nrrdloader(df_train, norms=config['training_data']['inputnorm'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size= 125, shuffle= True)

        # Load Validation Dataset
        valset = dataloader.Nrrdloader(df_val, norms=config['training_data']['inputnorm'])
        valloader = torch.utils.data.DataLoader(valset, batch_size = 125, shuffle= True)

        # Load testing Dataset
        testset = dataloader.Nrrdloader(df_test, norms=config['training_data']['inputnorm'])
        testloader = torch.utils.data.DataLoader(testset, batch_size= 125, shuffle= True)
        
        if tlearn == 'MAE':
            bar._update(task= 'Reconstruct', tlearn= tlearn, fold= k)
            if region != None:
                MAEmodel = utils.select_model(config=config, region=region, func='MAE')
            else:
                MAEmodel = utils.select_model(config=config, maskratio=maskratio, func='MAE')

            trial = eval.PyTorchTrials(fold=k, model=MAEmodel, tlearn = tlearn, task='Reconstruct', config=config, maskratio=maskratio, device=config['device'], progressbar=bar)
            trial.training(train_loader=trainloader, validation_loader=valloader)
            trial.savemodel()
            performance = trial._getmetrics_()
            trial.inference(test_loader=testloader) 

        for task in config['tasks']:        
            if tlearn != 'MAE':
                bar._update(task= task, tlearn= tlearn, fold= k)       
                model = utils.select_model(config=config, maskratio=maskratio, pretraining=tlearn, func=task)
                trial = eval.PyTorchTrials(fold=k, model=model, tlearn = tlearn, task=task, config=config, maskratio=0.0, device=config['device'], progressbar=bar)
                trial.training(train_loader=trainloader, validation_loader=valloader)
                performance = trial._getmetrics_()
                trial.inference(test_loader=testloader)
            
            else:
                bar._update(task= task, tlearn= tlearn, fold= k)
                bestMAE = trial.model       
                model = utils.transfer_weights(config=config, maskratio=maskratio, model=bestMAE, task=task)
                trial = eval.PyTorchTrials(fold=k, model=model, tlearn = tlearn, task=task, config=config, maskratio=maskratio, device=config['device'], progressbar=bar)
                trial.training(train_loader=trainloader, validation_loader=valloader)
                trial.savemodel()
                performance = trial._getmetrics_()
                trial.inference(test_loader=testloader) 
            
def masking_experiments(config, dataset, tlearn, maskratios):
    '''
    '''
    for masktype in config['experiment_params']['masktypes']:
        if masktype != 'Regional':
            for maskratio in maskratios:
                experiment(dataset, maskratio, tlearn, config)
        else:
            for region in config['experiment_params']['regions']:
                experiment(dataset, tlearn=tlearn, config=config, region=region)


def main(config, command_line_args):
    """
    Controls the experiment based on the user defined mode:
        (1) Training; trains a model defined in PulmonaryMAE/networks/ from scratch
        (2) Finetune: finetune a model defined in PulmonaryMAE/networks/ using user defined dataset
        (3) Inference: Loads model_checkpoint of a defined model and evaluates it on the defined dataset
    -----------
    Parameters:
        (1) config - dictionary
            dictionary containing the necessary information to initialize optimizer, loss functions, dataloader, along with the expected filetype and dataset location
    """

    config['device'] = utils.GPU_init(config['device'])
    maskratios = utils.maskratio_array(config['experiment_params'])
    utils.check_directories(config, maskratios=maskratios)
    sys.stdout.write('\n\r {0}\n Loading User data from: {1}\n {0}\n '.format('='*(24 + len(config['training_data']['datadir'])), config['training_data']['datadir']))
    
    dataset = dataloader.load_files(config)

    for tlearn in config['learn']:
        if tlearn == 'MAE':
            masking_experiments(config=config, dataset=dataset, tlearn=tlearn, maskratios=maskratios)
        else:
            experiment(dataset, tlearn=tlearn, config=config)

if __name__ == '__main__':
    """
    Initialization of the experiment, Searchers for yaml file that is provided by user to run experiments in:
        (1) Training Mode: Trains a defined model from scratch using the provided data
        (2) Finetune Mode: Loads a model_state_dictionary for a defined model in networks and finetunes on provided dataset.
        (3) Inference Mode: Loads a model_state_dictionary and evaluates the defined model on the provided dataset
    """
    logger = logging.getLogger()

    logging.basicConfig(level=logging.INFO)
    parser = builder.build_yaml()
    config = builder.build_config(parser.parse_args(), logger=logger)

    main(config, command_line_args=sys.argv)