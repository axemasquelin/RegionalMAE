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

from RegionalMAE import eval
from RegionalMAE.data import dataloader
from RegionalMAE.utils import progress
from RegionalMAE.utils import metrics
from RegionalMAE.utils import utils
from bin import builder
# ---------------------------------------------------------------------------- #

def experiment(dataset:pd.DataFrame,  tlearn:str, config:dict, region:str=None):
    '''
    Main function for Experiment - Will evaluate Self-trained MAE ViT-B-16 / Pretrained ViT-B-16 / Scratch ViT-B-16
    -----------
    Parameters:
        dataset: (pd.DataFrame) Contains all files location, classification, and segmentation map locations
        tlearn: (str) Describe which type of learning model will be utilizing (MAE/pre/scratch)
        config: (dic) dictionary containing all experiment variables
        region: (str) defines the region that will be masked when evaluating the MAE model
    '''
    
    bar =progress.ProgressBar(model= config['project'],
                              method = tlearn,
                              region=region,
                              maxfold= config['experiment_params']['folds'],
                              bar_length= 50)

    sensitivity = np.zeros((config['experiment_params']['folds']))
    specificity = np.zeros((config['experiment_params']['folds']))
    youdens = np.zeros((config['experiment_params']['folds']))
    precision = np.zeros((config['experiment_params']['folds']))
    dor = np.zeros((config['experiment_params']['folds']))
    aucs = np.zeros((config['experiment_params']['folds']))
    
    tprs, fprs = [], []
    for k in range(config['experiment_params']['folds']):
        
        df_train, df_test = train_test_split(dataset, test_size = config['training_data']['split'][2], random_state = k)
        df_train, df_val = train_test_split(df_train, test_size = config['training_data']['split'][1], random_state = k)
    
        df_train = dataloader.augment_dataframe(df_train, upsample=2, augment='rand')
        df_val = dataloader.augment_dataframe(df_val, upsample=2, augment='rand')
        df_test = dataloader.augment_dataframe(df_test, upsample=1, augment='infer')

        # Load Training Dataset
        trainset =  dataloader.Nrrdloader(df_train, norms=config['training_data']['inputnorm'])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size= 125, shuffle= True)

        # Load Validation Dataset
        valset = dataloader.Nrrdloader(df_val, norms=config['training_data']['inputnorm'])
        valloader = torch.utils.data.DataLoader(valset, batch_size = 125, shuffle= True)

        # Load testing Dataset
        testset = dataloader.Nrrdloader(df_test, norms=config['training_data']['inputnorm'])
        testloader = torch.utils.data.DataLoader(testset, batch_size= 125, shuffle= True)
        for task in config['tasks']:
            if tlearn == 'MAE':
                savepath = os.getcwd() + config['savepath'] + task + '/' + tlearn + '/' + region + '/'

                bar._update(task= 'Reconstruct', tlearn= tlearn, fold= k)
                MAEmodel = utils.select_model(config=config, region=region, tlearn=tlearn, func='MAE')
                trial = eval.PyTorchTrials(fold=k, model=MAEmodel, tlearn = tlearn, task='Reconstruct', region=region, config=config,  device=config['device'], progressbar=bar)
                trial.training(train_loader=trainloader, validation_loader=valloader)
                trial.savemodel()
                trial.performancelog() 
                performance = trial.inference(test_loader=testloader) 
            
                bar._update(task= 'Dx', tlearn= tlearn, fold= k)
                bestMAE = trial.model       
                model = utils.transfer_weights(config=config, model=bestMAE, task='Dx')
                trial = eval.PyTorchTrials(fold=k, model=model, tlearn = tlearn, task='Dx', config=config, device=config['device'], progressbar=bar)
                trial.training(train_loader=trainloader, validation_loader=valloader)
                trial.savemodel()
                trial.createlog() 
                performance = trial.inference(test_loader=testloader) 
        
            else:
                savepath = os.getcwd() + config['savepath'] + task + '/' + tlearn + '/'
                bar._update(task= 'Dx', tlearn= tlearn, fold= k)       
                model = utils.select_model(config=config, region=None, tlearn=tlearn, func='Dx')
                trial = eval.PyTorchTrials(fold=k, model=model, tlearn = tlearn, task='Dx', region=region, config=config, device=config['device'], progressbar=bar)
                trial.training(train_loader=trainloader, validation_loader=valloader)
                trial.savemodel()
                trial.createlog() 
                performance = trial.inference(test_loader=testloader)
        
        sensitivity[k] = performance['sensitivity']
        specificity[k] = performance['specificity']
        youdens[k] = performance['youden_index']
        precision[k] = performance['precision']
        dor[k] = performance['diagnostic_ratio']
        fprs.append(performance['fps'])
        tprs.append(performance['tps'])
            

    aucs[:] = metrics.calcAuc(fprs,
                              tprs,
                              region,
                              plot_roc=True,
                              savepath=savepath)
    

    performance_df = pd.DataFrame({
        'sensitivity':sensitivity,
        'specificity':specificity,
        'youdens_index':youdens,
        'precision':precision,
        'DiagnosticOddsRatio':dor,
        'aucs':aucs
        })
    
    
    utils.csv_save(performance_df, savedir=savepath, name='metrics')

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

    utils.check_directories(config)
    sys.stdout.write('\n\r {0}\n Loading User data from: {1}\n {0}\n '.format('='*(24 + len(config['training_data']['datadir'])), config['training_data']['datadir']))
    
    dataset = dataloader.load_files(config)

    for tlearn in config['learn']:
        if tlearn == 'MAE':
            for region in config['experiment_params']['regions']:
                experiment(dataset, tlearn=tlearn, config=config, region=region)
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