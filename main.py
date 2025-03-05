# coding: utf-8 
'''
""" MIT License """
    Project: PulmonaryMAE
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''

# ----------------Libaries--------------- #
from typing import Any, Dict, List
from argparse import Namespace

import pandas as pd
import torch
import argparse
import logging
import yaml
import sys
import os

from lib.trainer import mae_trainer, vit_trainer
from lib.data import datasets
from lib.utils import progress
from lib.utils import metrics
from lib.utils import utils
# ---------------------------------------------------------------------------- #
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logger = logging.getLogger()

"""
============== TODO LIST ==============
[] 3. Add post-hoc analysis of data, such that radiomic markers can assist in telling us impact of training on learned characteristics
"""

def get_trial(net, tlearn, cfg, region, device, bar):
    if tlearn == 'self' or tlearn=='cooc':
        trial = mae_trainer.PytorchTrials(
                            model=net,
                            tlearn=tlearn,
                            cfg= cfg,
                            region=region,
                            device = device,
                            progressbar=bar)
    else:
        trial = vit_trainer.PytorchTrials(
                            model=net,
                            tlearn=tlearn,
                            cfg= cfg,
                            device=device,
                            progressbar=bar)
    return trial

def experiment(COPDdata, NLSTdata, region:str, tlearn:str=None, cfg:dict=None, device:int=None):
    """
    Function controlling all experiment component of defined by the parsed function call. The experiment function does not return anything to the main function.
    At the place, it will save all results for the given methodology to its respective result folder(s). See github repo to ensure proper directories exists.
    \n-----------\n
    Parameters:\n
        1. dataset (pd.DataFrame): contains the pandas dataframe with the file path information, classification (ca), segmentation mask location, and radiomic information concatenated on PID.\n
        2. region (str): string defining which type of embedding the network will utilize. Deep-radiomics, Concept-Rads, and Guideline-Rads
        3. tlearn (str):Defines the style of learning the region bottleneck will utilize if the region is not defined as Deep-Rads
        4. cfg (dict): Dictionary containing all experiment parameters, optimizer parameters, training data informaiton, and savepaths
        5. device (?): Defines which GPU to train model on. 
    """
    
    bar =progress.ProgressBar(model= cfg['project'],
                              tlearn = tlearn,
                              method= region,
                              maxpoch = cfg['experiment']['rec_epchs'],
                              bar_length= 10)

    eval_df = pd.DataFrame()

    for k in range(cfg['experiment']['folds']): 
        net = utils.select_model(cfg=cfg, tlearn=tlearn, region=region)
        trial = get_trial(net, tlearn, cfg, region, device, bar)

        if tlearn == 'self' or tlearn=='cooc':     # Pre-training Strategy using COPDGene Data
            bar.reset(cfg['experiment']['rec_epchs'])
            if tlearn=='cooc':
                trainset, valset, testset = datasets.create_datasets(COPDdata, cohort='COPD',  cfg=cfg, seed=k)
            else: 
                trainset, valset, testset = datasets.create_datasets(NLSTdata, cohort='NLST', cfg=cfg, seed=k)

            trainloader = torch.utils.data.DataLoader(trainset, batch_size= cfg['experiment']['batchsize'], shuffle= True)
            valloader = torch.utils.data.DataLoader(valset, batch_size = cfg['experiment']['batchsize'], shuffle= True)
            testloader = torch.utils.data.DataLoader(testset, batch_size= cfg['experiment']['batchsize'], shuffle= True)
            
            trial.training(trainloader, valloader, task='reconstruction')
            trial._savemodel_()
            trial.tracker.plots()
            mae_performance = trial.evaluate(testloader, task='reconstruction')  
            
            # Training Strategy using NLST Data
            bar.reset(cfg['experiment']['dx_epchs'])
            trainset, valset, testset = datasets.create_datasets(NLSTdata, cohort='NLST', cfg=cfg, seed=k)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= cfg['experiment']['batchsize'], shuffle= True)
            valloader = torch.utils.data.DataLoader(valset, batch_size = cfg['experiment']['batchsize'], shuffle= True)
            testloader = torch.utils.data.DataLoader(testset, batch_size= cfg['experiment']['batchsize'], shuffle= True)
            
            trial.training(trainloader, valloader, task='dx')
            trial._savemodel_()
            trial.tracker.plots()
            dx_performance = trial.evaluate(testloader, task='dx')

            fold_metrics = pd.DataFrame([{**mae_performance, **dx_performance}])
            eval_df = pd.concat([eval_df, fold_metrics], ignore_index=True)
            
        else:
            bar.reset(cfg['experiment']['dx_epchs'])
            if tlearn=='scratch':
                trainset, valset, testset = datasets.create_datasets(NLSTdata, cohort='NLST', cfg=cfg, seed=k)
            else:
                trainset, valset, testset = datasets.create_datasets(NLSTdata, cohort='224xNLST', cfg=cfg, seed=k)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= cfg['experiment']['batchsize'], shuffle= True)
            valloader = torch.utils.data.DataLoader(valset, batch_size = cfg['experiment']['batchsize'], shuffle= True)
            testloader = torch.utils.data.DataLoader(testset, batch_size= cfg['experiment']['batchsize'], shuffle= True)

            trial.training(trainloader, valloader, task='dx')
            trial._savemodel_()
            trial.tracker.plots()

            fold_metrics = trial.evaluate(testloader, task='dx')  
            fold_metrics = pd.DataFrame([fold_metrics])
            eval_df = pd.concat([eval_df, fold_metrics], ignore_index=True)
    
    metrics.save_df(eval_df, cfg, tlearn=tlearn, subfolder=region)

def inference(data, region:str, cfg:dict, tlearn:str=None):
    """
    Infernce Evaluation of a Loaded Model
    -----------
    Parameters:
        dataset (pd.DataFrame):
            contains the pandas dataframe with the file path information, classification (ca), segmentation mask location, and radiomic information concatenated on PID.
        region (str):
            string defining which type of embedding the network will utilize. Deep-radiomics, Concept-Rads, and Guideline-Rads
        cfg (dict)
            Dictionary containing all experiment parameters, optimizer parameters, training data informaiton, and savepaths
        tlearn (str):
            Defines the style of learning the region bottleneck will utilize if the region is not defined as Dee
    """
    pass

def main(cfg, command_line_args):
    """
    Main function that initialize the planned experiments
    -----------
    Parameters:
    cfg - dictionary
        dictionary generated from yaml file located in cfg folder
    commmand_line_args:
        command line arguments provided by user
    """
    cfg['device'] = utils.GPU_init(loc=cfg['device_id'])
    
    utils.check_directories(cfg)
    
    COPDdata = datasets.load_files(cfg, filepath=cfg['training_data']['COPD'],
                                   ext=cfg['training_data']['filetype'],
                                   cohort='COPD',
                                   seed = cfg['seed'])
    
    NLSTdata = datasets.load_files(cfg, filepath=cfg['training_data']['NLST'],
                                   ext=cfg['training_data']['filetype'],
                                   cohort='NLST',
                                   resample= 'downsample',
                                   seed = cfg['seed'])
    
    if cfg['mode'] == "Training":
        for tlearn in cfg['learn']:
            if tlearn != 'self' and tlearn != 'cooc':
                experiment(COPDdata, NLSTdata, region=None, tlearn=tlearn, cfg=cfg, device=cfg['device'])
            else:
                for region in cfg['experiment']['regions']:
                    experiment(COPDdata, NLSTdata, region, tlearn, cfg, cfg['device'])
    else:
        for tlearn in cfg['learn']:
            if tlearn != 'MAE':
                inference(COPDdata, NLSTdata, net, region=None, tlearn=tlearn, cfg=cfg, device=cfg['device'])
            else:
                for region in cfg['experiment']['regions']:
                    net = utils.select_model(cfg=cfg, region=region, tlearn=tlearn)
                    inference(COPDdata, NLSTdata, net, region, tlearn, cfg, cfg['device'])

def build_yaml() -> argparse.ArgumentParser:
    """
    -----------
    Parameters:
    --------
    Returns:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to Yaml file for training/inference"
    ) 

    return parser


def build_config(args):
    """
    -----------
    Parameters:
    --------
    Returns:
    """
    with open(os.getcwd() + args.config, "r") as yamlfile:
        try:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            return data

        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = build_yaml()
    cfg = build_config(parser.parse_args())

    main(cfg, command_line_args=sys.argv)

