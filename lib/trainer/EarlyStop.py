""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from lib.utils.metrics import reconstructions
from lib.networks.checkpoints import checkpoint
from lib.networks.optimizers import *
from lib.utils.tracker import tracker

from sklearn.metrics import roc_curve, confusion_matrix, auc
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

import numpy as np
import random
import sys, os
import time
# --------------------------------------------

class EarlyStop:
    def __init__(self, patience:int=5, delta:float=0.001, monitor:str='lae'):
        """
        Args:
            patience (int): Number of Epochs to wait after last improvement
            delta (float): Minimum Change to qualify as improvement
            monitor (str): Key for validation loss to monitor
        """
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.best_loss = None
        self.counter=0
        self.stopping = False

    def __call__(self, loss, epoch):
        """
        Updates states of stopping for early stop based on the monitored loss over time
        #TODO Implement Early stop to monitor two metrics of interest or alternatively set independent downstream tasks to learn=False
        """
        if self.best_loss is None:
            self.best_loss = loss[self.monitor][epoch]
        
        elif loss[self.monitor][epoch] <= self.best_loss - self.delta:
            self.best_loss = loss[self.monitor][epoch]
            self.counter=0
        
        else:
            self.counter+=1
            if self.counter >= self.patience:
                self.stopping=True
        