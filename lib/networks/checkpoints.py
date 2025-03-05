""" MIT License """
'''
    Project: RadiomicConcept
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os, sys
# --------------------------------------------

class checkpoint():
    """
    Checkpoint class used to save the state dictionaries needed to load model for inference or finetuning
    """
    def __init__(self, tlearn, region=None, savepath='.'):
        self.logs = OrderedDict()
        self.tlearn = tlearn
        self.region = str(region)
        
        if tlearn=='cooc' or tlearn == 'self':
            self.savepath = savepath + tlearn + '/' + region + '/' + region
        else:
            self.savepath = savepath + tlearn + '/' + tlearn 
        print(self.savepath)
    def update(self, data):
        """
        Updates the checkpoint with new data.
        -----
        Args:
            - data (dict): A dictionary containing the new metric values to be added.
                           The keys should correspond to the metric names.
        """
        for key, group in data.items():
            if isinstance(group, dict):
                if key not in self.logs:
                    self.add(key, is_dict=True)    

                for loss, value in group.items():
                    if loss not in self.logs[key]:
                        self.add(loss, parent_dict=self.logs[key])

                    if isinstance(value,torch.Tensor):
                        self.logs[key][loss]= value.detach().cpu().item()
                    else:
                        self.logs[key][loss]= value
                
            else:
                if key not in self.logs:
                    self.add(key)

                if isinstance(group, torch.Tensor):
                    self.logs[key]= group.detach().cpu().item()
                else:
                    self.logs[key] = group

    def add(self, key, is_dict=False, parent_dict=None):
        """
        Adds a new key to the checkpoint
        -----
        Args:
            - key (str): The name of the metric to be added.
            - values: 
                                If True, the metric is initialized as an empty list
                                to store tensors. Otherwise, it's initialized as 0.
        """
        if parent_dict is None:
            parent_dict = self.logs

        assert key not in parent_dict, "Key present in parent dict"
        if is_dict:
            parent_dict[key] = OrderedDict()
        
        else:
            parent_dict[key]= 0
    
    def get(self, key):
        return self.logs[key]
    
    def save(self):
        sys.stdout.write('\n\r {0} | Saving Network - Epoch: {1} | {2}\n'.format('-'*10,
                                                                                self.logs['epoch'],
                                                                                # self.logs['acc'],
                                                                                '-'*10))
        
        torch.save(self.logs, os.getcwd() + self.savepath + '_bestperformance.pt')

    def load(self):
        return torch.load(os.getcwd() + self.savepath + '_bestperformance.pt')