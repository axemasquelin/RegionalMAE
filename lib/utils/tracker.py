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

class tracker():
    """
    Tracks and accumulates metrics during training.
    -----------
    Attributes:
        tracked (OrderedDict): An ordered dictionary to store the tracked metrics.
        epoch (list): A list to store epoch numbers or identifiers.
        savepath (str): The path where the tracked metrics will be saved.
    """
    def __init__(self, tlearn, region=None, savepath='.'):
        """
        Initializes the tracker.
        -----
        Args:
            - savepath (str): The path where the tracked metrics will be saved.
                             Defaults to the current directory.
        """
        self.logs = OrderedDict()
        self.epoch = []
        self.tlearn = tlearn
        self.region = region
        self.savepath = savepath
        
    def update(self, data):
        """
        Updates the tracked metrics with new data.
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

                    if isinstance(value, torch.Tensor):
                        self.logs[key][loss].append(value.detach().cpu().item())
                    else:
                        self.logs[key][loss].append(value)
            else:
                if key not in self.logs:
                    self.add(key)

                if isinstance(self.logs[group],list):
                    self.logs[key].append(group)
                else:
                    self.logs[key]+=group

    def add(self, key,  is_dict=None, parent_dict=None):
        """
        Adds a new metric to the tracker.
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
            parent_dict[key]= []
    
    def plots(self, key:list=None):
        """
        Plotting utility for all metrics stored in the tracker
        """
        if key == None:
            keys = list(self.logs['training'].keys())

        for key in keys:
            plt.figure()
            plt.plot(np.asarray(self.logs['training'][key]))
            plt.plot(np.asarray(self.logs['validation'][key]))
            plt.title(key, fontsize=20)
            plt.xlabel('epoch', fontsize = 18)
            plt.ylabel(key, fontsize = 18)
            plt.legend(['Training','Validation'], loc = 'upper right', fontsize = 18)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            if self.tlearn=='cooc' or self.tlearn=='self':
                loc = os.getcwd() + self.savepath + self.tlearn+ '/' +self.region + '/'+ key
            else:
                loc = os.getcwd() + self.savepath + self.tlearn + '/' + key
                
            plt.savefig(loc + '.png', dpi = 600)
            plt.close()


        #TODO: Add a set of keys to control y-label and generate manuscript ready performance plots.
        #TODO: Fix color scheme, 
    