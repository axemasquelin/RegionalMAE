# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
import time
import sys
import os
# ---------------------------------------------------------------------------- #

class ProgressBar():
    def __init__(self, model: str, method:str=None, maxfold:int=None, region:str=None,bar_length:int=50, chr:str='=') -> None:
        self.modelname = model
        self.method = method
        self.maxfold = maxfold
        self.chr = chr
        self.region = region
        self.barlength = bar_length

    def info(self,ms):
        """
        Parameters
        ----------
        1. model: str  - string containing model being used
        2. method: str - string containing method being used
        3. ms: float   - masks-ratio value
        """
        sys.stdout.write("\n Model: {0}, Method: {1}, Mask-ratio: {2}".format(self.modelname, self.methodname, ms))
        
    def _update(self, task:str, tlearn:str, fold:int):
        self.tlearn = tlearn
        self.task = task
        self.fold = fold

    def visual(self, epoch, max_epoch):
        """
        Definition:
        Inputs: 1) value        - int showing the number of repetition left
                2) endvalue     - int maximum number of repetition
                3) bar_length   - int shows how long the bar is
                4) chr          - str character to fill bar
        """
        
        percent = float(epoch) / max_epoch
        arrow = self.chr * int(round(percent * self.barlength)-1) + '>'
        spaces = ' ' * (self.barlength - len(arrow))
        sys.stdout.write("\r Region: {0} | Learn: {1} | Task: {2}  | Fold {3}/{4} | [{5}] {6}%".format(
                                                                                self.region,
                                                                                self.tlearn,
                                                                                self.task,
                                                                                self.fold,
                                                                                self.maxfold,
                                                                                arrow + spaces,
                                                                                int(round(percent*100))))
        
        

