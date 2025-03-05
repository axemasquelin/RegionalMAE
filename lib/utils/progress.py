# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
import torch
import time
import sys
import os
# ---------------------------------------------------------------------------- #
# TODO: Add Loss & Performance Logger to progress bar
class ProgressBar():
    def __init__(self, model:str, tlearn:str, method:str, maxpoch:int, bar_length:int=10, chr:str='=') -> None:
        self.params = {'model': model,
                       'method': method,
                       'tlearn': tlearn,
                       'chr': chr,
                       'bar_length': bar_length,
                       'epoch': 0,
                       'max_epoch': maxpoch,
                       'loss': 0.0,
                       'start_time': time.time(),
                       'time_elapsed': 0.0
                       }

    def update(self, data):
        for key, value in data.items():
            if key not in self.params:
                self.add(key)

            self.params[key] = value
    
    def add(self, key):
        """
        Add new key entry in dictionary
        -----
        Args:
            - key (str): string containing the new key that needs to be added into dictionary
        """
        assert key not in self.params, "Key already in dictionary"
        self.params[key] = None

    def reset(self, max_epoch:int):
        self.params['epoch'] = 0
        self.params['loss'] = 0.0
        self.params['start_time'] = time.time()
        self.params['time_elapsed'] = 0.0
        self.params['max_epoch'] = max_epoch
        
    def visuals(self):
        """
        Generates terminal visuals based on updated information for each epoch during training or inference.
        """
        epoch = self.params['epoch']
        max_epoch = self.params['max_epoch']
        percent = float(epoch) / max_epoch
        arrow = self.params['chr'] * int(round(percent * self.params['bar_length'])-1) + '>'
        spaces = ' ' * (self.params['bar_length'] - len(arrow))

        self.params['time_elapsed'] = time.time() - self.params['start_time']
        time_per_epoch = self.params['time_elapsed'] / (epoch + 1)  # Avoid division by zero
        estimated_remaining_time = (max_epoch - epoch) * time_per_epoch

        # Format the time for better readability
        time_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(self.params['time_elapsed']))
        estimated_remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))

        sys.stdout.write(
            f"\rMethod: {self.params['tlearn']} | "
            f"Region: {self.params['method']} | "
            f"Task: {self.params['task']} | "
            f"Epoch: [{arrow + spaces}] {epoch}/{max_epoch} | "
            f"Loss: {self.params['loss']:.4f} | "
            f"Time Elapsed: {time_elapsed_str} | "
            f"ETA: {estimated_remaining_time_str}"
        )
        sys.stdout.flush()

