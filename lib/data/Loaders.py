# coding: utf-8 
'''
""" MIT License """
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''
#----------------Libaries---------------#
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.nn import ModuleList

from sklearn.utils import resample
from PIL import Image

import pandas as pd
import numpy as np
import random
import torch
import cv2
import os
#---------------------------------------#


def normalize(img, normalization:str, epsilon=1e-6):
    '''
    Description:
    ----------
    Parameters:
    img - np.array
        raw image from nrrd file
    -------
    Outputs:
    img - np.array
        array containing a slice of the imag
    '''    
    available_methods = ['norm', 'stand', 'lognorm', 'logstand', 'dynamic_lognorm', 'wide_norm', 'scaled_hu', 'window_norm']
    assert normalization in available_methods, f"'{normalization}' is not an available methodology, skipping normalization"
    
    img = img.astype(np.float32)

    if normalization == 'norm':   
        img = (img - img.min())/((img.max()-img.min())+epsilon)
        
    if normalization == 'stand':
        pixelmean = img.mean()
        pixelstd = img.std()
        img = (img - pixelmean)/(pixelstd)
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'lognorm':
        log_img = np.log10(img)
        logmin = np.log10(img).min()
        logmax = np.log10(img).max()
        img = (np.log10(img)-logmin)/((logmax-logmin)+epsilon)

    if normalization == 'logstand':
        pixelmean = np.log10(img).mean()
        pixelstd = np.log10(img).std()
        img = (np.log10(img)-pixelmean)/pixelstd

    if normalization == 'dynamic_lognorm':
        '''
        NOTE:   Considering the meaning behind HU, this method should not be utilized as it will distort the calibration of CT images.
        '''
        log_img = np.log10(img)
        lower_percentile = np.percentile(log_img, 30)
        upper_percentile = np.percentile(log_img, 90)
        img = (log_img - lower_percentile) / ((upper_percentile - lower_percentile)+epsilon)
        img = (img - img.min()) / (img.max() - img.min() + epsilon)

    if normalization == 'wide_norm':
        '''
        NOTE:   Mixed results at this point regarding the performance, however could find some benefits in the application as the meaning of HU
                values is not distorted.
        '''
        target_min = -2.0
        target_max = 2.0
        img = (img - img.min()) / (img.max() - img.min() + epsilon)
        img = img * (target_max - target_min) + target_min
        
    if normalization == 'scaled_hu':
        '''
        NOTE:   This method is used to scale the HU values to a more manageable range for the network to learn from. Similar to the wide_norm
        '''
        scaling = 1/1000
        img = img * scaling
    
    if normalization == 'window_norm':
        '''
        NOTE:   This method is used to window the image to a specific range. This should be the prefered methodology as no information is lost especially when clipping to lung window.
                Values to consider:
                    - Air -1000
                    - Bone [Cancelous (300 - 400) | Cortical (500 - 1900)]
                    - Fat -120 to -90
                    - Lung -700 to -600
            Recommended for Lung CT: (-600, 1600) or (-500, 1500) | [center, width]
        '''
        center, width = -600, 1500
        window_min = center - width // 2
        window_max = center + width // 2
        img = np.clip(img, window_min, window_max)
        img = (img - window_min) / (window_max - window_min)

    return img

class HDF5(Dataset):
    def __init__(self, data:dict, cohort:str, task='MAE', norms:str='norm', augment:bool=True, testing:bool=False):
        super(HDF5,self).__init__()
        self.data = data
        self.tasks = task
        self.cohort = cohort
        self.augment = augment
        self.norms = norms
        self.testing = testing

    def __len__(self)->int:
        """
        Returns length of data
        """
        return len(self.data)
    
    def __getslice__(self, img, mask, slices):
        """
        Returns slice of data
        """

        # views = ['x','y','z']
        views = ['z']
        view = random.choice(views)
        slice = random.choice(slices)
        
        if view == 'x':
            img = img[slice,:,:].astype(np.float32)
            mask = mask[slice,:,:].astype(np.float32)
        elif view == 'y':
            img = img[:,slice,:].astype(np.float32)
            mask = mask[:,slice,:].astype(np.float32)
        elif view == 'z':
            img = img[:,:,slice].astype(np.float32)
            mask = mask[:,:,slice].astype(np.float32)

        return img, mask
    def __augment__(self, img):
        """
        Randomly applies an torch augment to an image
        """
        # print(img.shape)
        if self.testing==False:
            img = Image.fromarray(img)

            transforms = T.RandomApply(ModuleList([
                T.RandomRotation(degrees=(-10,10)),
                T.RandomAffine(degrees=0, translate=(0.1,0.1)),
                T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
                # T.ColorJitter(brightness=(0.9,1.1), contrast=(0.9,1.1), saturation=0, hue=0),
                ]), p=0.3)
            
            img = transforms(img)

        return np.asarray(img)   
    
    def __getitem__(self, index):    
        row =self.data.iloc[index]
        img = row['image']
        mask = row['mask']

        if np.isnan(img).any() or np.isnan(mask).any():
            raise ValueError("NaN values found in image or mask")

        if not np.isfinite(img).all() or not np.isfinite(mask).all():
            raise ValueError("Non-finite values found in image or mask")
        
        img = normalize(img, self.norms)
  
        if self.cohort == 'COPD':
            img, mask = self.__getslice__(img, mask, slices=[33,34,35])

            if self.augment:
                    img = self.__augment__(img)

            return {
                    'image': torch.tensor(np.reshape(img,(1,64,64)), dtype=torch.float32),
                    'mask': torch.tensor(np.reshape(mask,(1,64,64)), dtype=torch.float32),  
                    'id': str(row['pid']),
            }  
        elif self.cohort == '224xNLST':
            img, mask = self.__getslice__(img, mask, slices=[111,112,113])

            if self.augment:
                    img = self.__augment__(img)

            img = np.reshape(img,(1,224,224))
            img = img.astype(np.float32)
            img = np.repeat(img, 3, axis=0)
            hot_encoding = [int(int(row['ca'][()])==0),int(int(row['ca'][()])==1)]
            return {
                    'image': torch.tensor(img, dtype=torch.float32),
                    'class_label': torch.tensor(hot_encoding).float(),

                    'id': str(row['pid']),
            } 
        else:
            if self.augment:
                    img = self.__augment__(img)
            
            hot_encoding = [int(int(row['ca'][()])==0),int(int(row['ca'][()])==1)]
            return {
                    'image': np.reshape(img,(1,64,64)),
                    'mask': np.reshape(mask,(1,64,64)),
                    'class_label': torch.tensor(hot_encoding).float(),
                    'id': str(row['pid']),
                }
