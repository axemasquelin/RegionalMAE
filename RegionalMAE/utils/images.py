# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import torch
import sys
import cv2
import os
# ---------------------------------------------------------------------------- #
def saveimages(oimgs:torch.Tensor=None,
               pimgs:torch.Tensor=None,
               rimgs:torch.Tensor=None,
               maskratio:float=None,pids:list=None,
               savedir:str=None):
    '''
    '''
    pass


def unpatchify(imgs):
    N,L,W,H = imgs.shape

    reconstruct_arr = np.zeros((N,int(L**(0.5))*W,int(L**(0.5))*H))

    for n in range(N):
        channel = 0
        for i in range(int(L**(0.5))):
            for j in range(int(L**(0.5))):
                reconstruct_arr[n,W*i:W*(i+1),H*j:H*(j+1)] = imgs[n,channel,:,:]
                channel+=1

    return reconstruct_arr

def saveImages(original_imgs: torch.Tensor,
               masked_imgs: torch.Tensor,
               reconstructed_imgs: torch.Tensor,
               pids:list,
               maskratio: float):
    """
    -----------
    Parameters:
    """
    
    savedirectory = os.getcwd() + '/results/' + str(maskratio) + 'x/' + 'Images/'
    dir_exists = os.path.exists(savedirectory)
    if not dir_exists:
        sys.stdout.write('\n\r {0} | Creating Result Directories | {0}\n '.format('-'*50))
        os.mkdir(savedirectory)

    original_imgs = original_imgs.detach().cpu().numpy()
    masked_imgs = unpatchify(masked_imgs.detach().cpu().numpy())
    reconstructed_imgs = unpatchify(reconstructed_imgs.detach().cpu().numpy())
    
    for i, pid in enumerate(pids):
        cv2.imwrite(savedirectory + pid + '_original.png', original_imgs[i]*255)  
        cv2.imwrite(savedirectory + pid + '_masked.png', masked_imgs[i]*255)  
        cv2.imwrite(savedirectory + pid + '_reconstructed.png', reconstructed_imgs[i]*255)  


        