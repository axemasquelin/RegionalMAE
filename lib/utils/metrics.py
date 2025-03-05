# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description: Utility functions to plot metrics
'''
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.metrics import roc_curve, auc, confusion_matrix
from skimage.metrics import structural_similarity as ssmi
from torch.nn import Softmax
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import yaml
import os
import cv2
# ---------------------------------------------------------------------------- #
def save_df(data, cfg, tlearn, subfolder):
    """
    """
    if tlearn == 'cooc' or tlearn == 'self':
        output_file = os.getcwd() + cfg['savepath'] +  tlearn + '/' + subfolder + '_performance.csv'
    else:
        output_file = os.getcwd() + cfg['savepath'] + tlearn + '/' + str(tlearn) + '_performance.csv'
    
    data.to_csv(output_file)

def check_directory_status(dirpath:str):
    """
    Checks the status of a directory and performs the following actions:
    
    - If the directory exists, it removes all files with a ".png" extension within it.
    - If the directory does not exist, it creates the directory.
    
    Args:
        dirpath (str): The path to the directory to check and modify.
    """
    if os.path.exists(dirpath):
        for file in os.listdir(dirpath):
            if file.endswith(".png"):
                os.remove(os.path.join(dirpath, file))
    else:
        os.makedirs(dirpath)

def reconstructions(recs:torch.Tensor, masked:torch.Tensor, imgs:torch.Tensor, ids:list, cfg:dict, tlearn:str, region:str):
    """
    Saves Reconstructed Images for comparison to original images
    -----
    Args:
        
    """
    
    savedirectory = os.getcwd() + cfg['savepath'] + tlearn + '/' + region + '/Images/'
    check_directory_status(savedirectory)

    ssim_scores = []
    recs = recs.detach().cpu().numpy()
    masked = masked.detach().cpu().numpy()
    imgs = imgs.detach().cpu().numpy()
    
    for i, pid in enumerate(ids):
        if not cv2.imwrite(savedirectory + pid + '_reconstructed.png', np.transpose(recs[i]*255)):
            raise Exception("Could not write image")
        if not cv2.imwrite(savedirectory + pid + '_original.png', np.transpose(imgs[i]*255)):
            raise Exception("Could not write image")      
        if not cv2.imwrite(savedirectory + pid + '_masked.png', np.transpose(masked[i]*255)):
            raise Exception("Could not write image")
        
        ssim_scores.append(get_ssim(image=recs[i], target=imgs[i]))

    return ssim_scores

def get_ssim(image, target):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    Args:
        image (np.array): Reconstructed image.
        target (np.array): Target image.
    
    Returns:
        float: SSIM score between the reconstructed and target image.
    """
    return ssmi(image, target, win_size=11,  channel_axis=0, data_range=target.max() - target.min())

def get_dor(prediction,targets):
    '''
    '''
    tpos,fpos,tneg,fneg = 0,0,0,0
    
    for i in range(len(targets)):
        if prediction[i] == targets[i]:
            if prediction[i] == 1:
                tpos += 1
            else:
                tneg += 1
        else:
            if prediction[i] == 1:
                fpos += 1
            else:
                fneg += 1
    
    dor = (tpos*tneg) / (fneg*fpos)
    prec = tpos / (tpos + fpos)
    sens = tpos / (tpos + fneg)
    spec = tneg / (tneg + fpos)
    rec = tpos / (tpos + fneg)
    acc = (100 * (tpos + tneg)/(tpos+tneg+fpos+fneg))

    return {'dor': dor,
            'prec':prec,
            'sens':sens,
            'spec':spec,
            'rec': rec,
            'acc':acc
            }

def save_model_outcomes(prediction, targets, ids, cfg, tlearn, region:str=None):
    ""
    ""
    if tlearn=='cooc' or tlearn=='self':
        savedirectory = os.getcwd() + cfg['savepath'] + tlearn + '/' + region + '_outputs.csv'
    else:
        savedirectory = os.getcwd() + cfg['savepath'] + tlearn + '/' + 'outputs.csv'

    outcomes = pd.DataFrame({'pids':ids, 'pred':prediction,'ca':targets})
    outcomes.to_csv(savedirectory)
    
def plot_metric(params):
    '''
    Description: Function Plots Accuracies over epochs for all methodologies
    Inputs:
        - fig: Figure Value to avoid overwritting open figures
        - trainloss: Train Accuracy of latest trained netowrk
        - validLoss: Validation Accuracy of latest trained network
        - mode: Data type being processed (original, masked)
        - ms: mask size for masked dataset
    '''
    plt.figure()
    plt.plot(params['trainmetric'])
    plt.plot(params['valmetric'])
    plt.title(params['title'], fontsize = 20)
    plt.xlabel(params['xlabel'], fontsize = 18)
    plt.ylabel(params['ylabel'], fontsize = 18)
    plt.legend(params['legend'], loc = 'upper right', fontsize = 18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
   
    plt.savefig(params['savepath']+ params['savename']+'.png', dpi = 600)
    plt.close()

def calcAuc (fps:list,
             tps:list,
             region:str,
             plot_roc:bool=False,
             savepath:str=None
             ):
    ''' 
    Calculate mean ROC/AUC for a given set of true positives (tps) & false positives (fps)
    '''

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (_fp, _tp) in enumerate(zip(fps, tps)):
        tprs.append(np.interp(mean_fpr, _fp, _tp))
        tprs[-1][0] = 0.0
        roc_auc = auc(_fp, _tp)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(itr, figsize=(10,8))
            plt.plot(
                _fp, _tp, lw=1, alpha=0.5,
                # label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
    # print(len(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, region, savepath)
    return aucs

def plot_roc_curve(tprs,
                   mean_fpr,
                   mean_tpr,
                   mean_auc,
                   std_auc,
                   region,
                   savepath
                   ):
    ''' 
    Plot roc curve per fold and mean/std score of all runs
    -----------
    Parameters:
        tprs: list of true positives
        mean_fpr:
        mean_tpr, 
        mean_auc,
        std_auc, 
        savepath: location to save figure

    '''

    plt.figure(figsize=(10,8))

    plt.plot(
        mean_fpr, mean_tpr, color='k',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    )

    # plot std
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=.4, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if region != None:
        plt.title('ROC Curve for Mask-ratio: ' + region, fontsize=20)
    else:
        plt.title('ROC Curve', fontsize=20)
    
    plt.legend(loc="lower right", fontsize=18)

    
    plt.savefig(savepath + 'ROC.png', dpi = 600)
    
    plt.close()

def calcLoss_stats(loss, mode, static_fig, figure,
                   plot_loss = True,
                   plot_static = False):

    losses = []
    
    for itr, _loss in enumerate(loss):
        # print("_Loss: " + str(_loss))
        losses.append(_loss)
        
        if plot_loss == True:
            plt.figure(figure, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )
        if plot_static == True:
            plt.figure(static_fig, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )

    mean_loss = np.mean(losses, axis=0)
    std_loss = np.std(losses, axis=0)
    loss_upper = np.minimum(mean_loss + std_loss, 1)
    loss_lower = np.maximum(mean_loss - std_loss, 0)

    if plot_loss == True:
        plt.figure(figure)
        plt.plot(
            mean_loss, color='k',
            label=r'Mean Loss'
            )
        plt.fill_between(
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        
        plt.title(" Loss over Epochs - " + str(mode), fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=16)

    if plot_static == True:
        plt.figure(static_fig)
        plt.fill_between(
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.3, label=r'$\pm$ std.'
        )
        plt.title(" Loss over Epochs - All Approaches" , fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=16)

    return mean_loss, loss_upper, loss_lower

