# coding: utf-8
""" MIT License """
'''
    Project: Parenchymal Attention Network
    Authors: Axel Masquelin
    Description: Secondary Analysis showing the distribution of performance across all runs
'''
# Libraries
# ---------------------------------------------------------------------------- #
from argparse import ArgumentParser, Namespace
from email.policy import default

import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import logging
import string
import sys
import os
import glob
import csv

from PulmonaryMAE.utils import utils
# ---------------------------------------------------------------------------- #
def annotatefig(sig, x1, x2, y, h):
    if sig < 0.05:
        if (sig < 0.05 and sig > 0.01):
            sigtext = '*'
        elif (sig < 0.01 and sig > 0.001): 
            sigtext = '**'
        elif sig < 0.001: 
            sigtext = '***'

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color='k')
def violin_plots(df, metric, method, key,
                 sig1 = None, sig2 = None, sig3 = None,
                 sig4 = None, sig5 = None, sig6 = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """

    colors = {
                'Individual':   "BuGn",
                'Combined':     "RdBu",
             }

    titles = {
            'rf': 'Random Forest',
            'svm': 'Support Vector Machine',
            'Lasso': 'LASSO Regression'
            }

    fig, ax = plt.subplots()
    chart = sns.violinplot(data = df, cut = 0,
                           inner="quartile", fontsize = 16,
                           palette= sns.color_palette(colors[key], 8))

    chart.set_xticklabels(chart.get_xticklabels(), rotation=25, horizontalalignment='right')
    plt.xlabel("Dataset", fontsize = 14)

    if metric == 'AUC':
        plt.title(titles[method] + " " + metric.upper() + " Distribution", fontweight="bold", pad = 20)
        plt.yticks([0.5,0.6,.7,.8,.9,1], fontsize=15)
        plt.ylabel(metric.upper(), fontsize = 16)
        plt.ylim(0.45,1.07)
    else:
        plt.title(method + " " + metric.capitalize() + " Distribution", fontweight="bold")
        plt.ylabel(metric.capitalize(), fontsize = 16)
        
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if sig1 != None:
        x1, x2 = 0, 1                           # 0i vs. 10b/10ib
        y, h, col = .985, .0025, 'k'
        annotatefig(sig1, x1, x2, y, h)

    if sig2 != None:
        x1, x2 = 0, 2                           # 0i vs. 15b/15ib
        y, h, col = 1.015, .0025, 'k'
        annotatefig(sig2, x1, x2, y, h)

    if sig3 != None:
        x1, x2 = 1, 2                            # 10b/10ib vs 15b/15ib
        y, h, col = 0.993, .0025, 'k'
        annotatefig(sig3, x1, x2, y, h)

    if sig4 != None:
        x1, x2 = 0, 3                           # 0i vs. Maximum Diameter
        y, h, col = 1.065, .0025, 'k'
        annotatefig(sig4, x1, x2, y, h)

    if sig5 != None:
        x1, x2 = 1, 3                           # 10b/10ib vs Maximum Diameter
        y, h, col = 1.04, .0025, 'k'
        annotatefig(sig5, x1, x2, y, h)

    if sig6 != None:
        x1, x2 = 2, 3                           # 15b/15ib vs. Maximum Diameter
        y, h, col = .981, .0025, 'k'
        annotatefig(sig6, x1, x2, y, h)

    result_dir = os.path.split(os.getcwd())[0] + "/results/"
    if key == 'individual':
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
                    dpi=600)
    else:
        plt.savefig(result_dir + metric + "_" + method + "_Across_" + key + "_Dataset.png", bbox_inches='tight',
                    dpi=600)

    plt.close()
def lineplot(stats_dic,maskratios):
    x =np.asarray(maskratios.T)

    plt.style.use('seaborn-dark-palette')
    plt.figure()
    # plt.title('MAE Performance Across Mask-ratios', fontsize=18)
    plt.xlabel("Mask Ratio", fontsize=14)
    plt.ylabel("%", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.errorbar(maskratios, stats_dic['auc'][0][0]*100, stats_dic['auc'][0][1]*100, linestyle='--')

    plt.errorbar(maskratios, stats_dic['sensitivity'][0][0]*100, stats_dic['sensitivity'][0][1]*100, linestyle = '-.', alpha=0.6)
    
    plt.errorbar(maskratios, stats_dic['specificity'][0][0]*100, stats_dic['specificity'][0][1]*100, linestyle = ':', alpha= 0.4)
    plt.legend({'AUC','Sensitivity','Specificity'},loc ='lower left', fontsize=12)

    plt.fill_between(maskratios, y1=(stats_dic['auc'][0][0]+stats_dic['auc'][0][1])*100,
                                 y2=(stats_dic['auc'][0][0]-stats_dic['auc'][0][1])*100,
                                 alpha= 0.3)    
    plt.fill_between(maskratios, y1=(stats_dic['sensitivity'][0][0]+stats_dic['sensitivity'][0][1])*100,
                                 y2=(stats_dic['sensitivity'][0][0]-stats_dic['sensitivity'][0][1])*100,
                                 alpha= 0.2)
    plt.fill_between(maskratios, y1=(stats_dic['specificity'][0][0]+stats_dic['specificity'][0][1])*100,
                                 y2=(stats_dic['specificity'][0][0]-stats_dic['specificity'][0][1])*100,
                                 alpha= 0.25)
    plt.grid(visible=True, which='major', alpha = 0.15)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.getcwd() + '/results/' + 'AUC_across_maskratios.png', dpi = 600)

def main(args):
    """
    Definition:
    Inputs:
    Outputs:
    """

    # Network Parameters
    maskratios = args.mask_ratios
    metrics = args.metrics
    
    # Variable Flags
    data_dic = {'auc':[], 'sensitivity':[], 'specificity':[]}
    print(os.getcwd())
    
    for metric in metrics:
        print(metric)
        # Dataframe Inits_  
        df = pd.DataFrame()                # General Dataframe to generate Bar-graph data
        

        for maskratio in maskratios:
            directory = os.getcwd() + '/results/' 
            header = str(maskratio) + 'x'    
            filename = directory + header + '/' + metric + '.csv'
            values_=[]
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                next(reader)

                for row in reader:
                    for l in range(len(row)-1):
                        values_.append(float(row[l+1]))
                df[header] = np.transpose(values_)
                
        cols = df.columns.tolist()
        stats = np.zeros((2,len(cols)))
        for i, col in enumerate(cols):
            print(col)
            stats[0,i] = float(np.mean(df[col]))
            stats[1,i] = float(np.std(df[col]))
        
        df_stats = pd.DataFrame(stats, index=['mean','std'], columns=cols)
        # if args.checkstats:

        # if args.violinplot:

        data_dic[metric].append(stats)
    if args.lineplot:
        lineplot(data_dic, maskratios)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Mask Ration'
    parser.add_argument('--mask_ratios', type=range, default=np.round(np.arange(0,1.01,0.1),1))
    parser.add_argument('--metrics', type= list, default= ['auc', 'sensitivity', 'specificity'])
    # Statistics
    parser.add_argument('--violinplots', type = bool, default= True)
    parser.add_argument('--lineplot', type= bool, default= True)
    parser.add_argument('--checkstats', type=bool, default=True)
    
    return parser

if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)
   parser = build_parser()
   args = parser.parse_args()

   main(args=args)