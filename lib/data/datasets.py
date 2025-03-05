# coding: utf-8 
'''
""" MIT License """
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''
#----------------Libaries---------------#
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torch.utils.data import Dataset
from sklearn.utils import resample
from itertools import cycle
from PIL import Image

import pandas as pd
import numpy as np
# import tiffile
import h5py
import glob
import os, time

from lib.data import Loaders
#---------------------------------------#

def hdf5_2_df(data):
    """
    Converts an HDF5 Dictionary into a Dataframe. Does take into consideration the potential for metadata but I don't think this conversion is ideal. 
    TODO: Delete comments referencing internal debates... not like I can code for swack.
    """
    table = []
    for idx, group in data.items():
        flat_dict = {}
        flat_dict['pid'] = str(idx)
        for key in group.keys():
            flat_dict[key] = group[key][()]
        for key in group.attrs.keys():
            flat_dict[key] = group.attrs[key][()]
        table.append(flat_dict)
    df = pd.DataFrame(table)
    return df

def resample_hdf5(data,  method:str='downsample', random_state:int=2024):
    """
    Allows user to upsample or downsample data to have equal positive and negative cases. 
    I should not have done it this way. So many downstream problems
    -----
    Args:
        - data (dict)
        - method (str): whether upsampling or downsampling strategies will be used
        - random_state (int): random value to set the random seed to ensure reproduciblity
    --------
    Returns:
        samples (dict): Dictionary containing all components of the data
    """
    np.random.seed(seed=random_state)

    samples = pd.DataFrame()
    samples['pid'] = data['dataset']['pid'][()]
    samples['image'] = [data['dataset']['image'][idx] for idx in range(len(data['dataset']['image'][()]))]
    samples['mask'] = [data['dataset']['mask'][idx] for idx in range(len(data['dataset']['image'][()]))]
    samples['ca'] = data['dataset']['ca'][()]
    samples['series'] = data['dataset']['time'][()]
    samples['view'] = data['dataset']['view'][()]
    samples['slice'] = data['dataset']['Slice'][()]
    
    # samples = pd.merge(samples, left_index=True, right_index=True) #This is a temporary solution
    
    neg_samples = samples[samples.ca==0.0]
    pos_samples = samples[samples.ca==1.0]
    
    # Upsample the pos samples
    if method == 'upsample':
        data_pos_upsampled = resample(
            pos_samples,
            n_samples=len(neg_samples),
            replace=True, 
            random_state= random_state
        )
        return pd.concat([data_pos_upsampled, neg_samples])

    # Downsample the neg samples
    elif method == 'downsample':
        data_neg_downsampled = resample(
            neg_samples,
            n_samples=len(pos_samples),
            replace=True, 
            random_state= random_state
        )
        return pd.concat([pos_samples, data_neg_downsampled])
    
    # Return Raw Dataset
    elif method == 3:
        return pd.concat([pos_samples, neg_samples])

    else:
        print('Error: unknown method')

def train_test_hdf5(data, split:float=0.1, random_state:int=2024):
    """
    Custom train_test_split function designed for hdf5 filetypes
    parameters:
        data - hdf5.File
        split - float
        random_state - int
    returns:
        selected_data
        remaining_data
    """
    np.random.seed(seed=random_state)

    ids = [pid for pid in data.keys()]
    ca = [group['ca'][()] for idx,group in data.items()]
    ref_table = {'ids':ids, 'ca':ca}

    idx_0 = np.where(np.array(ref_table['ca'])==0)[0]
    idx_1 = np.where(np.array(ref_table['ca'])==1)[0]

    n_row0 = int(len(idx_0) * split)
    n_row1 = int(len(idx_1) * split)

    selected_idx0 = np.random.choice(idx_0, n_row0, replace=False)
    selected_idx1 = np.random.choice(idx_1, n_row1, replace=False)
    selected_idx = np.concatenate((selected_idx0, selected_idx1))

    remaining_idx = np.setdiff1d(np.arange(len(ref_table['ca'])), selected_idx)

    select_data = {}
    remain_data = {}
    
    for idx in selected_idx:
        if ref_table['ids'][idx] in data:
            select_data[ref_table['ids'][idx]] = data[ref_table['ids'][idx]]
    
    for idx in remaining_idx:
        if ref_table['ids'][idx] in data:
            remain_data[ref_table['ids'][idx]] = data[ref_table['ids'][idx]]
    
    return select_data, remain_data

def extract_integers_from_arrays(df):
    """
    Extracts integers from array-like values in a DataFrame and
    converts numpy.float64 to standard Python floats.

    Args:
        df: A pandas DataFrame potentially containing array-like values.

    Returns:
        A modified DataFrame where array-like values are replaced 
        with extracted integers, and numpy.float64 are converted 
        to standard floats.
    """

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, tuple, set, np.ndarray))).any():
            df[col] = df[col].apply(
                lambda x: [
                    float(val) if isinstance(val, np.float64) else int(val) 
                    for val in (x.tolist() if isinstance(x, np.ndarray) else x) 
                    if isinstance(val, (int, float, np.ndarray, str)) and (str(val).isdigit() if isinstance(val, str) else True)
                ]
            )
    return df

def resample_data(data, method:int=2, random_state:int=2024):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
        -- method = 3 - return all cases
    '''
    data = data.map(lambda x: int(x.decode('utf-8')) if isinstance(x,bytes) else x)
    data_neg = data[data['ca']==0]
    data_pos = data[data['ca']==1]

    # Upsample the pos samples
    if method == 1:
        data_pos_upsampled = resample(
            data_pos,
            n_samples=len(data_neg),
            replace=True, 
            random_state= random_state
        )
        return pd.concat([data_pos_upsampled, data_neg])

    # Downsample the neg samples
    elif method == 2:
        data_neg_downsampled = resample(
            data_neg,
            n_samples=len(data_pos),
            replace=True, 
            random_state= random_state
        )
        return pd.concat([data_pos, data_neg_downsampled])
    
    # Return Raw Dataset
    elif method == 3:
        return pd.concat([data_pos, data_neg])

    else:
        print('Error: unknown method')

def load_files(cfg, filepath:str, ext:str=None, cohort:str=None, resample:str=None, seed:int=2024) -> pd.DataFrame:
    """
    Load the data from the file and resample if necessary
    -----
    Args:
        - cfg (dict): Experiment parameter configuration
        - cohort (str): Name of the cohort to load
        - ext (str): File extension
        - resample (str): Resampling method
        - seed (int): Random seed value
    -----
    Returns:
        - data (pd.DataFrame): Dataframe containing the data from the h5 file
    """
    
    if ext== '.csv':
        data = pd.read_csv(os.getcwd() + cohort + ext)
        data = resample_data(data, method=2, random_state=cfg['seed'])
        data = data.drop(['Unnamed: 0'], axis = 1)
        data['pid'] = data['pid'].astype(int)

    elif ext=='.h5':
        filepath = os.getcwd() + filepath + ext
        data = h5py.File(filepath,'r')
        
        if cohort == 'COPD':
            data = hdf5_2_df(data)
            data = data.head(2016)

        if resample != None:
            if os.path.basename(filepath).split('_')[-1] == 'NLSTdata.h5':
                data = resample_hdf5(data=data, method=resample, random_state=cfg['seed'])
                
            else:
                data = hdf5_2_df(data)
                data = resample_data(data=data, method=2, random_state=cfg['seed'])
    else:
        raise Exception('Filetype not supported - please use .csv or .h5')

    return data

def create_datasets(data, cohort:str, cfg:dict, augment:bool=False, seed:int=2023) -> Dataset:
    """
    Generates a training, validation, and testing set for training model using a predefined testing split in configuration file.
    -----
    Args:
        - data (pd.DataFrame): pandas dataframe containing the data from the h5 file
        - cohort (str): Name of the cohort to load
        - cfg (dict): Experiment parameter configuration
        - seed (int): Random seed value
    Returns:
        - trainset (torch.Dataset): Returns a torch dataset class for training data
        - valset (torch.Dataset): Torch dataset class that handles transformations and normalization of data
        - testset (torch.Dataset): Torch dataset class that handles the testing data transformations, if any, and normalization of data
    """
    
    hdf5_train, hdf5_test = train_test_split(data, test_size=cfg['training_data']['split'][2], random_state=seed)
    hdf5_train, hdf5_val = train_test_split(hdf5_train, test_size=cfg['training_data']['split'][1], random_state=seed)

    trainset =  Loaders.HDF5(hdf5_train, cohort=cohort, norms=cfg['training_data']['norm'])
    valset = Loaders.HDF5(hdf5_val, cohort=cohort, norms=cfg['training_data']['norm'])
    testset = Loaders.HDF5(hdf5_test, cohort=cohort, norms=cfg['training_data']['norm'], testing=True)

    return trainset, valset, testset


