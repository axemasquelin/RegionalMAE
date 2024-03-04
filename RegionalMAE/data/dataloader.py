# coding: utf-8 
'''
""" MIT License """
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''
#----------------Libaries---------------#
from RegionalMAE.data import preprocess
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.nn import ModuleList

from sklearn.utils import resample
from itertools import cycle
from PIL import Image

import pandas as pd
import numpy as np
import tifffile
import random
import torch
import glob
import os

#---------------------------------------#

def normalize_img(img, normalization: str):
    '''
    Description:
    ----------
    Parameters:
    img - np.array
        raw image from nrrd file
    -------
    Outputs:
    img - np.array
        array containing a slice of the image
    '''

    if normalization == 'norm':
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'stand':
        pixelmean = img.mean()
        pixelstd = img.std()
        img = (img - pixelmean)/(pixelstd)
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'lognorm':
        img = (np.log10(img) - np.log10(img).min())/(np.log10(img).max()-np.log10(img).min())

    if normalization == 'logstand':
        pixelmean = np.log10(img).mean()
        pixelstd = np.log10(img).std()
        img = (np.log10(img)-pixelmean)/pixelstd

    return img

def resample_df(seed, df, method):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
        -- method = 3 - return all cases
    '''
    df_neg = df[df.ca==0]
    df_pos = df[df.ca==1]

    # Upsample the pos samples
    if method == 1:
        df_pos_upsampled = resample(
            df_pos,
            n_samples=len(df_neg),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos_upsampled, df_neg])

    # Downsample the neg samples
    elif method == 2:
        df_neg_downsampled = resample(
            df_neg,
            n_samples=len(df_pos),
            replace=True, 
            random_state= seed
        )
        return pd.concat([df_pos, df_neg_downsampled])
    
    # Return Raw Dataset
    elif method == 3:
        return pd.concat([df_pos, df_neg])

    else:
        print('Error: unknown method')

def generate_dataframe(config):
    '''
    '''
    filelist = glob.glob(config['training_data']['datadir'] + '/dataset/*' + config['training_data']['filetype'])
    df = pd.DataFrame()
    
    ca = []

    for i, filename in enumerate(filelist):   
        ca.append(int(os.path.splitext(filename.split('_')[-1])[0]))
        
    df['uri'] = filelist
    df['label'] = ca

    return df


def load_files(config):
    '''
    Function to generate a pandas dataframe containing all files for training/validation/evaluation of model
    -----
    Args:
        (1) config: configuration dictionary containing location of files and the expected filetype to be used for training
    TODO: Need to fix this to work with .nrrd files again in order to apply curriculum learning approach for future iterations. 
    '''

    if config['training_data']['filetype'] == '.csv':
        df = pd.read_csv(config['experiment']['data'])
        df = resample_df(config['experiment']['seed'], df, 2)
    
    if config['training_data']['filetype'] == '.tif': 
        original_filelist = glob.glob('./dataset/Original/*'+ config['training_data']['filetype'])
        pid = [os.path.basename(x).split('_')[0] for x in original_filelist]
        ca = [os.path.basename(x).split('_')[1].split('.')[0] for x in original_filelist]
        time = [os.path.basename(x).split('_')[2].split('.')[0] for x in original_filelist]
        df = pd.DataFrame(np.transpose([pid,original_filelist,ca,time]), columns=['pid','uri', 'ca', 'time'])
        
        masked_filelist = glob.glob('./dataset/Segmented/*'+ config['training_data']['filetype'])
        pid = [os.path.basename(x).split('_')[0] for x in masked_filelist]
        df2 = pd.DataFrame(np.transpose([pid,masked_filelist]), columns=['pid','thresh_uri'])
        
        df = pd.merge(df, df2, on='pid')
        df['ca'] = df['ca'].astype(int)
        df = resample_df(seed=2022, df=df, method=2)

    else:
        ValueError('WARNING: Filetype not compatible')

    return df


def augment_dataframe(df:pd.DataFrame, upsample:int=3,  augment:str='rand'):
    """
    Creates an augmented dataframe that randomly augments the dataset by taking slices adjacent to central slices, or takes all surrounding slices. 
    \n
    -----------
    Parameters: \n
    df - pandas.dataframe()
        Pandas dataframe containing Nrrd list
    upsample - int
        Number of slices to augment the dataset by
    augment - str
        type of augmentation employed by the function. Random will randomly select slices from the images
    --------
    Returns: \n
    df - pandas.dataframe()
    """

    if augment=='rand':
        df = df.loc[df.index.repeat(upsample)].reset_index(drop=True)  
        # df['view'] = [np.random.choice(['x','y','z']) for index in df.index]
        df['view'] = ['z' for index in df.index]

    else:
        views = cycle(['z'])
        # views = cycle(['x','y','z'])
        df = df.loc[df.index.repeat(upsample)].reset_index(drop=True)  
        df['view'] = [next(views) for view in range(len(df))]
        # df = prep.get_dims(df, augment)
        
    return df
class Nrrdloader(Dataset):
    '''
    '''
    def __init__(self, data, augmentations = None, imgsize=64, testing=False, norms=None):
        super().__init__()
        self.data = data
        self.augmentation = augmentations
        self.imgsize = imgsize
        self.normalization = norms
        self.testing = testing
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getslice__(self, img:np.array, thres:np.array, row:pd.DataFrame, edges:list, testing:bool=False):
        """
        returns slice of nrrd file
        -----------
        Parameters:
        --------
        Returns:
        im - np.array()
            Contains original image of size (1,64,64) 
        thres - np.array()
            Contains segmentation mask of size (1,64,64) 
        """
        im = np.zeros((1,img.shape[0],img.shape[1]))
        mask = np.zeros((1,thres.shape[0],thres.shape[1]))

        if edges[0] != edges[-1]:
            if testing:
                sliceid = edges[int(len(edges)/2)]
            else:
                sliceid = random.choice(edges)
        else:
            sliceid = edges[0]

        if row['view'] == 'x':
            im[0,:,:] = img[sliceid, :, :]
            mask[0,:,:] = thres[sliceid,:,:]

        if row['view'] == 'y':
            im[0,:,:] = img[:,sliceid, :]
            mask[0,:,:] = thres[:,sliceid,:]

        if row['view'] == 'z':
            im[0,:,:] = img[:,:,sliceid]
            mask[0,:,:] = thres[:,:,sliceid]
        
        return im, mask
    
    def __augment__(self, img):
        """
        Converts image from 1-channel grayscale to multi-channel grayscale
        """

        img = Image.fromarray(img[0])
        transforms = T.Compose([
            T.Grayscale(num_output_channels=3),
            #  T.GaussianBlur(kernel_size=5),
             T.RandomHorizontalFlip(),
            #  T.RandomVerticalFlip(),
            ])
            
        img = transforms(img)

        return np.asarray(img)
    
    def __getitem__(self, index:int):
        row = self.data.iloc[index]

        img = tifffile.imread(row['uri'])
        thres = tifffile.imread(row['thresh_uri'])

        edges = preprocess.scan_3darray(thres, view=row['view'], threshold=1)

        img, thres = self.__getslice__(img, thres, row, edges, testing=self.testing)
        img = preprocess.normalize_img(img, row['pid'], self.normalization)
        img = self.__augment__(img)
        label=  [int(int(row['ca'])==0),int(int(row['ca'])==1)]

        sample = {'image': img.T,
                  'Mask': thres,
                  'Dxlabel': torch.tensor(label).float(),
                  'id': os.path.basename(row['uri']).split('.')[0],
                }

        return sample

if __name__ == '__main__':
    maskeddir = os.path.split(os.getcwd())[0] + '/' + 'RaulSegmentedNodules/*.nrrd'
    
