# coding: utf-8
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description:
'''
# Dependencies
# ---------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
import math
# ---------------------------------------------------------------------------- #


def segmentation_map(im, maskmap):
    """"
    Description:
    -----------
    Parameters:
    --------
    Returns:
    """
    
    thres_img = np.zeros(im.shape)
    thres_img = im * maskmap

    return thres_img

def normalize_img(img, pid, normalization: str):
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
    # print(img[0,:,:])
    img += abs(img.min())

    if normalization == 'norm':   
        img = (img - img.min())/(img.max()-img.min())
        
    if normalization == 'stand':
        pixelmean = img.mean()
        pixelstd = img.std()
        if pixelmean <= 0 or pixelstd <=0:
            print('\n PID: {pid}, Mean: {pixelmean}, std: {pixelstd}')
        img = (img - pixelmean)/(pixelstd)
        img = (img - img.min())/(img.max()-img.min())

    if normalization == 'lognorm':
        img = (np.log10(img) - np.log10(img).min())/(np.log10(img).max()-np.log10(img).min())

    if normalization == 'logstand':
        pixelmean = np.log10(img).mean()
        pixelstd = np.log10(img).std()
        img = (np.log10(img)-pixelmean)/pixelstd

    return img

def create_bounds(shape:int, window_size:int, lower_val:int, upper_val:int):
    """
    -----------
    Parameters:
    shape - int
        Shape of the input image to verify where the edge of the image exists
    window_size - int
        prefered window size for selected object
    lower_val - int
        lowest idx value of an object in a given dimension
    upper_val - int
        highest idx value of an object in a given dimension
    --------
    Returns:
    pad_lower - int
        Modified lower bound to include necessary padding for image slice to fit window size
    pad_upper - int
        Modified upper bound to include necessary padding for image slice to fit window size
    mid_point - int
        Middle point of the object calculated based on upper and lower edge of object
    """

    pad = (window_size - (upper_val-lower_val))
    pad_lower = lower_val - (pad/2)
    pad_upper = upper_val + (pad/2)
    mid_point = lower_val + (upper_val-lower_val) / 2

    if (pad_upper >= shape):
        pad_lower -= (pad_upper - shape)
        if pad_lower < 0:
            pad_lower = 0
        pad_upper = shape
    if (pad_lower <= 0):
        pad_upper += (0 - pad_lower)
        if pad_upper > shape:
            pad_upper = shape
        pad_lower = 0

    return math.floor(pad_lower), math.floor(pad_upper), math.floor(mid_point)  

def create_roi(img_shape:list, x:list,y:list,z:list, window_size:int=64):
    """
    Create the Region of Interest in which the nodule exists and identifies the middle
    points \n 
    ----------- \n
    Parameters: \n
    row - pd.DataSeries
        Contains the information for the Nrrd files
    window_size - int
        Size of the Region of interest
    --------
    Returns:
    """  

    xlower, xupper, xmid = create_bounds(img_shape[0], window_size, x[0], x[-1])
    ylower, yupper, ymid = create_bounds(img_shape[1], window_size, y[0], y[-1])
    zlower, zupper, zmid = create_bounds(img_shape[2], window_size, z[0], z[-1])
    
    slice_idx = {
        'Xmin': xlower,
        'Xmid': xmid,
        'Xmax': xupper,
        'Ymin': ylower,
        'Ymid': ymid,
        'Ymax': yupper,
        'Zmin': zlower,
        'Zmid': zmid,
        'Zmax': zupper,
    }
    
    return slice_idx
def get_dims(df:pd.DataFrame, augment:str):
    """
    Get slice view of Nrrd file
    -----------
    Parameters:
    df - pd.Dataframe
        Panda dataframe containing paths for original nrrd and segmented nrrd
    augment - str
        string describing whether augmentation is random or inference
            random: randomly selects a slice based on a location
            central: repeats pattern to ensure selection of nearby centroid slices
    --------
    Returns:
    """
    slice_idx = np.zeros(len(df))
    bound1_lower = np.zeros(len(df))
    bound1_upper = np.zeros(len(df))
    bound2_lower = np.zeros(len(df))
    bound2_upper = np.zeros(len(df))

    for index in df.index:
        row = df.iloc[index]
        dim = [row['xdim'], row['ydim'], row['zdim']]
        roi = create_roi(row, dim, window_size=64)
        
        if row['view'] == 'x':
            slice_idx[index] = np.random.choice(np.arange(roi['Xmin'],roi['Xmax']))
            bound1_lower[index] = roi['Ymin']
            bound1_upper[index] = roi['Ymax']
            bound2_lower[index] = roi['Zmin']
            bound2_upper[index] = roi['Zmax']
        if row['view'] == 'y':
            slice_idx[index] = np.random.choice(np.arange(roi['Ymin'],roi['Ymax']))
            bound1_lower[index] = roi['Xmin']
            bound1_upper[index] = roi['Xmax']
            bound2_lower[index] = roi['Zmin']
            bound2_upper[index] = roi['Zmax']
        if row['view'] == 'z':
            slice_idx[index] = np.random.choice(np.arange(roi['Zmin'], roi['Zmax']))
            bound1_lower[index] = roi['Xmin']
            bound1_upper[index] = roi['Xmax']
            bound2_lower[index] = roi['Ymin']
            bound2_upper[index] = roi['Ymax']
    
    df['slice_idx'] = slice_idx
    df['bound1_lower'] = bound1_lower
    df['bound1_upper'] = bound1_upper
    df['bound2_lower'] = bound2_lower
    df['bound2_upper'] = bound2_upper

    return df
   
def scan_3darray(arr, view:str, threshold:int= 1):
    """
    Scan Array in a given view to get indices where nodule exists from mask
    -----------
    Parameters:
    arr
    --------
    Returns:
    x,y,z - list of position where nodule exists
    """ 
    edges=[]
    if view=='x':
        for i in range(arr.shape[0]):
            if arr[i,:,:].any() == threshold:
                edges.append(i)

    if view=='y':
        for i in range(arr.shape[1]):

            if arr[:,i,:].any() == threshold:
                edges.append(i)

    if view=='z':
        for i in range(arr.shape[2]):
            if arr[:,:,i].any() == threshold:
                edges.append(i)
    return edges