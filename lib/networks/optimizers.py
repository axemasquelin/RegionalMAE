""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# --------------------------------------------
def init_optimizer(net, opt):
    """
    Defines the optimizer for the experiment
    -----------
    Parameters:
    net - nn.Module()
        Initialize Neural Network
    opt - dictionary
        dictionary containing information regarding experiment optimizer, learning rate, and other parameters necessary for 
        various backpropagation functions
    --------
    Returns:
    optimizer - nn.optimizer
        optimizer for the neural network that will compute the gradients based on the provided loss functions
    """

    if opt['optim'] == 'Adam':
        optimizer = optim.Adam(net, lr = opt['lr'], betas= opt['betas'], eps= opt['eps'])
    if opt['optim'] == 'SGD':
        optimizer = optim.SGD(net, lr= opt['lr'], momentum= opt['momentum'])
    if opt['optim'] == 'AdamW':
        optimizer = optim.AdamW(net, lr=opt['lr'], eps=opt['eps'], weight_decay = opt['decay'])
    if opt['optim'] == 'Adadelta':
        optimizer = optim.Adadelta(net, lr = opt['lr'], rho = opt['rho'], eps = opt['eps'], weight_decay = opt['decay'])

    return optimizer

def select_loss(lossfunc):
    """
    Defines a loss function
    -----------
    Parameters:
    lossfunc - str
        string defining the desired loss function to train the model
    --------
    Returns:
    crit - nn.Loss()
    --------
    """
    if lossfunc == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()
    if lossfunc == 'MSE':
        crit = nn.MSELoss().cuda()
    if lossfunc == 'MAE':
        crit = nn.L1Loss().cuda()
    if lossfunc == 'BCE':
        crit = nn.BCELoss().cuda()
    if lossfunc == 'L1_loss':
        crit = nn.L1Loss().cuda()
    if lossfunc == 'NLLLoss':
        crit = nn.NLLLoss().cuda()

    return crit
