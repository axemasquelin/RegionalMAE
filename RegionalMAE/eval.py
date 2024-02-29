
""" MIT License """
'''
    Project: RegionalMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from RegionalMAE.utils import images
from RegionalMAE.utils import utils
from RegionalMAE.utils import metrics

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
import sys
# --------------------------------------------

def init_optimizer(model, opt):
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
        optimizer = optim.Adam(model.parameters(), lr = opt['lr'], betas= opt['betas'], eps= opt['eps'])
    if opt['optim'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= opt['lr'], momentum= opt['momentum'])
    if opt['optim'] == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr = opt['lr'], rho = opt['rho'], eps = opt['eps'], weight_decay = opt['decay'])

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

    # Defining Classifier Loss Function
    if lossfunc == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()
    if lossfunc == 'MSE':
        crit = nn.MSELoss().cuda()
    if lossfunc == 'BCE':
        crit = nn.BCELoss().cuda()
    if lossfunc == 'L1_loss':
        crit = nn.L1Loss().cuda()

    return crit

class PyTorchTrials():
    """
    Description: Model Training, Validation, and Evaluation Class using Pytorch
    """
    def __init__(self, fold:int, model:nn.Module, tlearn:str, task:str, region:str, config:dict, device:torch.device, progressbar):
        """
        Initialization function for Training/Validation/Evaluation Environment
        -----------
        Parameters:
        fold: int
        task: str
        model: nn.Module
        config - dictionary containing experiment parameters
        device - torch.device()
        """        
        self.fold = fold
        self.task = task
        self.tlearn = tlearn
        self.device = device
        self.region = region
        self.config = config
        self.flags = config['flags']
        self.progressbar = progressbar

        # Defining Optimizer for Models OptMAE, OptDx
        self.model = model
        self.optimizer = init_optimizer(model= self.model, opt=config['optimizer'])

        # Defining Loss Function for MAE and Dx Classifier
        if task == 'MAE':
            self.loss = select_loss(config['optimizer']['MAELoss'])
        elif task == 'Segment':
            self.loss = select_loss(config['optimizer']['SegLoss'])
        else: 
            self.loss = select_loss(config['optimizer']['DxLoss'])

        # Initialize Checkpoint Lists to save model
        self.checkpoints = []
    
    def _initmetrics_(self):
        """
        Stores metrics for training, validations, and testing
        """
        self.train_Loss = np.zeros((self.config['optimizer']['epchs'])) 
        self.validation_Loss = np.zeros((self.config['optimizer']['epchs']))

        if self.task == 'Segment':
            self.train_DICEmetric = np.zeros((self.config['optimizer']['epchs']))
            self.validation_DICEmetric = np.zeros((self.config['optimizer']['epchs']))

            self.train_HD95metric = np.zeros((self.config['optimizer']['epchs']))
            self.validation_HD95metric = np.zeros((self.config['optimizer']['epchs']))
        
        elif self.task == 'Dx':
            self.train_DxAcc = np.zeros((self.config['optimizer']['epchs']))
            self.validation_DxAcc = np.zeros((self.config['optimizer']['epchs']))

    def _getmetrics_(self):
        """
        Returns a dictionary containing metrics across training, validation, and evaluation epochs
        -----------
        Parameters:
        --------
        Returns:
        """ 
        if self.task == 'Segment':
            return {
                'train_Loss': self.train_Loss,
                'validation_Loss': self.validation_Loss,
                'train_DICE': self.train_DICEmetric,
                'validation_DICE': self.validation_DICEmetric,
                'train_HD95': self.train_HD95metric,
                'validation_HD95': self.validation_HD95metric    
            }
        elif self.task == 'Dx':
            return {
                'train_Loss': self.train_Loss,
                'validation_Loss': self.validation_Loss,
                'train_Acc': self.train_DxAcc,
                'validation_Acc': self.validation_DxAcc
            }
        
        return {'train_Loss': self.train_Loss,
                'validation_Loss': self.validation_Loss,}

    def calc_loss(self, labels:list, predictions:list, alpha:list):
        """
        Calculates Loss for a given batch
        -----------
        Parameters:
        --------
        Returns:
        """
        return self.loss(predictions, labels)

    def calc_metric(self, label: list, pred:list) -> float:
        """
        Returns metrics of intereset based on defined task
        -----------
        Parameters:
        --------
        Returns:
        """
        if self.task == 'Segment':
            metric= {'DICE': metrics.get_DICEcoeff(pred, label),
                   'HD95': metrics.get_HD95(pred, label)
                   }
        else: 
            metric=  {'accuracy': metrics.get_acc(pred, label)}
        
        return metric

    def _batches_(self, loader, train:str=False) -> dict: 
        """
        """
        if self.task == 'Segment':
            running_DICE = 0.0
            running_HD95 = 0.0

        elif self.task == 'Dx':
            running_DxAcc = 0.0

        running_loss = 0.0        

        for i, data in enumerate(loader):
            imgs = data['image'].to(device=self.config['device'], dtype= torch.float)
            imgs = torch.autograd.Variable(imgs)

            Dxlabels = data['Dxlabel'].to(device=self.config['device'])
            Dxlabels = torch.autograd.Variable(Dxlabels)

            tumorMask = data['Mask'].to(device=self.config['device'])
            tumorMask = torch.autograd.Variable(tumorMask)

            out = self.model(imgs)

            if self.task == 'Segment':
                loss = self.calc_loss(label=tumorMask, predictions=out['prediction'])
                metrics = self.calc_metric(label=tumorMask, pred=out)
                running_DICE += metrics['DICE']
                running_HD95 += metrics['HD95']
            elif self.task == 'Dx':
                loss = self.calc_loss(label=Dxlabels, predictions= out['class'])
                metrics = self.calc_metric(label=Dxlabels, pred=out['class'])
                running_DxAcc += metrics['accuracy']
            else:
                loss = self.calc_loss(label= out['label'], predictions=out['prediction'])

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss
                        
        if self.task == 'Segment':
            return {'loss': running_loss/(i+1),
                    'DICE': running_DICE/(i+1),
                    'HD95': running_HD95/(i+1)
            }
        
        elif self.task == 'Dx':
            return {'loss': running_loss/(i+1),
                    'DxACC': (running_DxAcc / (i+1)) * 100
            }

        return {'loss': running_loss/(i+1)}

    def training(self, train_loader, validation_loader):
        """
        Training Loop for Neural Network Model that logs the loss and accuracy of the model over each epoch
        """
        self._initmetrics_()

        for epoch in range(self.config['optimizer']['epchs']): 
            self.progressbar.visual(epoch, self.config['optimizer']['epchs'])

            out = self._batches_(loader= train_loader, train=True)

            self.train_Loss[epoch] = out['loss']
            if self.task =='Segment':
                self.train_DICEmetric[epoch]  = out['DICE']
                self.train_HD95metric[epoch]  = out['HD95']

            elif self.task == 'Dx':
                self.train_DxAcc[epoch]  = out['DxACC']

            self.validation(epoch= epoch, validation_loader=validation_loader)

            if epoch % 10 == 0:
                self._registerCheckpoint_(epoch=epoch)
        
    def validation(self, epoch:int, validation_loader):
        """
        Validation loop for Neural Network training that logs the loss and accuracy of the model over eachh epoch.
        -----------
        Parameters:
        epoch: int
            current epoch the model is on for training/validation loop
        validation_loader: torch.dataloader
            torch dataloader containing the validation data from the K-fold cross validation split
        """
        with torch.no_grad():
            out = self._batches_(loader=validation_loader)

            self.validation_Loss[epoch] = out['loss']
            if self.task == 'Segment':
                self.validation_DICEmetric[epoch]  = out['DICE']
                self.validation_HD95metric[epoch]  = out['HD95']

            elif self.task == 'Dx':
                self.validation_DxAcc[epoch]  = out['DxACC']

    def inference(self, test_loader):
        """
        Inference / Evaluation of user specified model on non-trained data. 
        -----------
        Parameters:
        test_loader - torch.DataLoader
        --------
        Returns:
        """
        self.loadmodel()
        savedir = self.config['savepath'] + self.task + '/' + self.tlearn + '/'

        with torch.no_grad():
            targets = []
            softpred = []
            embeds = []
            running_MSE = 0
            for i, data in enumerate(test_loader):
                imgs = data['image'].to(device=self.device, dtype=torch.float)
                Dxlabels = data['Dx_label'].to(device=self.device)
                tumorMask = data['Mask'].to(device=self.config['device'])
                pid = data['id']
                
                out = self.model(imgs)

                if self.task != 'Dx':
                    if self.flags['ReconstructImage'] and self.task == 'MAE':
                        images.saveImages(orimgs= data['image'],
                                   patchimgs = out['maskx'],
                                   reconimgs = out['reconx'],
                                   maskratio = self.maskratio,
                                   pids=pid, savedir=savedir)

                    if self.flags['SaveFigs']:
                        images.saveImages(orimgs= tumorMask,
                                   patchimgs = None,
                                   reconimgs = out['reconx'],
                                   maskratio = self.maskratio,
                                   pids=pid,savedir=savedir)
                    if self.tlearn == 'MAE':
                        running_MSE += metrics.get_MSE(pred=out['reconx'], label = out['patchx'])

                if self.task == 'Dx':
                    for i in range(len(Dxlabels)):
                        targets.append(Dxlabels[i,1].cpu().squeeze.numpy())
                        softpred.append(out['class'][i,1].cpu().squeeze().numpy())
                        embeds.append(out['embedding'[i].cpu().squeeze().numpy()])
                        if self.flags['ProjectEmbedding']:
                            metrics.visualizeEmbedding(embeds, targets, softpred) # TODO: Last minute, not relevant to current paper

        if self.task == 'Segment':
            performance = {'DICE': metrics.get_DICEcoeff(pred=out['predx'], label=tumorMask),
                       'HD95': metrics.get_HD95(pred=out['predx'], label=tumorMask),
                       }
            
        elif self.task == 'Dx':         
            fps, tps, threshold = roc_curve(targets[:], softpred[:])
            youden_value = (tps + (1-fps)) - 1
            youden_index = youden_value.argmax()
            print(f"Sensitivity: {tps[youden_index]}, Specificity: {1-fps[youden_index]}, Youden Index: {youden_value[youden_index]}, Threshold: {threshold[youden_index]}")

            performance = {'Sensitivity': tps[youden_index],
                            'Specificity': 1-fps[youden_index],
                            'Youden Index': youden_value[youden_index],
                            }   
        
        else: 
            performance = {'MSE': running_MSE/len(test_loader)}
        
        return performance

    def _registerCheckpoint_(self, epoch):
        """
        Creates a temporary copy of the current best iteration of the network for evaluation using validation performance metrics
        -----------
        Parameters:
        epoch - int
            epoch of the model
        --------
        Returns
        """
        if self.task == 'Segment':
            if self.validation_DICEmetric[epoch] >= max(self.validation_DICEmetric[:epoch-1]):
                self.checkpoints.append({
                    'epoch': epoch,
                    'fold': self.fold,
                    'maskratio': self.maskratio,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metric': self.validation_DICEmetric[epoch],
                    'Loss': self.validation_Loss[epoch],
                    })
        elif self.task == 'Dx':
            if self.validation_DxAcc[epoch] >= max(self.validation_DxAcc[:epoch-1]):
                self.checkpoints.append({
                    'epoch': epoch,
                    'fold': self.fold,
                    'maskratio': self.maskratio,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metric': self.validation_DxAcc[epoch],
                    'Loss': self.train_Loss[epoch],
                    })
        else:
            if self.validation_Loss[epoch] >= max(self.validation_Loss[:epoch-1]):
                self.checkpoints.append({
                    'epoch': epoch,
                    'fold': self.fold,
                    'maskratio': self.maskratio,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metric': None, 
                    'Loss': self.train_Loss[epoch],
                    })
    
    def find_best_checkpoint(self, metric:str='Dx_acc') -> dict:
        """
        Find the model checkpoint with the best performance based on a given checkpoint
        -----------
        Parameters:
        metric: str
        mode: str
        --------
        Returns:
        best_checkpoint: dict
        """
        if self.task == 'Dx':
            mode = 'max'
            best_value = float('-inf')
        else:
            mode = 'min'
            best_value = float('inf')

        best_checkpoint = None
        for checkpoint in self.checkpoints:
            if metric not in checkpoint:
                raise ValueError(f"Metric '{metric}' not found in checkpoint.")
            value = checkpoint[metric]

            if (mode =='max' and value > best_value) or (mode =='min' and value < best_value):
                best_value = value
                best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def savemodel(self):
        """
        Saves best performing checkpoint
        """
        checkpoint = self.find_best_checkpoint(metric='metric')

        sys.stdout.write('\n\r {0} | Saving Network - Epoch {1}, Accuracy {2} | {3}\n '.format('-'*10,
                                                                                            checkpoint['epoch'], 
                                                                                            checkpoint['Dx_acc'],
                                                                                            '-'*10)
                                                                                            )
        
        savedir = self.config['savedirectory'] + self.task + '/' + self.tlearn + '/'

        if self.tlearn == 'MAE':
            savedir = self.config['savedirectory'] + self.task + '/' + self.tlearn + '/'+ str(self.maskratio) + 'x/'

        torch.save(self.checkpoint, savedir + 'PulMAE_' + self.task + '_bestperformance.pt')

    def loadmodel(self):
        """
        Loads best performing checkpoint or user-specified checkpoint. Note - saved model needs to fit within built in network constraints at the moment
        """
        self.model = utils.select_model(config=self.config, maskratio= self.maskratio, func=self.task)
        self.optimizer = init_optimizer(net=self.net, opt=self.config['optimizer'])

        checkpoint = torch.load(self.config['savedirectory'] + str(self.maskratio) + 'x/' + 'PulMAE_' + self.task + '_bestperformance.pt')
        self.model.load_state_dict(checkpoint['MAE_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
