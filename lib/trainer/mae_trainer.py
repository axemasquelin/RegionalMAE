""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries and Dependencies
# --------------------------------------------
from lib.utils.metrics import reconstructions, get_dor, save_model_outcomes
from lib.networks.checkpoints import checkpoint
from lib.networks.optimizers import *
from lib.utils.tracker import tracker
from lib.trainer.EarlyStop import EarlyStop
from sklearn.metrics import roc_curve, confusion_matrix, auc

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
# --------------------------------------------
   
class PytorchTrials():
    """
    Description: Pytorch Trial Class that handles training, validating, and evaluating model while generating
    checkpoints when criterias are met. 
    """
    def __init__(self, model:nn.Module, tlearn:str, region:str, cfg:dict, progressbar=None, device=None):
        """
        Initialization function for Training/Validation/Evaluation Environment
        -----------
        Parameters:
            fold (int):
                integer defining the current fold of the K-fold cross-validation strategy.
            tlearn (str):
                Describe whether the model will apply a Joint, Independent, or Sequential Learning strategy for the embedded Concepts.
            concept (str):
                Describes whether the network will generate deep-radiomics, or whether guideline/concept radiomics will be imposed on the embedding.
            model (nn.Module):
                Variable storing the ConceptNet class.
            progressbar (class): 
                Variable for the progress bar.
        """
        # General Variables
        self.model = model
        self.device = cfg['device']
        self.cfg = cfg
        self.progressbar = progressbar
        
        # Experiment Specific Variables
        self.epochs = cfg['experiment']['rec_epchs']
        self.region = region
        self.flags = cfg['flags']
        self.tlearn = tlearn

        if cfg['experiment']['finetune']:
            self.optimizers = {'dx': init_optimizer(net=list(self.model.encoder.parameters()) + list(self.model.classifier.parameters()),opt=cfg['optimizer']),
                               'reconstruction': init_optimizer(net=list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), opt=cfg['optimizer'])}
        else:
            self.optimizers = {'dx': init_optimizer(net= list(self.model.classifier.parameters()),opt=cfg['optimizer']),
                                'reconstruction': init_optimizer(net=list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()), opt=cfg['optimizer'])}
        
        self.schedulers = {'dx': StepLR(self.optimizers['dx'], step_size=50, gamma=5e-5),
                           'reconstruction': CosineAnnealingWarmRestarts(self.optimizers['reconstruction'],
                                                                         T_0=100, T_mult=2, eta_min=0.0, last_epoch=-1)}


        # Initializing Checkpoint (General Variable)
        self.checkpoint = checkpoint(tlearn=tlearn, region=self.region, savepath=cfg['savepath'])
        self.tracker = tracker(tlearn=tlearn, region=self.region, savepath=cfg['savepath'])

    def _update_(self, loss, key):
        """
        Updates model parameters using the specified optimizer and scheduler.
        
        This method performs a single optimization step by:
            1.  Zeroing the gradients of the optimizer associated with the given key.
            2.  Performing backpropagation on the loss to calculate gradients.
            3.  Updating the model parameters using the optimizer's step function.
        -----
        Args:
            - loss (torch.Tensor): The loss tensor calculated during training.
            - key (str, optional): The key used to identify the optimizer and scheduler in the respective dictionaries.
                                   If None, it's assumed that there's only one optimizer and scheduler. Defaults to None.
        """
        loss.backward()
        self.optimizers[key].step()
        self.optimizers[key].zero_grad()
        
    def _batches_(self, loader:dict, task:str='dx', train:bool=False):
        """
        """
        
        running_loss = 0
        count = 0                    
        total = 0                         
        correct = 0   
        
        for i, data in enumerate(loader):
            imgs = data['image'].to(self.device, dtype = torch.float)

            if task == 'dx':
                # Getting the class labels for the given batch
                class_labels = data['class_label'].to(self.device, dtype=torch.float)

                # Forward pass of Diagnosis through the network
                y_hat, loss_class = self.model(imgs, labels=class_labels, lossfunc=self.cfg['optimizer']['DxLoss'], region=self.region, task=task)

                if train: self._update_(loss_class, key='dx')
                running_loss += loss_class
                
                # Calculating the number of correctly predicted images
                _, predicted = torch.max(y_hat, 1)                  # Finding predicted class from classifier
                _, actual = torch.max(class_labels,1)               # Finding actual class from target                      
                correct += (predicted == actual).sum().item()       # Number of correctly predicted images
                total += class_labels.size(0)                       # Total labels in batch
            
            else:
                # Forward pass of Reconstruction through the network
                loss_rec = self.model(imgs, lossfunc=self.cfg['optimizer']['MAELoss'], region=self.region, task=task)
                if train: self._update_(loss_rec, key='reconstruction')
                running_loss += loss_rec
            
            count += 1

        if task=='dx':
            details = {
                'dx': (running_loss/count),
                'accuracy': (correct/total)
                }
        else:
            details = {
                'reconstruction': (running_loss/count) 
                }
            
        return details

    def training(self, training_loader, validation_loader, task:str):
        """
        """
        if task == 'dx': self.epochs = self.cfg['experiment']['dx_epchs']

        early_stop = EarlyStop(patience=self.cfg['EarlyStop']['patience'],
                               delta=self.cfg['EarlyStop']['min_delta'],
                               monitor=task)
        
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            metrics = self._batches_(loader=training_loader, task=task, train=True)     # For the given epoch, train/validate model over all the batches in the given data loader
            self.tracker.update({'training': metrics})                  # Updates the tracker with given metric values of interest
            self.validation(validation_loader, task)                    # Runs the validation for epoch 1 to visualize and minimize overfitting

            self.schedulers[task].step()    # Once the training epoch is completed, we increment the scheduler.

            if epoch == 0:
                self._registerCheckpoint_(epoch=epoch, task=task)

            elif self.checkpoint.logs[task + '_loss'] > self.tracker.logs['validation'][task][epoch]:
                self._registerCheckpoint_(epoch=epoch, task=task)

            self.progressbar.update({'task': task,
                                     'epoch': epoch,
                                     'loss': self.tracker.logs['validation'][task][epoch]
                                     })
            
            # early_stop(self.tracker.logs['validation'], epoch=epoch) 
            # if early_stop.stopping:
            #     print(f"\n------- Early Stopping Trigged at Epoch {epoch} | Start Evaluation -------")
            #     break
          
            self.progressbar.visuals()                                  # Update the progress bar visuals after each epoch

    def validation(self, validation_loader, task):
        """
        Validates network using validation dataset
        -----------
        Parameters:
        validation_loader - DataLoader
            pytorch dataloader containing validation data that the network has not seen
        epoch - int
            current training epoch of the model
        --------
        Returns:
        """
        with torch.no_grad():
            metrics = self._batches_(loader=validation_loader, task=task, train=False)
            self.progressbar.update({'validation': metrics})
            self.tracker.update({'validation':metrics})

    def evaluate(self, evaluation_loader, task) -> pd.DataFrame:
        """
        Evaluates model using testing dataset
        -----------
        Parameters:
        --------
        Returns:

        TODO: Add controls that only allow specific tasks to be done
        """
        self._loadCheckpoint_()

        with torch.no_grad():
            targets = []
            softpred = []
            ids = []
            ssim_scores = []

            for i, data in enumerate(evaluation_loader):
                images = data['image'].to(self.device,  dtype = torch.float)

                out = self.model.inference(images, task, region=self.region)

                if task=='dx':
                    for i in range(len(images)):
                        diagnosis_labels = data['class_label'].to(self.device)                  # Getting the class labels for the given batch
                        targets.append(diagnosis_labels[i,1].cpu().squeeze().numpy())           # Appending the target labels
                        softpred.append(out['classification'][i,1].cpu().squeeze().numpy())
                        ids.append(data['id'][i])
                else:
                    ssmi = reconstructions(recs=out['reconstruction'], 
                                            masked=out['masked_images'],
                                            imgs=images,
                                            ids=data['id'],
                                            cfg=self.cfg,
                                            tlearn=self.tlearn,
                                            region= self.region
                                            )
                    ssim_scores.append(ssmi)
                        
        if task != 'dx':    
            ssim_scores = np.concatenate(ssim_scores)
            return {
                'mean_ssim': np.mean(ssim_scores)
                }
        
        else:
            fps, tps, threshold = roc_curve(targets[:], softpred[:])
            auc_score = auc(fps, tps)
            youden_value = (tps + (1-fps)) - 1 
            youden_index = youden_value.argmax() 
            
            print(f"Sensitivity: {tps[youden_index]}, Specificity: {1-fps[youden_index]}, AUC: {auc_score}, Youden Index: {youden_value[youden_index]}, Threshold: {threshold[youden_index]}")

            prediction = [1 if x > threshold[youden_index] else 0 for x in softpred]
            save_model_outcomes(prediction, targets, ids, self.cfg, self.tlearn, self.region)

            performance = get_dor(prediction, targets)
            conf_matrix = confusion_matrix(prediction, targets)
            
            performance['auc'] = auc_score
            
            return performance
    
    def _registerCheckpoint_(self, epoch:int, task:str):
        """
        TODO: Move to its own file just like tracker & scheduler
        Creates a temporary copy of the current best iteration of the network for evaluation using validation performance metrics
        -----------
        Parameters:
        epoch - int
            epoch of the model
        """
        taskloss = task + '_loss'
        self.checkpoint.update({
                    # 'fold': fold,
                    'epoch': epoch,
                    taskloss: self.tracker.logs['validation'][task][epoch],
                    'state_dicts':{
                        'model': self.model.state_dict(),
                        'ae_optim': self.optimizers['reconstruction'].state_dict(),
                        'dx_optim': self.optimizers['dx'].state_dict(),
                        },
                    'schedulers':{
                          'ae': self.schedulers['reconstruction'].state_dict(),
                          'dx': self.schedulers['dx'].state_dict(),
                        }
                    })

    def _loadCheckpoint_(self):
        # Load the best checkpoint
        best_checkpoint = self.checkpoint.load()
        self.model.load_state_dict(best_checkpoint['state_dicts']['model'])
        self.optimizers['reconstruction'].load_state_dict(best_checkpoint['state_dicts']['ae_optim'])
        self.optimizers['dx'].load_state_dict(best_checkpoint['state_dicts']['dx_optim'])
        self.schedulers['reconstruction'].load_state_dict(best_checkpoint['schedulers']['ae'])
        self.schedulers['dx'].load_state_dict(best_checkpoint['schedulers']['dx'])

    def _savemodel_(self):
        self.checkpoint.save()
