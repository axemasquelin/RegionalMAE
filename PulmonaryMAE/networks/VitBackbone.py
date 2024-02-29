# utf-8
""" MIT License """
'''
    Project: PulmonaryMAE
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from torchvision.models import ViT_B_16, ViT_B_16_Weights
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np
from PulmonaryMAE.utils import utils
# ---------------------------------------------------------------------------- #

class Classifier(nn.Module):
    """
    Classifier
    """
    # TODO: Play with Dropout, should aim to have it at ~ 0% - 10% based on existing literature.
    def __init__(self, embeddim: int, n_classes: int) -> None:
        super(Classifier,self).__init__()

        self.model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embeddim, 500),
            nn.ReLU(inplace = True),
            nn.Dropout(0.1),
            nn.Linear(500, 61),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(61, n_classes)
        )
    def forward(self, x):
        x = torch.flatten(x,1)
        y = self.model(x)
        
        return y


class Custom_ViT_B_16(nn.Module):
    def __init__(self, pretrain, n_class=2, img_size=64):
        super(Custom_ViT_B_16,self).__init__()
        if pretrain:
            encoder_weights = ViT_B_16_Weights
        else:
            encoder_weights = None
        
        self.encoder = ViT_B_16(weights = encoder_weights)
        self.classifier = nn.Sequential(
            nn.Lienar(chn_in = 768, chn_out = 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(chn_in=512, chn_out = n_class)
        )

    def forward(self,x):
        '''
        Args:
            x: Torch.Tensor of input image. 
        '''
        x = self.encoder(x)
        embedding = torch.flatten(x,1)
        y = self.classifier(embedding)
        
        return {'embedding':embedding,
                'class': y}