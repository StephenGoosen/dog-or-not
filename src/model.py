# model.py

import torch.nn as nn
import torchvision.models as models

import config

num_classes = config.Model.num_classes
pretrained_weights = config.Model.pretrained_weights
trainable = config.Model.trainable

'''
This file contains:
    The MobileNetV2 architecture from torchvision.models. It uses the ImageNet pretrained weights.
    The 'trainable' argument which allows training of the pretrained weights but requires longer training time.
    The custom classifier which uses 2 fully connected layers and outputs to num_classes.
    The model_get function which is used to compile the model architecture.
'''

class Classifier(nn.Module): # Custom Classifier function with two fully connected layers
    def __init__(self, in_features, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 1280) 
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
class MobileNetV2_Model(nn.Module):
    def __init__(self, num_classes, trainable=trainable):
        super(MobileNetV2_Model, self).__init__()

        self.model_base = models.mobilenet_v2(weights=pretrained_weights) # Uses pretrained mobilenet_v2 weights

        for params in self.model_base.parameters():
            params.requires_grad = trainable # Whether pre-trained weights can be adjusted through the training steps

        self.classifier = Classifier(1000, num_classes) # Replaces classifier with the custom Classifier() function

    def forward(self, x):
        x = self.model_base(x)
        x = self.classifier(x)
        return x
    
def model_get(num_classes, device, trainable=False): 
    model = MobileNetV2_Model(num_classes, trainable)
    model.to(device)
    return model
