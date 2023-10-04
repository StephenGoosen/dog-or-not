import torch
import torch.nn as nn
import torchvision.models as models

import utils
import config

num_classes = config.Model.num_classes
pretrained_weights = config.Model.pretrained_weights
trainable = config.Model.trainable

'''
This file contains:
    The MobileNetV2 architecture from torchvision.models. It uses the ImageNet pretrained weights.
    The use of the 'trainable' argument allows flexibility
    The custom classifier that uses 2 fully connected layers and outputs to num_classes.
    The model_get function can be used to create a model.
'''
class Classifier(nn.Module):
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

        self.model_base = models.mobilenet_v2(weights=pretrained_weights)

        for params in self.model_base.parameters():
            params.requires_grad = False

        self.classifier = Classifier(1000, num_classes) #Replaces classifier with the custom Classifier()

    def forward(self, x):
        x = self.model_base(x)
        x = self.classifier(x)
        return x
    
def model_get(num_classes, device, trainable=trainable):
    model = MobileNetV2_Model(num_classes)
    model.to(device)
    return model
