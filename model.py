import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

## Defining the NN architecture

# Load the pretrained VGG16 architechture from pytorch
vgg16 = models.vgg16(pretrained=True)

penultimate_layer = nn.Linear(4096, 512)
last_layer = nn.Linear(512, 2)

#Changing layers
vgg16.classifier[3] = penultimate_layer
vgg16.classifier[6] = last_layer
