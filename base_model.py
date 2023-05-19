import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet


class BaseModel():
    def __init__(self):
        self.model = alexnet(pretrained=True)
        self.num_layers = len(self.model.classifier)
        self.num_features = self.model.classifier[self.num_layers - 1].in_features

    def add_classifier(self, layer_index, classifier):
        self.model.classifier[layer_index - 1] = classifier
