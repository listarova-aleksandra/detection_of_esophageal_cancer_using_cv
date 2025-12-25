import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=1, pretrained=True):
    model = models.resnet18(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model