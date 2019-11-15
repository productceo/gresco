"""Genearates a representation for an image input.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SpatialResnetEncoder(nn.Module):
    """Resnet Model without the fc layer.
    """

    def __init__(self, num_classes=2, replace_stride_with_dilation=(8, 8, 8)):
        super(SpatialResnetEncoder, self).__init__()
        # create resnet
        resnet = models.resnet101(pretrained=True,
                                  replace_stride_with_dilation=replace_stride_with_dilation)
        self.inplanes = resnet.inplanes
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Conv2d(self.inplanes, num_classes, 1)

    def forward(self, features):
        batch_size = features.size(0)
        hidden_size = features.size(1)
        spatial_size = features.size(2)
        features = features.view(batch_size, hidden_size, spatial_size**2)
        features = features.transpose(1, 2)
        features = torch.tanh(self.fc(features))
        return features
