"""Genearates a representation for an image input.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self, output_size, resnet='resnet50'):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = getattr(models, resnet)(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
        """
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors.
        """
        features = self.cnn(images)
        output = self.bn(features)
        return output


class SpatialVGGEncoder(nn.Module):
    """Generates a 3D representation for an image.
    """

    def __init__(self, output_size):
        """Loads the pretrained VGG-16 model and uses it's conv features.
        """
        super(SpatialVGGEncoder, self).__init__()
        self.cnn = models.vgg16(pretrained=True).features
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(512, output_size)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
        """
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        # N * 3 * 224 * 224 -> N * 512 * 7 * 7
        batch_size = images.size(0)
        images = self.cnn(images)
        hidden_size = images.size(1)
        spatial_size = images.size(2)

        # N * 512 * 7 * 7 -> N * 512 * 49 -> N * 49 * 512
        images = images.view(batch_size, hidden_size, spatial_size**2)
        images = images.transpose(1, 2)

        # N * 196 * 512 -> N * 196 * self.output_size
        image_embeddings = torch.tanh(self.fc(images))
        return image_embeddings


class SpatialResnetEncoder(nn.Module):
    """Resnet Model without the fc layer.
    """

    def __init__(self, output_size, delete_starting="res3", feats_available = True):
        """Loads pretrained Resnet50 and removes all non-spatial layers at the end.
        When delete_starting == 'fc', output size is (BATCH_SIZE, 2048, 1, 1).
        When delete_starting == 'res3', output size is (BATCH_SIZE, 1024, 14, 14).
        """
        super(SpatialResnetEncoder, self).__init__()
        if not feats_available:
            # create resnet
            resnet = models.resnet101(pretrained=True)
            if delete_starting == "fc":
                modules = list(resnet.children())[:-1] # delete the last fc layer.
            if delete_starting == "res3":
                modules = list(resnet.children())[:-3]
            else:
                modules = list(resnet.children())[:-3]
            self.resnet = nn.Sequential(*modules)
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(1024, output_size)
        self.feats_available = feats_available

    def forward(self, features):
        if not self.feats_available:
            features = self.resnet(features)
        batch_size = features.size(0)
        hidden_size = features.size(1)
        spatial_size = features.size(2)
        features = features.view(batch_size, hidden_size, spatial_size**2)
        features = features.transpose(1, 2)
        features = torch.tanh(self.fc(features))
        return features


class CompleteResnetEncoder(nn.Module):
    """Resnet model that returns both spatial and final fc layer features.
    """

    def __init__(self, output_size):
        """Loads pretrained Resnet50 and removes all non-spatial layers at the end.
        """
        super(CompleteResnetEncoder, self).__init__()
        # create resnet
        resnet = models.resnet50(pretrained=True)
        spatial = list(resnet.children())[:-3]
        final = list(resnet.children())[-3:-1]
        self.spatial = nn.Sequential(*spatial)
        self.final = nn.Sequential(*final)
        self.spatial_fc = nn.Linear(1024, output_size)
        self.fc = nn.Linear(2048, output_size)
        for param in self.spatial.parameters():
            param.requires_grad = False
        for param in self.final.parameters():
            param.requires_grad = False
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
        """
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        self.spatial_fc.weight.data.normal_(0.0, 0.02)
        self.spatial_fc.bias.data.fill_(0)

    def forward(self, images):
        """Featurizes the image.

        Args:
            images: Batch of images.

        Returns:
            Spatial output size is
                (BATCH_SIZE, 2048, 1, 1) -> (BATCH_SIZE, OUTPUT_SIZE).
            FC output size is
                (BATCH_SIZE, 1024, 14, 14) -> (BATCH_SIZE, 196, OUTPUT_SIZE).
        """
        # Forward pass through Resnet.
        spatial_features = self.spatial(images)
        features = self.final(spatial_features)

        # Calculate spatial features.
        batch_size = images.size(0)
        hidden_size = spatial_features.size(1)
        spatial_size = spatial_features.size(2)
        spatial_features = spatial_features.view(
                batch_size, hidden_size, spatial_size**2)
        spatial_features = spatial_features.transpose(1, 2)
        spatial_features = torch.tanh(self.spatial_fc(spatial_features))

        # Calculate final features.
        features = features.squeeze(3).squeeze(2)
        features = self.bn(torch.tanh(self.fc(features)))
        return features, spatial_features
