import torch.nn as nn
import torchvision.models as models


class SpatialEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SpatialEncoder, self).__init__()
        self.cnn = models.resnet152(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.conv = nn.Sequential(*list(self.cnn.children())[:-3])
        self.mlp = nn.Linear(1024, hidden_size)

    def init_weights(self):
        self.mlp.weight.data.normal_(0.0, 0.02)
        self.mlp.bias.data.fill_(0)

    def forward(self, img):
        features = self.conv(img)
        features = features.view(features.shape[0], features.shape[1], 196)
        features = features.permute(0, 2, 1)
        output = self.mlp(features)
        return output
