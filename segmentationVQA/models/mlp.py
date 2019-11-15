"""A simple MLP.
"""

from torch import nn
from torch.nn.utils.weight_norm import weight_norm

import math


class MLP(nn.Module):
    """A simple MLP.
    """

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, dropout_p=0.0, w_norm=False):
        """Constructor for MLP.

        Args:
            input_size: The number of input dimensions.
            hidden_size: The number of hidden dimensions for each layer.
            num_classes: The size of the output.
            num_layers: The number of hidden layers.
            dropout_p: Dropout probability.
            weight_norm: Whether to normalize the Linear weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.w_norm = w_norm
        layers = []
        for i in range(num_layers):
            idim = hidden_size
            odim = hidden_size
            if i == 0:
                idim = input_size
            if i == num_layers-1:
                odim = num_classes
            fc = nn.Linear(idim, odim)
            fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
            fc.bias.data.fill_(0)
            if w_norm:
                fc = weight_norm(fc, dim=None)
            layers.append(fc)
            if i != num_layers-1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_p))
        self.layers = nn.Sequential(*layers)

    def params_to_train(self):
        return self.layers.parameters()

    def forward(self, x):
        """Propagate through all the hidden layers.

        Args:
            x: Input of self.input_size dimensions.
        """
        out = self.layers(x)
        return out
