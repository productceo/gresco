"""Simple CNN-RNN model that uses measures engagement of image and question.
"""

from .EncoderCNN import EncoderCNN
from .EncoderRNN import EncoderRNN
from .mlp import MLP

import torch
import torch.nn as nn


class EngagementModel(nn.Module):
    """Encodes image and question using CNN and RNN and outputs
       engagment score using an MLP.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 num_rnn_layers=1, num_layers=1, bidirectional=False,
                 rnn_cell='LSTM', input_dropout_p=0, dropout_p=0):
        """Constructor for EngagementModel.

        Args:
            vocab_size: The vocabulary size of our model.
            max_len: The maximum length questions to generate.
            hidden_size: The size of the hidden dimensions of the RNN.
            num_rnn_layers: Number of layers in the RNN.
            num_layers: Number of layers in the MLP.
            rnn_cell: LSTM or GRU or RNN.
            bidirectional: Boolean for whether the RNN is bidirectional.
            input_dropout_p: dropout percentage for the RNN inputs.
            dropout_p: dropout percentage for the RNN hiddens.
        """
        super(EngagementModel, self).__init__()
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_rnn_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      variable_lengths=True)
        self.num_directions = 2 if bidirectional else 1
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.mlp = MLP(self.num_directions * hidden_size, hidden_size, 1,
                       num_layers=num_layers)

    def params_to_train(self):
        params = list(self.encoder_rnn.parameters())\
                 + list(self.encoder_cnn.cnn.fc.parameters())\
                 + list(self.encoder_cnn.bn.parameters())\
                 + list(self.mlp.params_to_train())
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()

    def forward(self, images, questions, lengths):
        h0 = self.encoder_cnn(images)
        h0 = h0.unsqueeze(0)
        h0 = h0.expand(self.num_directions * self.num_rnn_layers,
                       h0.size(1),
                       h0.size(2)).contiguous()
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            h0 = (h0, h0)
        outputs, _ = self.encoder_rnn(questions, lengths, h0)
        masks = torch.Tensor(lengths).type_as(outputs) - 1
        masks = masks.long().unsqueeze(1).unsqueeze(2).expand(outputs.size())
        outputs = outputs.gather(1, masks)
        outputs = self.mlp(outputs[:, 0, :])
        return outputs
