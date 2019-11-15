"""VQG model.
"""

import torch.nn as nn
import torch.nn.functional as F

from .EncoderCNN import EncoderCNN
from .DecoderRNN import DecoderRNN


class VQGModel(nn.Module):
    """Simple CNN-RNN model to generate questions from an image.

    Components:
        encoder (EncoderCNN): object of EncoderCNN.
        decoder (DecoderRNN): object of DecoderRNN.
        decode_function (func, optional): function to generate
            symbols from output hidden states (default: F.log_softmax).
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id, n_layers=1, rnn_cell='lstm',
                 bidirectional=False, input_dropout_p=0, dropout_p=0,
                 use_attention=False):

        super(VQGModel, self).__init__()
        self.encoder = EncoderCNN(hidden_size)
        self.decoder = DecoderRNN(vocab_size, max_len, hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=1,
                                  rnn_cell=rnn_cell,
                                  bidirectional=bidirectional,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  use_attention=use_attention)

    def params_to_train(self):
        params = list(self.decoder.parameters())\
                 + list(self.encoder.cnn.fc.parameters())\
                 + list(self.encoder.bn.parameters())
        return params

    def flatten_parameters(self):
        self.decoder.rnn.flatten_parameters()

    def forward(self, image, target=None,
                teacher_forcing_ratio=0,
                decode_function=F.log_softmax):
        feature = self.encoder(image)
        # TODO: unsqueeze(num_layers * num_directions)
        feature = feature.unsqueeze(0)
        if self.decoder.rnn_cell is nn.LSTM:
            feature = (feature, feature)
        result = self.decoder(inputs=target,
                              encoder_hidden=feature,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
