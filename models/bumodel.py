"""Bottom up Attention based VQA.
"""

from .bottom_up_attention import NewAttention
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class BUModel(nn.Module):

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False,
                 answer_max_len=None, q_hidden_size=512,
                 v_hidden_size=2048, embedding=None):
        super(BUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_embed_size = vocab_embed_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        answer_max_len = answer_max_len if answer_max_len is not None else max_len
        self.encoder_rnn  = EncoderRNN(vocab_size, max_len, hidden_size,
                                       input_dropout_p=input_dropout_p,
                                       dropout_p=dropout_p,
                                       n_layers=num_layers,
                                       bidirectional=bidirectional,
                                       rnn_cell=rnn_cell,
                                       vocab_embed_size=vocab_embed_size,
                                       variable_lengths=True,
                                       embedding=embedding)
        self.bu_attention = NewAttention(v_hidden_size,
                                         self.bidirectional_multiplier*hidden_size,
                                         hidden_size)
        self.decoder = DecoderRNN(vocab_size, answer_max_len,
                                  self.bidirectional_multiplier*hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  bidirectional=bidirectional,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  use_attention=use_attention)

    def params_to_train(self):
        params = (list(self.decoder.parameters()) +
                  list(self.encoder_rnn.parameters()) +
                  list(self.bu_attention.parameters()))
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, visual_features, questions, answers=None, qlengths=None,
                teacher_forcing_ratio=0.5,
                decode_function=F.log_softmax):

        # Get question embedding -> [BATCH_SIZE, FEATURE_SIZE]
        _, encoder_hidden = self.encoder_rnn(questions, qlengths, None)
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            encoder_hidden = encoder_hidden[0]
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous()

        if self.bidirectional_multiplier == 2:
            encoder_hidden = torch.cat((encoder_hidden[:, 0], encoder_hidden[:, -1]), dim=1)
        else:
            encoder_hidden = encoder_hidden[:, -1]

        # Apply top down attention
        attended_hidden = self.bu_attention(visual_features, encoder_hidden)
        attended_hidden = attended_hidden.unsqueeze(0).repeat(
                self.bidirectional_multiplier*self.num_layers, 1, 1)

        # Decode the answer.
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            memory = torch.zeros(attended_hidden.size()).cuda()
            attended_hidden = (attended_hidden, memory)

        result = self.decoder(inputs=answers,
                              encoder_hidden=attended_hidden,
                              encoder_outputs=None,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
