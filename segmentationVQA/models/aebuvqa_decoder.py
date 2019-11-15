"""Contains model that uses Bottom Up Attention AE for VQA.
"""

from .bottom_up_attention import NewAttention
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .mlp import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


class AEBUVQA_DECODERModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0.0, dropout_p=0.0, q_hidden_size=512,
                 v_hidden_size=2048, mlp_num_layers=2, mlp_output=256,
                 dropout_mlp=0.5, embedding=None, answer_max_len=None):

        """Constructor for VQAModel.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
            embedding: Pretrained word embeddings.
        """
        super(AEBUVQA_DECODERModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_embed_size = vocab_embed_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      vocab_embed_size=vocab_embed_size,
                                      variable_lengths=True,
                                      embedding=embedding)
        self.bu_attention = NewAttention(v_hidden_size,
                                         self.bidirectional_multiplier * hidden_size,
                                         hidden_size)
        self.encoder_mlp = MLP(hidden_size,
                               hidden_size,
                               mlp_output,
                               num_layers=mlp_num_layers,
                               dropout_p=dropout_mlp)
        if answer_max_len is None:
            answer_max_len = max_len
        self.decoder = DecoderRNN(vocab_size, answer_max_len,
                                  self.bidirectional_multiplier*mlp_output,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  bidirectional=bidirectional,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p,
                                  use_attention=False)

    def params_to_train(self):
        params = (list(self.decoder.parameters()) +
                  list(self.encoder_rnn.parameters()) +
                  list(self.encoder_mlp.parameters()) +
                  list(self.bu_attention.parameters()))
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()

    def forward(self, visual_features, questions, answers=None,
                qlengths=None, decode_function=F.log_softmax,
                teacher_forcing_ratio=0.0):
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

        # use attended_hidden
        caption_embeddings = self.encoder_mlp(attended_hidden)
        caption_embeddings_decoder = caption_embeddings.unsqueeze(0).repeat(
                self.bidirectional_multiplier*self.num_layers, 1, 1)

        # Decode the answer.
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            attended_hidden = (caption_embeddings_decoder,
                               torch.zeros(caption_embeddings_decoder.size()).cuda())
        result = self.decoder(inputs=answers,
                              encoder_hidden=attended_hidden,
                              encoder_outputs=None,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return caption_embeddings, result
