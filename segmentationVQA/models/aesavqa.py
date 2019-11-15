"""Contains model that uses Stacked Attention for VQA.
"""

from .EncoderCNN import SpatialResnetEncoder
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .stacked_attention import StackedAttention
from .mlp import MLP

import torch
import torch.nn as nn


class AESAVQAModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0.0, dropout_p=0.0, num_att_layers=2,
                 att_ff_size=512, mlp_num_layers=2, mlp_output=100,
                 dropout_mlp=0.0, embedding=None, has_decoder=False,
                 answer_max_len=None):
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
        super(AESAVQAModel, self).__init__()
        self.spatial_encoder_cnn = SpatialResnetEncoder(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      vocab_embed_size=vocab_embed_size,
                                      variable_lengths=True,
                                      embedding=embedding)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_embed_size = vocab_embed_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        self.san = nn.ModuleList([StackedAttention(
                input_image_dim=hidden_size,
                input_question_dim=(self.bidirectional_multiplier *
                                    num_layers * hidden_size),
                hidden_dim=att_ff_size,
                dropout=True if (dropout_p > 0.0) else False)] * num_att_layers)

        self.encoder_mlp = MLP(self.bidirectional_multiplier * num_layers * hidden_size,
                               hidden_size,
                               mlp_output,
                               num_layers=mlp_num_layers,
                               dropout_p=dropout_mlp)

        self.has_decoder = has_decoder
        if has_decoder:
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
        params = (list(self.encoder_rnn.parameters()) +
                  list(self.encoder_mlp.parameters()) +
                  list(self.spatial_encoder_cnn.fc.parameters()) +
                  list(self.san.parameters()))
        if self.has_decoder:
            params += list(self.decoder.parameters())
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()

    def encode(self, features, questions, qlengths=None):
        """Passes the image and the question through a VQA model and embeds answers.

        Args:
            features: Batch of image features Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.

        Returns:
            Embedded answer
        """
        # features is (N * 196 * HIDDEN_SIZE)
        features = self.spatial_encoder_cnn(features)
        features = self.dropout(features)

        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        _, encoder_hidden = self.encoder_rnn(questions, qlengths, None)

        # Reshape encoder_hidden ((BIDIRECTIONAL x NUM_LAYERS * N) * HIDDEN_SIZE).
        first_dim = self.bidirectional_multiplier * self.num_layers
        batch_size = features.size(0)
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            encoder_hidden = encoder_hidden[0]
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous()

        # Reshape encoder_hidden (N * (BIDIRECTIONAL x NUM_LAYERS) * HIDDEN_SIZE).
        encoder_hidden = encoder_hidden.view(batch_size, first_dim * self.hidden_size)

        # Pass the features through the stacked attention network.
        attended_hidden = encoder_hidden
        for att_layer in self.san:
            attended_hidden = att_layer(features, attended_hidden)
        attended_hidden = self.dropout(attended_hidden)

        # use attended_hidden
        embeddings = self.encoder_mlp(attended_hidden)
        return embeddings

    def decode(self, embeddings, answers=None,
               teacher_forcing_ratio=0.0):
        """Decodes the embeddings into answers.

        Args:
            embeddings: Tensor of embeddings.

        Returns:
            The generated answers.
        """
        embeddings = embeddings.unsqueeze(0).repeat(
                self.bidirectional_multiplier*self.num_layers, 1, 1)
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            embeddings = (embeddings,
                          torch.zeros(embeddings.size()).cuda())
        result = self.decoder(inputs=answers,
                              encoder_hidden=embeddings,
                              encoder_outputs=None,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def forward(self, features, questions, qlengths=None, answers=None,
                teacher_forcing_ratio=0.0):
        """Encodes question and image features and then decodes the answers.

        Args:
            features: Batch of image features Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.
            answers: The answers to generate.
            teacher_forcing_ratio: Whether to use the outputs
                to supervise the decoder.

        Returns:
            Generated answers.
        """
        outputs = self.encode(features, questions, qlengths=qlengths)
        if self.has_decoder:
            outputs = self.decoder(outputs, answers=answers,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs
