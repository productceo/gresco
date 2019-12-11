"""Contains model that uses Stacked Attention for VQA classification.
"""

from .EncoderCNN import SpatialResnetEncoder
from .EncoderRNN import EncoderRNN
from .stacked_attention import StackedAttention

import torch.nn as nn
import torch.nn.functional as F


class MultiSAVQAModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size, vocab_embed_size,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, num_att_layers=2,
                 att_ff_size=512):
        """Constructor for VQAModel.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            vocab_embed_size: Number of dimensions of RNN embedding.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
        """
        super(MultiSAVQAModel, self).__init__()
        self.spatial_encoder_cnn = SpatialResnetEncoder(hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.encoder_rnn  = EncoderRNN(vocab_size, max_len, hidden_size,
                                       input_dropout_p=input_dropout_p,
                                       dropout_p=dropout_p,
                                       n_layers=num_layers,
                                       bidirectional=bidirectional,
                                       rnn_cell=rnn_cell,
                                       vocab_embed_size=vocab_embed_size,
                                       variable_lengths=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_embed_size = vocab_embed_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        question_dim = self.bidirectional_multiplier * num_layers * hidden_size
        self.san = nn.ModuleList([StackedAttention(
                input_image_dim=hidden_size,
                input_question_dim=question_dim,
                hidden_dim=att_ff_size)] * num_att_layers)
        self.classifier = nn.Linear(question_dim, 1000)

    def params_to_train(self):
        params = (list(self.classifier.parameters()) +
                  list(self.encoder_rnn.parameters()) +
                  list(self.spatial_encoder_cnn.fc.parameters()) +
                  list(self.san.parameters()))
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()

    def forward(self, features, questions, qlengths=None):
        """Passes the image and the question through a VQA model
        and classifies answers.

        Args:
            images: Batch of image Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.

        Returns:
            The output scores for all classes.
        """

        # features is (N * 196 * HIDDEN_SIZE)
        features = self.spatial_encoder_cnn(features)
        features = self.dropout(features)

        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        encoder_outputs, encoder_hidden = self.encoder_rnn(
            questions, qlengths, None)

        # Reshape encoder_hidden (N * (BIDIRECTIONAL x NUM_LAYERS) * HIDDEN_SIZE).
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

        # Classify the answer.
        result = self.classifier(attended_hidden)
        return result

    def predict(self, features, questions, qlengths=None,
                answers=None, teacher_forcing_ratio=0,
                decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            features: Batch of image features.
            questions: Batch of question features.
            qlengths: List of question lengths.

        Returns:
            A tensor with BATCH_SIZE X 1 where each element is the index
            into the vocab word.
        """
        outputs, _, _ = self.forward(features, questions, qlengths=qlengths)

        # Take argmax for each timestep
        outputs = outputs.max(1)[1]
        return outputs
