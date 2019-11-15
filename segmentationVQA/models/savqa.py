"""Contains model that uses Stacked Attention for VQA.
"""

from .EncoderCNN import SpatialResnetEncoder
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .stacked_attention import StackedAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAVQAModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False,
                 answer_max_len=None, num_att_layers=2, att_ff_size=512,
                 embedding=None):
        """Constructor for VQAModel.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            vocab_embed_size: Number of dimensions of RNN embedding.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            use_attention: Whether to use attention with decoder.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
            embedding: Pretrained Embedding weights.
        """
        super(SAVQAModel, self).__init__()
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
        answer_max_len = answer_max_len if answer_max_len is not None else max_len
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
        self.san = nn.ModuleList([StackedAttention(
                input_image_dim=hidden_size,
                input_question_dim=(self.bidirectional_multiplier *
                                    num_layers * hidden_size),
                hidden_dim=att_ff_size)] * num_att_layers)

    def params_to_train(self):
        params = (list(self.decoder.parameters()) +
                  list(self.encoder_rnn.parameters()) +
                  list(self.spatial_encoder_cnn.fc.parameters()) +
                  list(self.san.parameters()))
        # Don't train the embedding weights.
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def modify_hidden(self, func, hidden):
        """Applies the function func to the hidden representation.

        This method is useful because some RNNs like LSTMs have a tuples.

        Args:
            func: A function to apply to the hidden representation.
            hidden: A RNN (or LSTM or GRU) representation.

        Returns:
            func(hidden)
        """
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            return (func(hidden[0]), func(hidden[1]))
        return func(hidden)

    def forward(self, features, questions, qlengths=None, answers=None,
                teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        """Passes the image and the question through a VQA model and generates answers.

        Args:
            images: Batch of image Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.
            answers: Batch of answer Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """

        # features is (N * 196 * HIDDEN_SIZE)
        features = self.spatial_encoder_cnn(features)
        features = self.dropout(features)

        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        encoder_outputs, encoder_hidden = self.encoder_rnn(
            questions, input_lengths=qlengths)

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

        # Reshape encoder_hidden (N * (BIDIRECTIONAL x NUM_LAYERS) * HIDDEN_SIZE).
        attended_hidden = attended_hidden.view(batch_size, first_dim, self.hidden_size)

        # Reshape encoder_hidden ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        attended_hidden = attended_hidden.transpose(0, 1).contiguous()

        # Decode the answer.
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            attended_hidden = (attended_hidden,
                               torch.zeros(attended_hidden.size()).cuda())
        result = self.decoder(inputs=answers,
                              encoder_hidden=attended_hidden,
                              encoder_outputs=encoder_outputs,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def predict(self, features, questions, qlengths=None,
                answers=None, teacher_forcing_ratio=0,
                decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            images: Batch of image Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.
            answers: Batch of answer Variables.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        outputs, _, _ = self.forward(  # List(max_len * Tensor(batch * vocab_size))
            features, questions, qlengths=qlengths, answers=answers,
            teacher_forcing_ratio=teacher_forcing_ratio,
            decode_function=decode_function)

        # Take argmax for each timestep
        outputs = [o.max(1)[1] for o in outputs]  # List(max_len * Tensor(batch))

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs
