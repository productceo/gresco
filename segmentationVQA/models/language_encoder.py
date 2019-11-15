"""Contains model that uses Stacked Attention for VQA.
"""

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class LosslessTripletLoss(nn.Module):
    """
    Lossless Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, N=1024, beta=1024, epsilon=1e-8):
        super(LosslessTripletLoss, self).__init__()
        self.epsilon = epsilon
        self.N = N
        self.beta = beta

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        pos_dist = -torch.log(-torch.div(distance_positive, self.beta) + 1 + self.epsilon)
        neg_dist = -torch.log(-torch.div((self.N-distance_negative), self.beta) + 1 + self.epsilon)
        losses = pos_dist + neg_dist
        return losses.mean() if size_average else losses.sum()


class LAEModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False,
                 answer_max_len=None, num_att_layers=2, att_ff_size=512):
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
            use_attention: Whether to use attention with decoder.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
        """
        super(LAEModel, self).__init__()
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      variable_lengths=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
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

    def params_to_train(self):
        params = (list(self.decoder.parameters()) +
                  list(self.encoder_rnn.parameters()))
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

    def forward(self, anchor, positive=None, negative=None, slengths=None,
                teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        """Passes the sequence through the autoencoder.

        Args:
            sequences: Batch of triplet sequence Variables.
            slengths: List of question lengths.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - encoder_outputs: outputs from the encoder.
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        if positive is not None:
            anchor_encoder_outputs, anchor_encoder_hidden = self.encoder_rnn(anchor, slengths[0], None)
            _, positive_encoder_hidden = self.encoder_rnn(positive, slengths[1], None)
            _, negative_encoder_hidden = self.encoder_rnn(negative, slengths[2], None)
        else:
            anchor_encoder_outputs, anchor_encoder_hidden = self.encoder_rnn(anchor, slengths, None)

        # Decode the answer.
        decoder_outputs = self.decoder(inputs=anchor,
                                       encoder_hidden=anchor_encoder_hidden,
                                       encoder_outputs=anchor_encoder_outputs,
                                       function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)
        if positive is not None:
            return (anchor_encoder_hidden, positive_encoder_hidden, negative_encoder_hidden), decoder_outputs

        return (None), decoder_outputs

    def predict(self, sequences, slengths=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            sequences: Batch of sequence Variables.
            slengths: List of question lengths.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        _, decoder_outputs = self.forward(  # List(max_len * Tensor(batch * vocab_size))
            sequences, slengths=slengths,
            teacher_forcing_ratio=teacher_forcing_ratio,
            decode_function=decode_function)

        # Take argmax for each timestep
        outputs, _, _ = decoder_outputs
        outputs = [o.max(1)[1] for o in outputs]  # List(max_len * Tensor(batch))

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs
