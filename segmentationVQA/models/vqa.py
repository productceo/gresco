"""VQA model implementation.
"""

from .EncoderCNN import EncoderCNN
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQAModel(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Properties:
        encoder_cnn: object of EncoderCNN.
        encoder_rnn: object of EncoderRNN.
        decoder: object of DecoderRNN.
        decode_function (func, optional): function to generate symbols from
            output hidden states (default: F.log_softmax).
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 num_layers=1, rnn_cell='lstm', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False,
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
            use_attention: Whether to use attention with decoder.
        """
        super(VQAModel, self).__init__()
        self.encoder_cnn = EncoderCNN(hidden_size)
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      variable_lengths=True)
        self.num_layers = num_layers
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
                  list(self.encoder_rnn.parameters()) +
                  list(self.encoder_cnn.cnn.fc.parameters()) +
                  list(self.encoder_cnn.bn.parameters()))
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, image, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Passes the image and the question through a VQA model and generates answers.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        feature = self.encoder_cnn(image)
        # Initialize hidden state for encoder rnn.
        h0 = feature.unsqueeze(0)
        h0 = feature.expand(self.bidirectional_multiplier * self.num_layers,
                            h0.size(1), h0.size(2)).contiguous()
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            h0 = (h0, h0)
        encoder_outputs, encoder_hidden = self.encoder_rnn(
            input_variable, input_lengths, h0)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def predict(self, image, input_variable, input_lengths=None,
                target_variable=None, teacher_forcing_ratio=0,
                decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        outputs, _, _ = self.forward(  # List(max_len * Tensor(batch * vocab_size))
            image, input_variable, input_lengths=input_lengths,
            target_variable=target_variable,
            teacher_forcing_ratio=teacher_forcing_ratio,
            decode_function=decode_function)
        # Take argmax for each timestep
        outputs = [o.max(1)[1] for o in outputs]  # List(max_len * Tensor(batch))

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs

    def vqa_encode(self, image, input_variable, input_lengths=None):
        """Passes the image and the question through a VQA model and generates answers.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        feature = self.encoder_cnn(image)
        # Initiali hidden state for encoder rnn
        image_encoding = feature.unsqueeze(0)
        h0 = feature.unsqueeze(0)
        h0 = feature.expand(self.bidirectional_multiplier * self.num_layers,
                            h0.size(1), h0.size(2)).contiguous()
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            h0 = (h0, h0)
        encoder_outputs, encoder_hidden = self.encoder_rnn(
            input_variable, input_lengths, h0)
        result = torch.cat((encoder_hidden[0].squeeze(0), image_encoding.squeeze(0)), 1)
        return result
