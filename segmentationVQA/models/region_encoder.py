"""Contains model that encodes sequences based on regions features.
"""

from .EncoderCNN import EncoderCNN
from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .mlp import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAEModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, embed_space_size,
                 sos_id, eos_id, num_layers=1, rnn_cell='LSTM',
                 bidirectional=False, input_dropout_p=0.0, dropout_p=0.0,
                 dropout_mlp=0.0, answer_max_len=None,
                 embedding=None):
        """Constructor for VQAModel.

        Args:
            vocab_size: Number of words in the vocabulary.
            max_len: The maximum length of the answers we generate.
            hidden_size: Number of dimensions of RNN hidden cell.
            vocab_embed_size: Number of dimensions of RNN embedding.
            embed_space_size: Number of dimensions of semantic space.
            sos_id: Vocab id for <start>.
            eos_id: Vocab id for <end>.
            num_layers: The number of layers of the RNNs.
            rnn_cell: LSTM or RNN or GRU.
            bidirectional: Whether the RNN is bidirectional.
            input_dropout_p: Dropout applied to the input question words.
            dropout_p: Dropout applied internally between RNN steps.
            num_att_layers: Number of stacked attention layers.
            att_ff_size: Dimensions of stacked attention.
        """
        super(RAEModel, self).__init__()
        self.encoder_cnn = EncoderCNN(embed_space_size)
        self.encoder_rnn = EncoderRNN(vocab_size, max_len, hidden_size,
                                      input_dropout_p=input_dropout_p,
                                      dropout_p=dropout_p,
                                      n_layers=num_layers,
                                      bidirectional=bidirectional,
                                      rnn_cell=rnn_cell,
                                      vocab_embed_size=vocab_embed_size,
                                      embedding=embedding,
                                      variable_lengths=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        answer_max_len = answer_max_len if answer_max_len is not None else max_len

        self.encoder_mlp = MLP(self.bidirectional_multiplier * num_layers * hidden_size,
                               hidden_size, embed_space_size,
                               num_layers=1, dropout_p=dropout_mlp)
        self.decoder_mlp = MLP(embed_space_size, hidden_size,
                               self.bidirectional_multiplier * num_layers * hidden_size,
                               num_layers=1, dropout_p=dropout_mlp)

        self.decoder = DecoderRNN(vocab_size, answer_max_len,
                                  self.bidirectional_multiplier*hidden_size,
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  n_layers=num_layers,
                                  rnn_cell=rnn_cell,
                                  bidirectional=bidirectional,
                                  input_dropout_p=input_dropout_p,
                                  dropout_p=dropout_p)

    def params_to_train(self):
        params = (list(self.encoder_cnn.cnn.fc.parameters()) +
                  list(self.encoder_mlp.parameters()) +
                  list(self.decoder.parameters()) +
                  list(self.decoder_mlp.parameters()) +
                  list(self.encoder_rnn.parameters()))
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, sequences, regions, slengths=None,
                teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        """Passes the sequence through the autoencoder.

        Args:
            sequences: Batch of sequence Variables.
            regions: Image regions for the corresponding sequences.
            slengths: List of question lengths.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - features: outputs from the encoder.
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """
        features, caption_embeddings = self.encode(
                sequences, regions, slengths=slengths)
        decoder_outputs = self.decode(
                sequences, caption_embeddings,
                teacher_forcing_ratio=teacher_forcing_ratio,
                decode_function=decode_function)
        return (features, caption_embeddings), decoder_outputs

    def encode(self, sequences, regions, slengths=None):
        """Passes the sequence through the encoder.

        Args:
            sequences: Batch of sequence Variables.
            regions: Image regions for the corresponding sequences.
            slengths: List of question lengths.

        Returns:
            - features: Image embeddings.
            - caption_embeddings: Embeddings of the input sequences.
        """
        # regions is (N * 224 * 224 * 3)
        features = None
        if regions is not None:
            features = self.encoder_cnn(regions)

        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        _, encoder_hidden = self.encoder_rnn(
            sequences, slengths, None)
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            encoder_hidden = encoder_hidden[0]
        encoder_hidden = encoder_hidden.transpose(0, 1).contiguous().view(
                sequences.size(0),
                self.bidirectional_multiplier * self.num_layers * self.hidden_size)
        caption_embeddings = self.encoder_mlp(encoder_hidden)
        return features, caption_embeddings

    def decode(self, sequences, caption_embeddings,
               teacher_forcing_ratio=0.5, decode_function=F.log_softmax):
        """Decodes the embedded sequences.

        Args:
            sequences: Batch of sequence Variables.
            caption_embeddings: Embeddings of the sequences to be generated.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            - decoder_outputs: The outputs from DecoderRNN.
        """
        reconstruct_hidden = self.decoder_mlp(caption_embeddings)
        reconstruct_hidden = reconstruct_hidden.view(
                reconstruct_hidden.size(0),
                self.bidirectional_multiplier * self.num_layers,
                self.hidden_size)
        reconstruct_hidden = reconstruct_hidden.transpose(0, 1).contiguous()
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            reconstruct_hidden = (reconstruct_hidden,
                                  torch.zeros(reconstruct_hidden.size()).cuda())
        decoder_outputs = self.decoder(inputs=sequences,
                                       encoder_hidden=reconstruct_hidden,
                                       encoder_outputs=None,
                                       function=decode_function,
                                       teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_outputs

    def predict(self, sequences, regions, slengths=None,
                teacher_forcing_ratio=0, decode_function=F.log_softmax):
        """Outputs the predicted vocab tokens for the answers in a minibatch.

        Args:
            sequences: Batch of sequence Variables.
            regions: Image regions for the corresponding sequences.
            slengths: List of question lengths.
            teacher_forcing_ratio: Whether to predict with teacher forcing.
            decode_function: What to use when choosing a word from the
                distribution over the vocabulary.

        Returns:
            A tensor with BATCH_SIZE X MAX_LEN where each element is the index
            into the vocab word.
        """
        _, decoder_outputs = self.forward(  # List(max_len * Tensor(batch * vocab_size))
            sequences, regions, slengths=slengths,
            teacher_forcing_ratio=teacher_forcing_ratio,
            decode_function=decode_function)

        # Take argmax for each timestep
        outputs, _, _ = decoder_outputs
        outputs = [o.max(1)[1] for o in outputs]  # List(max_len * Tensor(batch))

        outputs = torch.stack(outputs)  # Tensor(max_len, batch)
        outputs = outputs.transpose(0, 1)  # Tensor(batch, max_len)
        return outputs
