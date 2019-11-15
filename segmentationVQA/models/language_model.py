"""Used to score how well a generated question is.
"""

from .baseRNN import BaseRNN

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F  # NOQA


class LanguageModel(BaseRNN):
    """Applies a multi-layer RNN to an input sequence.

    Inputs: inputs, input_lengths
        **inputs**: list of sequences, whose length is the batch size and
                within which each sequence is a list of token IDs.
        **input_lengths** (list of int, optional): list that contains the
                lengths of sequences in the mini-batch, it must be provided when
                using variable length RNN (default: `None`)

    Outputs: output, hidden
        **output** (batch, seq_len, hidden_size): tensor containing the encoded
                features of the input sequence
        **hidden** (num_layers * num_directions, batch, hidden_size): tensor
                containing the features in the hidden state `h`

    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 num_layers=1, bidirectional=False,
                 rnn_cell='LSTM', variable_lengths=False,
                 embedding=None):
        """Constructor for Language Model.

        Args:
            vocab_size (int): size of the vocabulary
            max_len (int): a maximum allowed length for the sequence to be processed
            hidden_size (int): the number of features in the hidden state `h`
            input_dropout_p (float, optional): dropout probability for the input
                    sequence (default: 0)
            dropout_p (float, optional): dropout probability for the output sequence
                    (default: 0)
            num_layers (int, optional): number of recurrent layers (default: 1)
            bidirectional (bool, optional): if True, becomes a bidirectional encodr
                    (defulat: False)
            rnn_cell (str, optional): type of RNN cell (default: LSTM)
            variable_lengths (bool, optional): if use variable length RNN
                    (default: False)
            embedding (vocab_size, hidden_size): Tensor of embeddings or
                None. If None, embeddings are learned.
        """
        super(LanguageModel, self).__init__(
                vocab_size,
                max_len,
                hidden_size,
                input_dropout_p,
                dropout_p,
                num_layers,
                rnn_cell)

        self.variable_lengths = variable_lengths
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.embedding = embedding
        self.rnn = self.rnn_cell(hidden_size, hidden_size, num_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout_p)
        if bidirectional:
            self.linear = nn.Linear(2*hidden_size, vocab_size)
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
        if embedding is not None:
            self.embed_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights.
        """
        if self.embedding is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embed.weight = nn.Parameter(self.embedding,
                                             requires_grad=False)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def apply_embedding(self, input):
        """Embed input. If use glove embeddings, apply linear transformation.
        """
        if self.embedding is None:
            return self.embed(input)
        else:
            return self.embed_linear(self.embed(input))

    def params_to_train(self):
        params = self.parameters()
        # Don't train the embedding weights.
        params = filter(lambda p: p.requires_grad, params)
        return params

    def forward(self, input_var, input_lengths=None, h0=None):
        """Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the
                    input sequence.
            input_lengths (list of int, optional): A list that contains the
                    lengths of sequences in the mini-batch
            h0 : tensor containing initial hidden state.

        Returns: output
            **output** (batch, seq_len, hidden_size): variable containing the
                    encoded features of the input sequence
        """

        # Embed input minibatch.
        embedded = self.apply_embedding(input_var)  # (batch, seq_len, hidden_size)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            # Pack variable sequences together across batches.
            # (batch * seq_len, hidden_size)
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True)

        # Get output and hidden.
        # output: (batch, seq_len, hidden_size)
        # hidden: num_directions-tuple of (num_layers, batch, hidden_size)
        output, hidden = self.rnn(embedded, h0)
        if self.variable_lengths:
            # Unpack from (batch * seq_len, hidden_size).
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                         batch_first=True)

        # Linear transformation to vocab_size.
        output = self.linear(output)  # (batch, seq_len, vocab_size)
        return output

    def sample(self, input_var=None, soq=1, pad=0, eos=3):
        """Generate a likely question by sampling the language model.

        Only supports batch size of 1, and seeds only using first word of
        input_var. Used in test_lm.py to sanity check language model.

        Args:
            input_var (batch, seq_len): Variable containing the input sequence
                as vocab ids with start token and at least one word.
                Has type long.
            eos (int): End of sequence token.

        Returns:
            predicted_ids (batch, seq_len): Tensor containing the predicted
                output sequence sampled at each timestep and intialized from
                input_var. Begins with start token, ends with end token.
                Has type long.
        """

        # Initialize.
        if input_var is None or input_var.size(1) == 2:
            # If input question is empty string with start and end tokens.
            input_var = Variable(torch.Tensor([soq]).long().unsqueeze(0))
            if torch.cuda.is_available():
                input_var = input_var.cuda()
            input = Variable(torch.zeros(1, 1).type_as(
                                 input_var))
            input[0, 0] = input_var[0, 0]
            predicted_ids = [input_var[:, :1]]
            input = self.apply_embedding(input)
            hidden = None
            mask = torch.zeros(1, 1).type_as(input_var)
        else:
            input = Variable(torch.zeros(1, 1).type_as(
                             input_var))
            input[0, 0] = input_var[0, 1]
            predicted_ids = [input_var[:, :2]]
            input = self.apply_embedding(input)
            hidden = None
            mask = torch.zeros(1, 1).type_as(input_var)

        for i in range(self.max_len - 1):
            out, hidden = self.rnn(input, hidden)
            out = self.linear(out.squeeze(1))  # (batch, vocab_size)
            pred = F.log_softmax(out, dim=1)
            pred = torch.multinomial(torch.exp(pred), 1)  # (batch_size, 1)
            ids = torch.mul((1 - mask), pred)  # If done, pad.
            mask = mask | (ids == eos).long()
            predicted_ids.append(ids)
            input = self.apply_embedding(pred)

        predicted_ids = torch.cat(predicted_ids, dim=-1)  # (batch, seq_len)
        return predicted_ids

    def predict(self, input_var, eos=3):
        """Complete the input question with most likely word at each token.

        Only supports batch size of 1. Unlike sample, generates
        a question by completing the input. Used in test_lm.py to sanity
        check language model.

        Args:
            input_var (batch, seq_len): Variable containing the input sequence
                as vocab ids with start and end tokens. Has type long.
            eos (int): End of sequence token.

        Returns:
            predicted_ids (batch, seq_len): Tensor containing the predicted
                output sequence completd from input_var. Has type long.
        """

        # Initialize.
        input = input_var[:, 0:-1]  # Remove end token.
        predicted_ids = [input]
        input = self.apply_embedding(input)  # (batch, timestep, seq_len)
        hidden = None
        mask = torch.zeros(1, 1).type_as(input_var)

        input_len = input.size(1)
        for i in range(self.max_len - 1 - input_len):
            out, hidden = self.rnn(input, hidden)
            if out.size(1) != 1:
                out = out[:, out.size(1) - 1:out.size(1), :]
            out = self.linear(out.squeeze(1))  # (batch, vocab_size)
            pred = torch.argmax(out, dim=1, keepdim=True)  # (batch, 1)
            ids = torch.mul((1 - mask), pred)  # If done, pad.
            mask = mask | (ids == eos).long()
            predicted_ids.append(ids)
            input = self.apply_embedding(pred)

        predicted_ids = torch.cat(predicted_ids, dim=-1)  # (batch, seq_len)
        return predicted_ids
