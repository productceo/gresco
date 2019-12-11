"""Contains model that uses Stacked Attention for VQA.
"""

from models.EncoderCNNSeg import SpatialResnetEncoder
from models.EncoderRNN import EncoderRNN

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
from utils import process_lengths



class SEGModel(nn.Module):
    """Stacked attention sequence-to-sequence model.
    """

    def __init__(self, vocab_size, max_len, hidden_size,
                 vocab_embed_size, sos_id, eos_id,
                 num_layers=1, rnn_cell='LSTM', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, answer_max_len=None, 
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
            embedding: Pretrained Embedding weights.
        """
        super(SEGModel, self).__init__()
        self.encoder_cnn = SpatialResnetEncoder(2)
        self.encoder_rnn  = EncoderRNN(vocab_size, max_len, hidden_size,
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

    def params_to_train(self):
        params = (list(self.encoder_rnn.parameters()) +
                  list(self.encoder_cnn.fc.parameters()) )
        # Don't train the embedding weights.
        params = filter(lambda p: p.requires_grad, params)
        return params

    def flatten_parameters(self):
        self.encoder_rnn.rnn.flatten_parameters()

    def forward(self, images, answers, alengths=None, questions=None):
        """Passes the image and the question through a VQA model and generates answers.

        Args:
            images: Batch of image Variables.
            questions: Batch of question Variables.
            qlengths: List of question lengths.
            answers: Batch of answer Variables.

        Returns:
            - outputs: The output scores for all steps in the RNN.
            - hidden: The hidden states of all the RNNs.
            - ret_dict: A dictionary of attributes. See DecoderRNN.py for details.
        """

        # features is (N * 2048 * 56 * 56)

        input_spatial_dim = images.size()[2:]
        features = self.encoder_cnn.resnet(images)

        # encoder_hidden is ((BIDIRECTIONAL x NUM_LAYERS) * N * HIDDEN_SIZE).
        _ , encoder_hidden_ans = self.encoder_rnn(answers, alengths, None)
        
        if self.encoder_rnn.rnn_cell is nn.LSTM:
            encoder_hidden_ans = encoder_hidden_ans[0]
        encoder_hidden_ans = encoder_hidden_ans.transpose(0, 1).contiguous()

        if self.bidirectional_multiplier == 2:
            encoder_hidden = torch.cat((encoder_hidden_ans[:, 0], encoder_hidden_ans[:, -1]), dim=1)
        else:
            encoder_hidden = encoder_hidden_ans[:, -1]
        
        if questions is not None:
            alengths = process_lengths(questions)
            # Reorder based on length
            sort_index = sorted(range(len(alengths)), key=lambda x: alengths[x].item(), 
                                reverse=True)
            questions = questions[sort_index]
            alengths = np.array(alengths)[sort_index].tolist()
            _ , encoder_hidden_qs = self.encoder_rnn(
                    questions, alengths, None)
            if self.encoder_rnn.rnn_cell is nn.LSTM:
                encoder_hidden_qs = encoder_hidden_qs[0]
                encoder_hidden_qs = encoder_hidden_qs.transpose(0, 1).contiguous()

            if self.bidirectional_multiplier == 2:
                encoder_hidden_qs = torch.cat((encoder_hidden_qs[:, 0], encoder_hidden_qs[:, -1]), dim=1)
            else:
                encoder_hidden_qs = encoder_hidden_qs[:, -1]
            
            # Reorder to match answer ordering
            ordering = [sort_index.index(i) for i in range(images.size(0))]
            encoder_hidden_qs = encoder_hidden_qs[ordering]
            encoder_hidden = torch.cat([encoder_hidden, encoder_hidden_qs], dim=1)
        

        # Pass the features through the stacked attention network.
        encoder_hidden = encoder_hidden.unsqueeze(2).unsqueeze(2).repeat(1, 1, 
                                                                         features.size(2),
                                                                         features.size(3))
        features = self.encoder_cnn.fc(features * encoder_hidden)
        result = nn.functional.upsample_bilinear(input=features, size=input_spatial_dim)
    
        return result
