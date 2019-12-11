"""Stacked Attention model to combine visual and question embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedAttention(nn.Module):
    """Model to combine image and question embeddings.

    Can be stacked together.
    """

    def __init__(self, input_image_dim=1024, input_question_dim=1024,
                 hidden_dim=512, dropout=True):
        """Constructor for Stacked Attention.

        Args:
            input_image_dim: Dimension size of image inputs.
            input_question_dim: Dimension size of question inputs.
            hidden_dim: Dimension size of hiddens.
            dropout: Boolean to decide if we are using dropout.
        """
        super(StackedAttention, self).__init__()
        self.ff_image = nn.Linear(input_image_dim, hidden_dim)
        self.ff_ques = nn.Linear(input_question_dim, hidden_dim)
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(hidden_dim, 1)
        self.i2q = nn.Linear(input_image_dim, input_question_dim)
        self.input_image_dim = input_image_dim
        self.input_question_dim = input_question_dim
        self.hidden_dim = hidden_dim

    def init_weights(self):
        """Initialize the weights.
        """
        for layer in [self.ff_image, self.ff_ques,
                      self.ff_attention, self.i2q]:
            layer.weight.data.normal_(0.0, 0.02)
            layer.bias.data.fill_(0)

    def forward(self, images, questions):
        """Forward function for Stacked Attention.

        Args:
            images: Batch of image embeddings (N * 196 * INPUT_IMAGE_DIM).
            questions: Batch of question embeddings (N * INPUT_QUESTION_DIM).

        Returns:
            Attention vector representing new visual features
            (N * INPUT_IMAGE_DIM).
        """
        # N * 196 * INPUT_IMAGE_DIM -> N * 196 * HIDDEN_DIM
        hi = self.ff_image(images)

        # N * INPUT_QUESTION_DIM -> N * HIDDEN_DIM -> N * 1 * HIDDEN_DIM
        hq = self.ff_ques(questions).unsqueeze(dim=1)

        # N * 196 * HIDDEN_DIM
        ha = torch.tanh(hi + hq)
        if self.use_dropout:
            ha = self.dropout(ha)

        # N * 196 * HIDDEN_DIM -> N * 196 * 1 -> N * 49
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = F.softmax(ha, dim=1)

        # (N * 196 * 1, N * 196 * INPUT_IMAGE_DIM) -> N * INPUT_IMAGE_DIM
        images_attended = (pi.unsqueeze(dim=2) * images).sum(dim=1)

        # output (N * NUM_QUESTION_DIM)
        u = self.i2q(images_attended) + questions
        return u
