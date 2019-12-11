"""Bottom Up model's attention layer.
"""

import torch
import torch.nn as nn

from .mlp import MLP


class Attention(nn.Module):

    def __init__(self, visual_size, question_size, hidden_size=512):
        super(Attention, self).__init__()
        self.nonlinear = MLP(visual_size + question_size, None, hidden_size,
                             w_norm=True)
        self.linear = MLP(hidden_size, None, 1, w_norm=True)
        self.qestion_net = MLP(question_size, None, hidden_size, w_norm=True)
        self.visual_net = MLP(visual_size, None, hidden_size, w_norm=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, visual_feats, question_feats):
        # (BATCH SIZE, NUMBER of OBJECTS, FEATURE SIZE)
        num_objs = visual_feats.size(1)

        # Repeat question features to number of objects.
        question_feats = question_feats.unsqueeze(1).repeat(1, num_objs, 1)

        # Attend over the image features.
        joint_representation = torch.cat((visual_feats, question_feats), 2)
        joint_representation = self.nonlinear(joint_representation)
        logits = self.linear(joint_representation)
        w = self.softmax(logits)

        # Weighted sum over image locations -> (BATCH_SIZE, FEATURE_SIZE).
        visual_emb = (w * visual_feats).sum(1)

        # Get question and visual representation.
        q_representation = self.question_net(question_feats)
        v_representation = self.visual_net(visual_emb)

        # Element-wise product.
        joint_representation = q_representation * v_representation
        return joint_representation


class NewAttention(nn.Module):

    def __init__(self, visual_size, question_size, hidden_size, dropout=0.2):
        super(NewAttention, self).__init__()
        self.visual_projector = MLP(visual_size, None, hidden_size, w_norm=True)
        self.question_projector = MLP(question_size, None, hidden_size, w_norm=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = MLP(hidden_size, None, 1, w_norm=True)
        self.question_net = MLP(question_size, None, hidden_size, w_norm=True)
        self.visual_net = MLP(visual_size, None, hidden_size, w_norm=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, visual_feats, question_feats):
        # (BATCH SIZE, NUMBER of OBJECTS, FEATURE SIZE)
        num_objs = visual_feats.size(1)
        v_proj = self.visual_projector(visual_feats)

        # Repeat question features to number of objects.
        q_proj = self.question_projector(question_feats)
        q_proj = q_proj.unsqueeze(1).repeat(1, num_objs, 1)

        # Attend over the image features.
        joint_representation = v_proj * q_proj
        joint_representation = self.dropout(joint_representation)
        logits = self.linear(joint_representation)
        w = self.softmax(logits)

        # Weighted sum over image locations -> (BATCH_SIZE, FEATURE_SIZE).
        visual_emb = (w * visual_feats).sum(1)

        # Get question and visual representation
        q_representation = self.question_net(question_feats)
        v_representation = self.visual_net(visual_emb)

        # Element-wise product
        joint_representation = q_representation * v_representation
        return joint_representation
