import torch
import torch.nn as nn

class CoAttention(nn.Module):
    def __init__(self, v_dim, q_dim, hidden_size, hidden_size_attn):
        super(CoAttention, self).__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.linear_aff = nn.Linear(hidden_size, hidden_size) # to compute affinity matrix
        self.linear_v = nn.Linear(hidden_size, hidden_size_attn)
        self.linear_q = nn.Linear(hidden_size, hidden_size_attn)
        self.linear_hv = nn.Linear(hidden_size_attn, 1)
        self.linear_hq = nn.Linear(hidden_size_attn, 1)

    def init_weights(self):
        """Initialize the weights."""
        self.linear_aff.weight.data.normal_(0.0, 0.02)
        self.linear_aff.bias.data.fill_(0)

        self.linear_v.weight.data.normal_(0.0, 0.02)
        self.linear_v.bias.data.fill_(0)

        self.linear_q.weight.data.normal_(0.0, 0.02)
        self.linear_q.bias.data.fill_(0)

        self.linear_hv.weight.data.normal_(0.0, 0.02)
        self.linear_hv.bias.data.fill_(0)

        self.linear_hq.weight.data.normal_(0.0, 0.02)
        self.linear_hq.bias.data.fill_(0)

    def _get_affinity_mat(self, v, q):
        weighted_v = torch.transpose(self.linear_aff(v), 1, 2)
        return torch.bmm(q, weighted_v)

    def _get_hidden(self, v, q, affinity_mat, for_v=True):
        if for_v:
            wq = torch.transpose(self.linear_q(q), 1, 2)
            wqc = torch.transpose(torch.bmm(wq, affinity_mat), 1, 2)
            return self.linear_v(v) + wqc
        else:
            wv = torch.transpose(self.linear_v(v), 1, 2)
            wvc = torch.transpose(torch.bmm(wv, torch.transpose(affinity_mat, 1, 2)), 1, 2)
            return self.linear_q(q) + wvc

    def _get_attn(self, h, for_v=True):
        return self.softmax(self.linear_hv(h)) if for_v else self.softmax(self.linear_hq(h))

    def forward(self, v, q):
        """
        v: [14 x 14, 512]
        q: [20, 512]
        """
        affinity_mat = self.tanh(self._get_affinity_mat(v, q))
        h_v = self.tanh(self._get_hidden(v, q, affinity_mat))
        h_q = self.tanh(self._get_hidden(v, q, affinity_mat, for_v=False))
        a_v = self._get_attn(h_v)
        a_q = self._get_attn(h_q, for_v=False)
        v_hat = torch.bmm(torch.transpose(a_v, 1, 2), v) # 1 x 196
        q_hat = torch.bmm(torch.transpose(a_q, 1, 2), q) # 1 x 20
        return v_hat, q_hat
