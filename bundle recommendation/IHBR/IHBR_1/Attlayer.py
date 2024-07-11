import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer(nn.Module):
    def __init__(self, attention_dim, weight_regularizer):
        super(AttLayer, self).__init__()
        self.attention_dim = attention_dim
        self.W = nn.Parameter(torch.Tensor(attention_dim, attention_dim))
        self.b = nn.Parameter(torch.Tensor(attention_dim))
        self.u = nn.Parameter(torch.Tensor(attention_dim, 1))
        self.weight_regularizer = weight_regularizer
        self.dropout = nn.Dropout(0.3)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W)
        nn.init.normal_(self.b)
        nn.init.normal_(self.u)

    def forward(self, x, mask=None):
        uit = torch.tanh(torch.matmul(x, self.W) + self.b)
        ait = torch.matmul(uit, self.u)
        ait = ait.squeeze(-1)
        ait = torch.exp(ait)

        if self.training:
            ait = self.dropout(ait)

        if mask is not None:
            ait = ait * mask.unsqueeze(-1).to(torch.float32)

        ait /= torch.sum(ait, dim=0, keepdim=True) + torch.finfo(torch.float32).eps

        ait = ait.unsqueeze(1)
        weighted_input = x * ait
        output = torch.sum(weighted_input, dim=0)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[1],)