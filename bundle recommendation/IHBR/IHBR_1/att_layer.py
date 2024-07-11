
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class MHAtt(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.3):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        len_q, len_k, len_v = q.size(0), k.size(0), v.size(0)

        residual = q

        q = self.w_qs(q).view(len_q, n_head, d_k)
        k = self.w_ks(k).view(len_k, n_head, d_k)
        v = self.w_vs(v).view(len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)


        return q


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output

class Att(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Att, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=0)
        return beta * z

class Adapter(nn.Module):
    def __init__(self,hidden_size=128, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.hidden_size = hidden_size
        self.init_scale= init_scale
    # define your model layers here
    def forward(self,input_tensor):
        in_size = input_tensor.size(-1)
        w1 = nn.Parameter(torch.Tensor(in_size,self.hidden_size))
        b1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(w1, std=self.init_scale)
        nn.init.zeros_(b1)
        w1=w1.to('cuda')
        b1=b1.to('cuda')
        net = torch.matmul(input_tensor,w1) + b1
        net = F.gelu(net)
        w2 = nn.Parameter(torch.Tensor(self.hidden_size,in_size))
        b2 = nn.Parameter(torch.Tensor(1, in_size))
        w2=w2.to('cuda')
        b2=b2.to('cuda')
        nn.init.normal_(w2, std= self.init_scale)
        nn.init.zeros_(b2)
        net = torch.matmul(net, w2) + b2
        return net + input_tensor