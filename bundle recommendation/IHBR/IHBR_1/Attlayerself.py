import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayerSelf(nn.Module):
    def __init__(self, weight_regularizer):
        super(AttLayerSelf, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        q = x
        v = x
        k = x
        att = torch.bmm(q.unsqueeze(2), k.unsqueeze(1)) / 10.0
        weight = F.softmax(att, dim=1)
        weight = self.dropout(weight)
        output = torch.bmm(weight, v.unsqueeze(2)).squeeze(2)
        output = output + x
        return output