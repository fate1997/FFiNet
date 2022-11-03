import torch
import torch.nn as nn
from typing import List
import numpy as np
from torch.autograd import Variable


def cal_distance(m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    vector = m - n
    distance = torch.norm(vector, p=2, dim=1)
    torch.clamp_(distance, min=0.1)
    return distance


def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    dot = torch.matmul(ba.unsqueeze(-1).transpose(-2, -1), bc.unsqueeze(-1))
    cosine_angle = dot.squeeze(-1) / (torch.norm(ba, p=2, dim=1).reshape(-1, 1) * torch.norm(bc, p=2, dim=1).reshape(-1, 1))
    cosine_angle = torch.where(torch.logical_or(cosine_angle > 1, cosine_angle < -1), torch.round(cosine_angle), cosine_angle)
    angle = torch.arccos(cosine_angle)

    return angle


def cal_dihedral(a, b, c, d):
    ab = a - b
    cb = c - b
    dc = d - c

    cb /= torch.norm(cb, p=2, dim=1).reshape(-1, 1)
    v = ab - torch.matmul(ab.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    w = dc - torch.matmul(dc.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    x = torch.matmul(v.unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
    y = torch.matmul(torch.cross(cb, v).unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)

    return torch.atan2(y, x)


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, dim_per_layer: List, dropout=0.0, activation=nn.ELU()):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(dim_per_layer) - 2):
            self.layers.append(nn.Linear(dim_per_layer[i], dim_per_layer[i + 1]))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(dim_per_layer[-2], dim_per_layer[-1]))
        self.model = nn.Sequential(*self.layers)
        self.model.apply(init_weight)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class PositionEncoder(nn.Module):
    def __init__(self, d_model, seq_len=4, device='cuda:0'):
        super().__init__()
        # position_enc.shape = [seq_len, d_model]
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(seq_len)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        self.position_enc = torch.tensor(position_enc, device=device).unsqueeze(0).float()

    def forward(self, x):
        # x.shape = [batch_size, seq_length, d_model]
        x = x * Variable(self.position_enc, requires_grad=False)
        return x