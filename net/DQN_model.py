# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 9:05
# @Author  : hxc
# @File    : DQN_model.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GATConv

import torch.nn.functional as f

# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = GATConv(in_channels=4, out_channels=128, heads=4, concat=False)
        self.Norm1 = nn.BatchNorm1d(128)

        self.conv2 = GATConv(in_channels=128, out_channels=64, heads=4, concat=False)
        self.Norm2 = nn.BatchNorm1d(64)

        self.conv3 = GATConv(in_channels=64, out_channels=32, heads=4, concat=False)
        self.Norm3 = nn.BatchNorm1d(32)
        self.actor = nn.Sequential(
            Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            Linear(64, 1)
        )

    def forward(self, data):
        in_x = data.x[:, 1:-1]
        x = self.Norm1(self.conv1(in_x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.Norm2(self.conv2(x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.Norm3(self.conv3(x, data.edge_index))
        x = f.dropout(x, training=True)
        actor = self.get_actor(data.x, x)
        logits = torch.tanh(actor).view(1, -1)
        prob = f.softmax(logits, dim=1)

        return logits, prob

    def get_actor(self, fea, emb_fea):
        # 找出满足条件的行索引
        condition = (fea[:, 2] == 1) & (fea[:, 5] == 0)
        indices = torch.nonzero(condition).squeeze()
        filtered_second_tensor = emb_fea[indices, :]

        # pooled_x = torch.mean(filtered_second_tensor, dim=1, keepdim=True)
        return self.actor(filtered_second_tensor)
