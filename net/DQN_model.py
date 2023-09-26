# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 9:05
# @Author  : hxc
# @File    : DQN_model.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import GATConv

import torch.nn.functional as f

# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = GATConv(in_channels=5, out_channels=128, heads=4, concat=False)
        self.Norm1 = nn.BatchNorm1d(128)

        self.conv2 = GATConv(in_channels=128, out_channels=64, heads=4, concat=False)
        self.Norm2 = nn.BatchNorm1d(64)

        self.conv3 = GATConv(in_channels=64, out_channels=32, heads=4, concat=False)
        self.Norm3 = nn.BatchNorm1d(32)
        self.actor = nn.Sequential(
            Linear(1, 128),
            # nn.LayerNorm(128),
            # nn.ReLU(),
            Linear(128, 64),
            # nn.LayerNorm(64),
            # nn.ReLU(),
            Linear(64, 1)
        )

    def forward(self, data):
        in_x = data.x[:, :-1]
        x = self.Norm1(self.conv1(in_x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.Norm2(self.conv2(x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.Norm3(self.conv3(x, data.edge_index))
        x = f.dropout(x, training=True)
        pooled_x = global_mean_pool(x, None)
        actor = self.get_actor(data.x, x, pooled_x)
        logits = torch.tanh(actor).view(1, -1)
        prob = f.softmax(logits, dim=1)

        return logits, prob

    def get_actor(self, fea, emb_fea, pooled_x):
        pooled_x = pooled_x.view(-1, 1)
        condition = (fea[:, 3] == 1)
        action_num = torch.sum(condition)
        indices = torch.nonzero(condition).view(action_num, )
        filtered_second_tensor = emb_fea[indices]

        actor_input = torch.mm(filtered_second_tensor, pooled_x)

        return self.actor(actor_input)