# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 15:59
# @Author  : hxc
# @File    : AC_model.py
# @Software: PyCharm
# import numpy as np
import torch
import torch.nn as nn
# from threading import Thread
from torch.nn import Linear
from threading import Thread
# from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, GATConv, GINConv, GIN, GCN
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AC(torch.nn.Module):
    def __init__(self):
        super(AC, self).__init__()
        # torch.manual_seed(2022)

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

        self.critic = nn.Sequential(
            Linear(32, 128),
            nn.LayerNorm(128),
            # nn.Dropout(),
            nn.ReLU(),
            Linear(128, 32),
            nn.LayerNorm(32),
            # nn.Dropout(),
            nn.ReLU(),
            Linear(32, 1)
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
        pooled_x = global_mean_pool(x, None)

        actor = self.get_actor(data.x, x)
        value = self.critic(pooled_x).view(1, 1)
        logits = torch.tanh(actor).view(1, -1)
        prob = f.softmax(logits, dim=1)
        log_prob = f.log_softmax(logits, dim=1)

        return prob, value, log_prob

    def get_actor(self, fea, emb_fea):
        # 找出满足条件的行索引
        condition = (fea[:, 2] == 1) & (fea[:, 5] == 0)
        indices = torch.nonzero(condition).squeeze()
        filtered_second_tensor = emb_fea[indices, :]

        # pooled_x = torch.mean(filtered_second_tensor, dim=1, keepdim=True)
        return self.actor(filtered_second_tensor)

    def get_batch_p_v(self, buf):
        log_prob_list = []
        value_list = []
        entropy = []

        for i in range(len(buf.state_list)):
            p, v, log_p = self.forward(buf.state_list[i])
            policy_head = Categorical(probs=p)
            value_list.append(v)
            a = torch.from_numpy(buf.action_list[i]).view(-1, 1).to(device)
            log_prob_list.append(torch.sum(policy_head.log_prob(a)).view(1, -1))
            entropy.append(policy_head.entropy())

        values = torch.cat(value_list, dim=0).view(1, -1)
        log_prob = torch.cat(log_prob_list, dim=0).view(1, -1)
        entropy = torch.cat(entropy, dim=0).view(1, -1)

        return values, log_prob, entropy
