# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 15:59
# @Author  : hxc
# @File    : gcn.py
# @Software: PyCharm
# import numpy as np
import torch
import torch.nn as nn
# from threading import Thread
from torch.nn import Linear
# from torch_geometric.nn import GIN
from torch_geometric.nn import GATConv
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, GATConv, GINConv, GIN, GCN
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):
    def __init__(self, reg_num, node_num):
        super(GCN, self).__init__()
        # torch.manual_seed(2022)
        self.reg_num = reg_num
        # self.p_num, self.d_num = node_num
        self.conv1 = GATConv(in_channels=reg_num, out_channels=round(reg_num/2), heads=4)
        # self.conv1 = GCNConv(4, 2)
        # self.conv1 = GIN(2, 4, num_layers=2)
        self.conv2 = GATConv(in_channels=round(reg_num/2)*4, out_channels=round(reg_num/4), heads=4)
        # self.conv2 = GCNConv(2, 2)
        # self.conv2 = GIN(4, 4, num_layers=2)
        self.conv3 = GATConv(in_channels=round(reg_num/4)*4, out_channels=round(reg_num/8), heads=4)
        # self.conv3 = GCNConv(2, 1)
        # self.conv3 = GIN(4, 1, num_layers=2)
        self.L1 = nn.Sequential(
            Linear(node_num*round(reg_num/8)*4, reg_num),
            Linear(reg_num, reg_num)
        )
        self.critic = nn.Sequential(
            Linear(node_num*round(reg_num/8)*4, reg_num),
            Linear(reg_num, 1)
        )
        # self.embedding_process = nn.Linear(4, 4)
        # self.embedding_p = nn.Linear(1, 4)
        # self.embedding_d = nn.Linear(2, 4)

        # self.apply(weights_init)

        # self.critic.weight.data = normalized_columns_initializer(
        #     self.critic.weight.data, 1)
        # self.critic.bias.data.fill_(0)

    def forward(self, nodes, edge_index, edge_attr):

        nodes = torch.tensor(nodes, dtype=torch.float32).to(device)

        edge_index = torch.tensor(edge_index).to(device)

        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)

        x = self.conv1(nodes, edge_index, edge_attr)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.conv2(x, edge_index)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.conv3(x, edge_index)
        x = x.view(1, -1)
        value = self.critic(x)
        prob = f.softmax(torch.tanh(self.L1(x)), dim=1)

        return prob, value

    def get_batch_p_v(self, buf):
        log_prob_list = []
        value_list = []
        entropy = []
        for i in range(buf.state_list.shape[0]):
            p, v = self.forward(buf.state_list[i], buf.edge_list[i], buf.edge_attr_list[i])
            policy_head = Categorical(probs=p)
            value_list.append(v)
            log_prob_list.append(policy_head.log_prob(torch.tensor(buf.action_list[i]).to(device)))
            entropy.append(policy_head.entropy())

        values = torch.cat(value_list, dim=0).view(1, -1)
        log_prob = torch.cat(log_prob_list, dim=0).view(1, -1)
        entropy = torch.cat(entropy, dim=0).view(1, -1)

        return values, log_prob, entropy
