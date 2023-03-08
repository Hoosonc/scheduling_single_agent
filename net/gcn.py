# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 15:59
# @Author  : hxc
# @File    : gcn.py
# @Software: PyCharm
# import numpy as np
import torch
import torch.nn as nn
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
        self.node_num = node_num
        self.conv1 = GATConv(in_channels=1, out_channels=8, heads=4)
        # self.conv1 = GCNConv(4, 2)
        # self.conv1 = GIN(2, 4, num_layers=2)
        self.conv2 = GATConv(in_channels=32, out_channels=8, heads=4)
        # self.conv2 = GCNConv(2, 2)
        # self.conv2 = GIN(4, 4, num_layers=2)
        # self.conv3 = GATConv(in_channels=8, out_channels=1, heads=1)
        # self.conv3 = GCNConv(2, 1)
        # self.conv3 = GIN(4, 1, num_layers=2)
        self.L1 = nn.Sequential(
            Linear(node_num*32, reg_num*2),
            Linear(reg_num*2, reg_num)
        )
        self.critic = nn.Sequential(
            Linear(node_num*32, reg_num),
            Linear(reg_num, 1)
        )

        # self.apply(weights_init)

        # self.critic.weight.data = normalized_columns_initializer(
        #     self.critic.weight.data, 1)
        # self.critic.bias.data.fill_(0)

    def forward(self, nodes, edge_index, edge_attr, env):
        nodes = torch.tensor(nodes, dtype=torch.float32).to(device)

        edge_index = torch.tensor(edge_index).to(device)

        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(device)

        x = self.conv1(nodes, edge_index, edge_attr)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.conv2(x, edge_index, edge_attr)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        # x = self.conv3(x, edge_index, edge_attr)
        x = x.view(1, -1)
        value = self.critic(x)

        p_ = torch.where(torch.tensor(env.patients.action_mask).view(1, -1).to(device),
                         torch.tanh(self.L1(x)), torch.full((1, self.reg_num), -999999.).to(device))
        policy_head = Categorical(probs=f.softmax(p_, dim=-1))
        # action = torch.argmax(policy_head.probs)
        action = torch.multinomial(policy_head.probs, num_samples=1)

        return policy_head, value, action

    def choose_action(self, data, edge, edge_attr, env):

        # edge_index = torch.tensor(edge).to(device)

        policy_head, value, action = self.forward(data, edge, edge_attr, env)

        # self.state_list.append(data)
        # self.edge_list.append(edge_index)
        # self.action_list.append(action)
        # self.value_list.append(value)
        # self.log_prob_list.append(policy_head.log_prob(action))

        return action.item(), value, policy_head.log_prob(action)

    def get_batch_p_v(self, buffer_list):
        log_prob_list = [[] for _ in range(len(buffer_list))]
        value_list = [[] for _ in range(len(buffer_list))]
        entropy = [[] for _ in range(len(buffer_list))]
        buf_n = 0
        for buf in buffer_list:
            for i in range(len(buf.state_list)):
                p, v, _ = self.forward(buf.state_list[i], buf.edge_list[i], buf.edge_attr_list[i], buf)
                value_list[buf_n].append(v)
                log_prob_list[buf_n].append(p.log_prob(torch.tensor(buf.action_list[i]).to(device)))
                entropy[buf_n].append(p.entropy())
            buf_n += 1

        values = torch.cat([torch.cat(value, dim=0).view(1, -1) for value in value_list],
                           dim=0).view(len(buffer_list), -1)
        log_prob = torch.cat([torch.cat(log_prob, dim=0).view(1, -1) for log_prob in log_prob_list],
                             dim=0).view(len(buffer_list), -1)
        entropy = torch.cat([torch.cat(e_list, dim=0).view(1, -1) for e_list in entropy],
                            dim=0).view(len(buffer_list), -1)

        return values, log_prob, entropy
