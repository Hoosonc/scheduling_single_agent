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
    def __init__(self, reg_num, patient, doctor):
        super(GCN, self).__init__()
        # torch.manual_seed(2022)
        self.reg_num = reg_num
        self.conv1 = GATConv(in_channels=12, out_channels=6, heads=2)
        # self.conv1 = GCNConv(4, 2)
        # self.conv1 = GIN(2, 4, num_layers=2)
        self.conv2 = GATConv(in_channels=12, out_channels=3, heads=2)
        # self.conv2 = GCNConv(2, 2)
        # self.conv2 = GIN(4, 4, num_layers=2)
        self.conv3 = GATConv(in_channels=6, out_channels=1, heads=1)
        # self.conv3 = GCNConv(2, 1)
        # self.conv3 = GIN(4, 1, num_layers=2)
        self.L1 = nn.Sequential(
            Linear(97, reg_num*2),
            Linear(reg_num*2, reg_num)
        )
        self.critic = nn.Sequential(
            Linear(97, reg_num),
            Linear(reg_num, 1)
        )
        self.embedding_process = nn.Linear(4, 12)
        self.embedding_p = nn.Linear(3, 12)
        self.embedding_d = nn.Linear(3, 12)
        self.state_list = []
        self.edge_list = []
        self.action_list = []
        self.value_list = []
        self.log_prob_list = []
        self.patient = patient
        self.doctor = doctor

        # self.apply(weights_init)

        # self.critic.weight.data = normalized_columns_initializer(
        #     self.critic.weight.data, 1)
        # self.critic.bias.data.fill_(0)

    def forward(self, process_state=None, multi_reg_state=None, d_state=None, edge_index=None, data=None):
        if data != None:
            data = data
        else:
            embedding_process = self.embedding_process(process_state)
            embedding_p = self.embedding_p(multi_reg_state)
            embedding_d = self.embedding_d(d_state)
            data = torch.cat([embedding_process, embedding_p, embedding_d], dim=0).to(device)

        x = self.conv1(data, edge_index)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.conv2(x, edge_index)
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.conv3(x, edge_index)
        x = x.view(1, -1)
        # x = self.hx
        value = self.critic(x)
        # input_x = self.L1(x)
        # input_x = f.dropout(input_x)
        p_ = torch.where(torch.tensor(self.patient.action_mask).view(1, -1).to(device),
                         torch.tanh(self.L1(x)), torch.full((1, self.reg_num), -999999.).to(device))
        policy_head = Categorical(probs=f.softmax(p_, dim=-1))
        # action = torch.argmax(policy_head.probs)
        action = torch.multinomial(policy_head.probs, num_samples=1)

        return policy_head, value, action, data

    def choose_action(self, process_state, multi_reg_state, d_state, edge):

        process_state = torch.tensor(process_state, dtype=torch.float32).to(device)
        multi_reg_state = torch.tensor(multi_reg_state, dtype=torch.float32).to(device)
        d_state = torch.tensor(d_state, dtype=torch.float32).to(device)
        edge_index = torch.tensor(edge).to(device)

        policy_head, value, action, data = self.forward(process_state, multi_reg_state, d_state, edge_index)
        self.state_list.append(data)
        self.edge_list.append(edge_index)
        self.action_list.append(action)
        self.value_list.append(value)
        self.log_prob_list.append(policy_head.log_prob(action))

        return action.item()

    def reset(self):
        self.state_list = []
        self.edge_list = []
        self.edge_attr_list = []
        self.action_list = []
        self.value_list = []
        self.log_prob_list = []

    def get_batch_p_v(self, state_batch, batch_edges, p_action_batch):
        log_prob_list = []
        value_list = []
        entropy = []
        for epoch in range(len(state_batch)):
            log_prob_ = []
            v_ = []
            entropy_ = []
            for i in range(len(state_batch[epoch])):
                p, v, _, _ = self.forward(data=state_batch[epoch][i], edge_index=batch_edges[epoch][i])
                v_.append(v)
                log_prob_.append(p.log_prob(torch.tensor([p_action_batch[epoch][i]]).to(device)))
                entropy_.append(p.entropy())
            log_prob_ = torch.cat([lg.view(1, -1) for lg in log_prob_], dim=1).view(1, -1)
            v_ = torch.cat([lg.view(1, -1) for lg in v_], dim=1).view(1, -1)
            entropy_ = torch.cat([e.view(1, -1) for e in entropy_], dim=1).view(1, -1)

            log_prob_list.append(log_prob_)
            value_list.append(v_)
            entropy.append(entropy_)

        values = torch.cat(value_list, dim=0)
        log_prob = torch.cat(log_prob_list, dim=0)
        entropy = torch.cat(entropy, dim=0)

        return values, log_prob, entropy
