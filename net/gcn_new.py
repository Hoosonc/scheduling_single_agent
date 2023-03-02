# -*- coding: utf-8 -*-
# @Time    : 2023/1/5 21:49
# @Author  : hxc
# @File    : gcn_new.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(torch.nn.Module):
    def __init__(self, reg_num, out_num):
        super(Net, self).__init__()

        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 2)
        self.conv3 = GCNConv(2, 1)
        self.module = nn.Sequential(
            # self.conv1 = GIN(6, 3, num_layers=2)
            # self.conv2 = GIN(3, 3, num_layers=2)
            # self.conv3 = GIN(3, 1, num_layers=2)
            nn.Linear(reg_num, reg_num*2, bias=True),
            nn.Tanh(),
            nn.Linear(reg_num*2, out_num, bias=True),
            nn.Tanh())

    def get_output(self, data, edge):
        out = torch.relu(self.conv1(data, edge))
        out = torch.relu(self.conv2(out, edge))
        out = self.conv3(out, edge)
        out = out.view(1, -1)
        out = self.module(out)
        return out


class GCN(torch.nn.Module):
    def __init__(self, reg_num, p_num, d_num, patient, doctor):
        super(GCN, self).__init__()
        self.actor = Net(reg_num, reg_num)
        # self.actor = self.actor.module
        self.critic = Net(reg_num, 1)
        # self.critic = self.critic.module
        self.reg_num = reg_num

        self.state_list = []
        self.edge_list = []
        self.action_list = []
        self.value_list = []
        self.log_prob_list = []

        self.patient = patient
        self.doctor = doctor

        # self.apply(weights_init)

        # self.critic.weight.data = normalized_columns_initializer(
        #     self.critic.weight.data, 10)
        # self.critic.bias.data.fill_(0)

    def forward(self, data, edge_index):
        actor = self.actor.get_output(data, edge_index)
        actor = actor.view(1, -1)
        critic = self.critic.get_output(data, edge_index)
        critic = critic.view(1, -1)

        value = critic

        p_ = torch.where(torch.tensor(self.patient.action_mask).view(1, -1).to(device),
                         actor, torch.full((1, self.reg_num), -999999.).to(device))
        policy_head = Categorical(probs=f.softmax(p_, dim=-1))
        action = torch.argmax(policy_head.probs)

        return policy_head, value, action

    def choose_action(self, data, edge):

        data = torch.tensor(data, dtype=torch.float32).detach().to(device)
        edge_index = torch.tensor(edge).detach().to(device)

        policy_head, value, action = self.forward(data, edge_index)

        self.state_list.append(data)
        self.edge_list.append(edge_index)
        self.action_list.append(action)
        self.value_list.append(value)
        self.log_prob_list.append(policy_head.log_prob(action))

        return action.item()

    def reset(self):
        self.state_list = []
        self.edge_list = []
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
                p, v, _ = self.forward(state_batch[epoch][i], batch_edges[epoch][i])
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
