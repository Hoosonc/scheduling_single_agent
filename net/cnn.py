# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 11:32
# @Author  : hxc
# @File    : cnn.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(torch.nn.Module):
    def __init__(self, reg_num, p_num, d_num, patient, doctor):
        super(CNN, self).__init__()
        torch.manual_seed(2022)
        self.reg_num = reg_num
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (2, 2), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (2, 2), padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), (2, 2), padding=1)
        self.lstm = self.lstm = nn.LSTMCell(256, 128)
        self.L1 = Linear(128, reg_num)
        self.critic = Linear(128, 1)
        self.state_list = []
        self.edge_list = []
        self.action_list = []
        self.value_list = []
        self.log_prob_list = []

        self.hx = torch.zeros(1, 128).to(device)
        self.cx = torch.zeros(1, 128).to(device)
        self.patient = patient
        self.doctor = doctor

        # self.apply(weights_init)

        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, -500)
        self.critic.bias.data.fill_(0)

    def forward(self, data):
        x = self.conv1(data)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = x.view(1, -1)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        x = self.hx
        value = self.critic(x)
        # input_x = self.L1(x)
        # input_x = f.dropout(input_x)
        p_ = torch.where(torch.tensor(self.patient.action_mask).view(1, -1).to(device),
                         self.L1(x), torch.full((1, self.reg_num), -999999.).to(device))
        policy_head = Categorical(probs=f.softmax(p_, dim=-1))
        action = torch.argmax(policy_head.probs)

        return policy_head, value, action

    def choose_action(self, data):

        data = torch.tensor(data, dtype=torch.float32).view(1, 1, self.reg_num, 4).to(device)
        # edge_index = torch.tensor(edge).to(device)

        policy_head, value, action = self.forward(data)

        self.state_list.append(data)
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

    def reset_h_c(self):
        self.hx = torch.zeros(1, 128).to(device)
        self.cx = torch.zeros(1, 128).to(device)

    def get_batch_p_v(self, state_batch, batch_edges, p_action_batch):
        log_prob_list = []
        value_list = []
        entropy = []
        for epoch in range(len(state_batch)):
            log_prob_ = []
            v_ = []
            entropy_ = []
            for i in range(len(state_batch[epoch])):
                p, v, _ = self.forward(state_batch[epoch][i])
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
