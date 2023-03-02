# -*- coding : utf-8 -*-
# @Time :  2022/8/2 14:37
# @Author : hxc
# @File : controller.py
# @Software : PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, p_in, d_in, init_rate=0.01):
        super(Critic, self).__init__()

        self.p_out = nn.Linear(p_in, d_in)
        self.d_out = nn.Linear(d_in, 5)
        self.L = nn.Linear(5, 1)
        # self.apply(weights_init)
        self.L.weight.data = normalized_columns_initializer(
            self.L.weight.data, init_rate)
        self.L.bias.data.fill_(0)

    def forward(self, p_feature, d_feature):
        p_feature = p_feature.view(1, p_feature.shape[1]*p_feature.shape[3])
        d_feature = d_feature.view(d_feature.shape[1], d_feature.shape[3])
        p_out = self.p_out(p_feature)
        p_out = f.dropout(p_out)
        d_out = self.d_out(d_feature)
        d_out = f.dropout(d_out)
        final_feature = torch.dot(p_out, d_out)
        out = self.L2(final_feature)
        out = f.dropout(out)
        return out


class Cnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cnn, self).__init__()
        torch.manual_seed(2022)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), padding=1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, (1, 1), (1, 1))
        self.train()

    def forward(self, inputs):
        inputs = torch.from_numpy(inputs).view(1, inputs.shape[0], 1, inputs.shape[1]).to(device)
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x) + inputs)
        x = f.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, action_space, type_id):
        super(MLP, self).__init__()

        self.lstm = nn.LSTMCell(in_dim, in_dim//2)
        self.action_linear = nn.Linear(in_dim//2, action_space)

        # self.apply(weights_init)
        self.action_linear.weight.data = normalized_columns_initializer(
            self.action_linear.weight.data)
        self.action_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = x.view(-1, x.shape[1] * x.shape[3])
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        x = self.action_linear(x)
        x = f.dropout(x)
        policy_head = Categorical(probs=f.softmax(x, dim=-1))
        # print(a.entropy())
        # print(a.log_prob())
        # print(a.probs)
        # print(a.sample())

        return policy_head, (hx, cx)
