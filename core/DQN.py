# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 8:25
# @Author  : hxc
# @File    : DQN_model.py
# @Software: PyCharm
import torch
import torch.nn as nn


class DQN_update:
    def __init__(self, net, device, args):
        self.net = net
        self.args = args
        self.gamma = args.gamma
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.loss_fn = nn.MSELoss()

    def learn(self, buffer):
        q = buffer.q_list
        q_returns = buffer.q_returns

        loss = 0.5*(q-q_returns).pow(2).sum()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        return loss
