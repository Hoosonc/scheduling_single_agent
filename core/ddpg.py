# -*- coding: utf-8 -*-
# @Time    : 2023/8/22 15:49
# @Author  : hxc
# @File    : ddpg.py
# @Software: PyCharm
import torch
import torch.nn as nn


class DDPG_update:
    def __init__(self, net, device, args):
        self.net = net
        self.args = args
        self.gamma = args.gamma
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        self.loss_fn = nn.MSELoss()

    def learn(self, buffer):
        q = buffer.log_prob
        q_returns = buffer.q_returns
        actor_loss = -torch.mean(buffer.log_prob, dim=-1)
        critic_loss = 0.5*(q-q_returns).pow(2).sum()

        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()

        self.optimizer.step()
        return loss
