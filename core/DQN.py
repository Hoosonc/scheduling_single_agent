# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 8:25
# @Author  : hxc
# @File    : DQN_model.py
# @Software: PyCharm
import torch
import numpy as np
import torch.nn as nn


class DQN_update:
    def __init__(self, net, device, args):
        self.net = net
        self.args = args
        self.gamma = args.gamma
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr_v)
        self.loss_fn = nn.MSELoss()

    def learn(self, buffer):
        rewards = buffer.buffer_list[0].reward_list
        terminates = buffer.buffer_list[0].terminal_list
        terminates = torch.from_numpy(np.array(terminates, dtype=int)).to(self.device).detach().view(1, -1)
        n_steps = torch.from_numpy(np.arange(len(rewards))).to(self.device).view(1, -1)
        q_list = buffer.buffer_list[0].q_list
        q_list.append(torch.tensor([0]).view(1, 1).to(self.device))
        rewards = torch.tensor(rewards).to(self.device).view(1, -1)
        q = torch.cat(q_list[:-1], dim=1)
        q_next = torch.cat(q_list[1:], dim=1)
        returns = rewards + (self.gamma ** n_steps) * (1 - terminates) * q_next
        loss = self.loss_fn(q, returns).to(torch.float32)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
