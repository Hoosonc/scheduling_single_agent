# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 8:25
# @Author  : hxc
# @File    : DQN_model.py
# @Software: PyCharm
import torch


class DQN_update:
    def __init__(self, net, device, args):
        self.net = net
        self.args = args
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr_v)

    def update_by_dqn(self, buffer, model, gamma):
        rewards = buffer.buffer_list[0].reward_list
        values = buffer.buffer_list[0].log_prob_list

        R = torch.zeros(1, 1)
        values.append(R)
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

        self.optimizer.zero_grad()

        value_loss.backward()

        self.optimizer.step()
