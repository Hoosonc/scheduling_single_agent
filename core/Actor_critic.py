# -*- coding: utf-8 -*-
# @Time    : 2023/5/21 8:26
# @Author  : hxc
# @File    : Actor_critic.py
# @Software: PyCharm
import torch


class AC_update:
    def __init__(self, net, device, args):
        self.net = net
        self.args = args
        self.device = device
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

    def learn(self, buffer):
        returns = buffer.returns
        values = buffer.value_list
        adv = buffer.adv
        log_probs = buffer.log_prob_list
        # log_probs = torch.cat([lg.view(1, 1) for lg in log_probs], dim=1)
        # values = torch.cat(values, dim=1)
        value_loss = 0.5*(values-returns).pow(2).mean()
        policy_loss = -(log_probs * adv).mean()
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        return loss
