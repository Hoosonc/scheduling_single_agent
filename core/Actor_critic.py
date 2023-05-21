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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr_v)