# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 15:59
# @Author  : hxc
# @File    : AC_model.py
# @Software: PyCharm
# import numpy as np
import torch
import torch.nn as nn
# from threading import Thread
from torch.nn import Linear
from threading import Thread
# from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, GATConv, GINConv, GIN, GCN
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AC(torch.nn.Module):
    def __init__(self, jobs, machines):
        super(AC, self).__init__()
        # torch.manual_seed(2022)
        self.jobs = jobs
        self.machines = machines

        self.conv1 = GATConv(in_channels=4, out_channels=4, heads=4)
        self.Norm1 = nn.BatchNorm1d(16)

        self.conv2 = GATConv(in_channels=16, out_channels=16, heads=4)
        self.Norm2 = nn.BatchNorm1d(64)

        self.conv3 = GATConv(in_channels=64, out_channels=32, heads=1)
        self.Norm3 = nn.BatchNorm1d(32)
        self.conv4 = GATConv(in_channels=32, out_channels=1, heads=1)
        # self.actor = nn.Sequential(
        #     Linear(128, 64),
        #     nn.Dropout(),
        #     # nn.ReLU(),
        #     # nn.BatchNorm1d(64),
        #     Linear(64, 32),
        #     nn.Dropout(),
        #     # nn.BatchNorm1d(32),
        #     Linear(32, 1)
        # )
        self.critic = nn.Sequential(
            Linear(64, 32),
            # nn.Dropout(),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            Linear(32, 16),
            # nn.Dropout(),
            nn.ReLU(),
            # nn.BatchNorm1d(16),
            Linear(16, 1)
        )
        self.mini_bf_list = [mini_bf() for _ in range(20)]

    def forward(self, data):

        x = self.Norm1(self.conv1(data.x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        x = self.Norm2(self.conv2(x, data.edge_index))
        x = f.elu(x)
        x = f.dropout(x, training=True)
        # x = self.conv3(x, data.edge_index)
        # x = f.dropout(x, training=True)
        pooled_x = global_mean_pool(x, None)

        # dummy = candidate.view(1, -1, 1).expand(-1, self.jobs, x.size(-1))
        # candidate_feature = torch.gather(x.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        # h_pooled_repeated = pooled_x.unsqueeze(1).expand_as(candidate_feature)
        # concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        actor = self.conv4(self.Norm3(self.conv3(x, data.edge_index)), data.edge_index)

        value = self.critic(pooled_x).view(1, 1)
        # logits = torch.tanh(self.actor(concateFea)).view(1, -1)
        logits = torch.tanh(actor).view(1, -1)
        prob = f.softmax(logits, dim=1)
        log_prob = f.log_softmax(logits, dim=1)

        return prob, value, log_prob

    def get_batch_p_v(self, buf):
        log_prob_list = []
        value_list = []
        entropy = []
        t_list = []
        # 20个线程，每个线程获取多个step的值
        mini_batch = len(buf.state_list) // 20
        remainder = len(buf.state_list) % 20
        for i in range(20):
            start = i * mini_batch
            end = (i+1) * mini_batch
            if i == 19:
                end += remainder
            t = Thread(target=self.get_forward, args=(buf, start, end, i))
            t.start()
            t_list.append(t)

        for thread in t_list:
            thread.join()

        for bf in self.mini_bf_list:
            value_list.extend(bf.value_list)
            log_prob_list.extend(bf.log_prob_list)
            entropy.extend(bf.entropy)
            bf.reset()

        values = torch.cat(value_list, dim=0).view(1, -1)
        log_prob = torch.cat(log_prob_list, dim=0).view(1, -1)
        entropy = torch.cat(entropy, dim=0).view(1, -1)

        return values, log_prob, entropy

    def get_forward(self, buf, start, end, bf_nb):
        for i in range(start, end):
            p, v, log_p = self.forward(buf.state_list[i])
            policy_head = Categorical(probs=p)
            self.mini_bf_list[bf_nb].value_list.append(v)
            self.mini_bf_list[bf_nb].log_prob_list.append(policy_head.log_prob(
                torch.tensor(buf.action_list[i]).to(device)))
            self.mini_bf_list[bf_nb].entropy.append(policy_head.entropy())


class mini_bf:
    def __init__(self):
        self.log_prob_list = []
        self.value_list = []
        self.entropy = []

    def reset(self):
        self.log_prob_list = []
        self.value_list = []
        self.entropy = []
