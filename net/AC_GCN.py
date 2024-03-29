# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 8:23
# @Author  : hxc
# @File    : AC_GCN.py
# @Software: PyCharm
import torch
import torch.nn as nn
# from threading import Thread
from torch.nn import Linear
from threading import Thread
# from torch_geometric.nn import GIN
from torch_geometric.nn import global_mean_pool
# from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GCNConv, GATConv, GINConv, GIN, GCN
import torch.nn.functional as f
from torch.distributions.categorical import Categorical
# from net.utils import normalized_columns_initializer, weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AC_GCN(torch.nn.Module):
    def __init__(self):
        super(AC_GCN, self).__init__()

        self.conv1 = GCNConv(in_channels=4, out_channels=32)
        self.Norm1 = nn.BatchNorm1d(32)

        self.conv2 = GCNConv(in_channels=32, out_channels=64)
        self.Norm2 = nn.BatchNorm1d(64)

        self.conv3 = GCNConv(in_channels=64, out_channels=32)
        self.Norm3 = nn.BatchNorm1d(32)
        self.conv4 = GCNConv(in_channels=32, out_channels=1)

        self.critic = nn.Sequential(
            Linear(64, 32),
            nn.LayerNorm(32),
            nn.Dropout(),
            nn.ReLU(),
            Linear(32, 16),
            nn.LayerNorm(16),
            nn.Dropout(),
            nn.ReLU(),
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