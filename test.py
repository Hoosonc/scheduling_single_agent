# -*- coding : utf-8 -*-
# @Time :  2022/7/28 22:15
# @Author : hxc
# @File : test.py
# @Software : PyCharm
import random
from sklearn.utils import shuffle
import numpy as np
import torch
import torch.nn.functional as f
from torch.distributions import Categorical # import the categorical distribution class
import math
import pandas as pd
import datetime


def com_idle_time():
    idle_time = 0
    pm = [3, 5, 8, 10]
    a_idle_time = np.zeros((2, 4))
    p_idle_time = np.zeros((2, 90))
    a = pd.read_csv("./data/am.csv")
    p = pd.read_csv("./data/pm.csv")
    a = a.groupby("pid")
    p = p.groupby("pid")
    for pa in p:
        b = pa[1].sort_values(by="start").values
        if len(b) > 1:
            for i in range(len(b) - 1):
                idle_time += (b[i + 1][2] - b[i][3])
    print(idle_time)


def transition_time():
    a = np.zeros((10, 10))
    for i in range(9):
        for j in range(i + 1, 10):
            a[i][j] = random.randint(2, 5)
            a[j][i] = a[i][j]
    return a


if __name__ == '__main__':
    x = torch.tensor([1, 2, 3, 4, 5])  # a tensor of shape [3, 4]
    mask = torch.ones_like(x).bool()  # a boolean mask tensor of shape [3, 4]
    mask[1] = False  # set the mask value at position (0,1) to False
    # mask[3] = False  # set the mask value at position (2,3) to False
    probs = torch.tensor(
        [0.04, 0.80, 0.04, 0.04, 0.04])  # a tensor of shape [3 ,2], representing the probabilities for each row
    probs[~mask] = 0
    probs = probs / probs.sum()
    dist = Categorical(probs)  # create a categorical distribution object with probs as parameter
    for i in range(1000):
        sampled_index = dist.sample().item()  # sample an index from the distribution for each row
        if sampled_index == 1:
            print(sampled_index)
