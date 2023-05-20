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
    data = np.array([[0, 1], [6, 9], [1, 3], [2, 3], [3, 3], [4, 5], [5, 8]])
    a = np.arange(25).reshape(5, 5)
    b = np.array([[1, 2], [3, 4], [2, 3]])

    # 创建布尔矩阵mask
    # mask = np.zeros_like(a, dtype=bool)
    # mask[b[:, 0], b[:, 1]] = True
    #
    # # 使用掩码将a中对应位置置为0
    # a[mask] = 0
    a[b[:, 0], b[:, 1]] = 0
    print(a)
    # arr = np.array([1, 2, 3, 4])
    # arr1 = np.array([5, 6])
    # # 生成所有可能的组合
    # grid = np.meshgrid(arr1, arr)

    # 将结果转换为所需的格式
    # combinations = np.vstack(grid).reshape(2, -1).T
    #
    # print(combinations)
    # 获取开始时间的唯一值
    # start_times = np.unique(data[:, 1])  # [1 3 5 8 9]
    # print(start_times)
    # # 创建连接矩阵
    # n = len(start_times)
    # adj_matrix = np.zeros((7, 7), dtype=int)
    #
    # # 将连接关系填充到矩阵中
    # for i in range(n):
    #     if i == 0:
    #
    #         indices = np.where(data[:, 1] == start_times[i])[0]
    #         for j in indices:
    #             adj_matrix[i, np.where(start_times == data[j, 1])] = 1
    #     else:
    #
    #
    # print(adj_matrix)
