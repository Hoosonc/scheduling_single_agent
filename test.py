# -*- coding : utf-8 -*-
# @Time :  2022/7/28 22:15
# @Author : hxc
# @File : test.py
# @Software : PyCharm
import random
from sklearn.utils import shuffle
from scipy.optimize import linprog
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
    import torch

    # 创建一个示例的N*M的tensor
    print(1/(1+(math.e**(0.1*(-(0))))))
    # for i in range(10):
    #     a = np.random.permutation([2, 3, 8, 10])
    #     print(a)
    # 定义方程的系数矩阵和右侧常数向量
    # A_eq = np.array([[-0.1, 1, 1], [0.9, 3, 2]])
    # b_eq = np.array([0, 232])
    #
    # # 定义变量的界限
    # x_bounds = (0, None)
    # y_bounds = (0, None)
    # z_bounds = (0, None)
    #
    # # 使用线性规划求解
    # result = linprog(c=[0, 0, 0], A_eq=A_eq, b_eq=b_eq, bounds=[x_bounds, y_bounds, z_bounds])
    #
    # # 输出结果
    # x, y, z = result.x
    # print("x =", x)
    # print("y =", y)
    # print("z =", z)
