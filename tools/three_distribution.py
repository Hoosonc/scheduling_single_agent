# -*- coding : utf-8 -*-
# @Time :  2022/8/23 9:08
# @Author : hxc
# @File : three_distribution.py
# @Software : PyCharm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import math
"""
对于正态分布：

.cdf(a自由度,分布的参数)  求某个分布的分布函数值
.pdf(a自由度,分布的参数)  求某个分布的概率密度函数值
.isf(a自由度,分布的参数)  求某个分布的上a分位点j客

stats.norm.cdf(α,均值,方差)；

stats.norm.pdf(α,均值,方差)；

stats.norm.isf(α,均值,方差)；

对于t分布：

stats.t.cdf(α,自由度)；

stats.t.pdf(α,自由度)；

stats.t.isf(α,自由度)；

对于F分布：

stats.f.cdf(α,自由度1,自由度2)；

stats.f.pdf(α,自由度1,自由度2)；

stats.f.isf(α,自由度1,自由度2)；

"""


def norm_image(mean, std):
    x = np.linspace(mean - 3 * math.sqrt(std), mean + 3 * math.sqrt(std), 1000)
    y = stats.norm.pdf(x, mean, std)
    plt.plot(x, y, c="red")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（中文乱码问题）
    plt.rc('axes', unicode_minus=False)
    plt.title('正态分布的概率密度函数')
    plt.tight_layout()
    # 显示网格
    plt.grid(True)
    # plt.savefig("正态分布的概率密度函数", dpi=300)
    plt.show()


def chi2_image(degrees_of_freedom):
    x = np.linspace(0, 100, 10000)
    color = ["blue", "green", "darkgrey", "darkblue", "orange"]
    y = stats.chi2.pdf(x, df=degrees_of_freedom)
    plt.plot(x, y, c=color[1])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('卡方分布')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def t_image(degrees_of_freedom):
    x = np.linspace(-5, 5, 10000)
    plt.rc('axes', unicode_minus=False)
    y = stats.t.pdf(x, degrees_of_freedom)
    plt.plot(x, y, c="orange")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('t分布的概率密度函数')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def norm_t():
    x_norm = np.linspace(-5, 5, 100000)
    y_norm = stats.norm.pdf(x_norm, 0, 1)
    plt.plot(x_norm, y_norm, c="black")
    plt.rc('axes', unicode_minus=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    color = ["green", "darkblue", "orange"]

    x_t = np.linspace(-5, 5, 100000)
    for i in range(1, 4, 1):
        y_t = stats.t.pdf(x_t, i)
        plt.plot(x_t, y_t, c=color[int(i - 1)])

    plt.title('t分布和正态分布的概率密度函数对比图')

    plt.tight_layout()
    plt.grid(True)
    plt.show()


def f_image():
    x = np.linspace(-1, 8, 10000)
    y1 = stats.f.pdf(x, 1, 10)
    y2 = stats.f.pdf(x, 5, 10)
    y3 = stats.f.pdf(x, 10, 10)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(0, 1)
    plt.rc('axes', unicode_minus=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('F分布的概率密度函数')

    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # norm_image(10, 10)
    chi2_image(10)
    # a = np.random.chisquare(1, size=10)
    # pdf = []
    # d = []
    # for r in a:
    #     chi = stats.chi2.pdf(r, 1)
    #     pdf.append(chi)
    #     d.append(math.log(chi))
    # print(a)
    # print(pdf)
    # print(d)
    # t_image(10)
    # norm_t()
    # f_image()
