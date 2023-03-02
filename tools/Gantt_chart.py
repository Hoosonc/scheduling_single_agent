# -*- coding: utf-8 -*-
# @Time    : 2022/10/27 21:53
# @Author  : hxc
# @File    : Gantt_chart.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.patches as mpatches
import pandas as pd


def gantt(path, p_num, d_num):
    data = pd.read_csv(path, header=None).drop([0]).values
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = []
    for j in range(p_num):
        col = "#"
        c = np.random.choice(a=color_arr, size=6, replace=True)
        for i in range(6):
            col = col + c[i]
        color.append(col)

    # 画布设置，大小与分辨率
    plt.figure(figsize=(100, 30), dpi=80)
    # barh-柱状图换向，循坏迭代-层叠效果
    pm = [2, 4, 7, 9]
    for i in range(data.shape[0]):
        a = 0
        if int(data[i][0]) in pm:
            a = 0
        plt.barh(int(data[i][0]), height=0.8, width=int(float(data[i][3])),
                 left=int(float(data[i][2])) + a,
                 color=color[int(float(data[i][1])) - 1])
        plt.text(int(float(data[i][2])) + a + (int(float(data[i][3])) / 2),
                 int(data[i][0]),
                 int(float(data[i][1])),
                 fontsize=100,
                 verticalalignment="center",
                 horizontalalignment="center"
                 )

    plt.title("Gantt chart", fontsize=160)
    plt.xlabel("time", fontsize=100)
    plt.tick_params(labelsize=60)
    y_labels = [i for i in range(d_num)]
    plt.yticks(range(d_num), y_labels, rotation=0, fontsize=100)
    plt.show()


if __name__ == '__main__':
    gantt("../data/save_data/1338_764_2102_2.csv", 60, 10)
