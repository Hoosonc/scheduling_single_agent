# -*- coding: utf-8 -*-
# @Time    : 2022/11/1 19:58
# @Author  : hxc
# @File    : curve_.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show():
    file = pd.read_csv("../data/save_data/reward_loss/r_l.csv")

    # 设定画布。dpi越大图越清晰，绘图时间越久
    fig=plt.figure(figsize=(10, 10), dpi=80)
    # 导入数据
    x = [i for i in range(1, file.values.shape[0]+1)]
    y1 = file["p_loss"].values
    y2 = file["d_loss"].values
    y3 = file["loss"].values
    # 绘图命令
    plt.plot(x, y1, lw=4, ls='-', c='b', alpha=0.5)
    plt.plot(x, y2, lw=4, ls='-', c='r', alpha=0.5)
    plt.plot(x, y3, lw=4, ls='-', c='g', alpha=0.5)
    plt.plot()
    # show出图形
    plt.show()
    # 保存图片
    # fig.savefig("画布")


if __name__ == '__main__':
    show()
