# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 16:38
# @Author  : hxc
# @File    : seaborn_drawer.py
# @Software: PyCharm
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from data_pro import get_sc_data
import numpy as np


def sub_plots(data_frame):
    sns.set(style="darkgrid")  # 设置样式

    # 创建一个包含三个子图的画布
    fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(15, 18), dpi=200)

    # 绘制每个算法的loss图
    sns.lineplot(data=data_frame, x='ep', y='PPO2', ax=axes[0], color='#E44C4A')
    sns.lineplot(data=data_frame, x='ep', y='DQN', ax=axes[2], color='#007EB1')
    sns.lineplot(data=data_frame, x='ep', y='Actor-Critic', ax=axes[1], color='#E17A1D')

    # 设置每个子图的标题和y轴标签
    # axes[0].set_title('PPO2')
    axes[0].set_ylabel('PPO2', fontsize=20)
    axes[0].set_xlabel('Episode', fontsize=22)
    # 设置x轴刻度大小
    axes[0].tick_params(axis='x', labelsize=20)
    # 设置y轴刻度大小
    axes[0].tick_params(axis='y', labelsize=20)
    # axes[1].set_title('Actor-Critic')
    axes[1].set_ylabel('DQN', fontsize=20)
    axes[1].set_xlabel('Episode', fontsize=22)
    # 设置x轴刻度大小
    axes[1].tick_params(axis='x', labelsize=20)
    # 设置y轴刻度大小
    axes[1].tick_params(axis='y', labelsize=20)
    axes[2].set_ylabel('Actor-Critic', fontsize=20)
    axes[2].set_xlabel('Episode', fontsize=22)
    # 设置x轴刻度大小
    axes[2].tick_params(axis='x', labelsize=20)
    # 设置y轴刻度大小
    axes[2].tick_params(axis='y', labelsize=20)

    # 设置整个图的标题
    plt.suptitle('Comparison of Reward by Algorithm', fontsize=25)
    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()


def line_plot(data_frame):
    plt.figure(figsize=(10, 6), dpi=200)  # 设置图形大小
    sns.set_theme(style="white")
    # fmri = sns.load_dataset("fmri")
    # sns.relplot(x="timepoint", y="signal", kind="line", data=fmri)

    # Plot the responses for different events and regions
    sns.relplot(x="Episodes", y="Idle time of doctors", data=data_frame,
                kind="line", col="Sample_num", col_wrap=5)
    # 使用 seaborn.lmplot 添加平滑曲线（线性回归）
    # sns.lmplot(x='Episode', y='Idle time of doctor', data=data_frame, ci=None)
    # 显示图形
    plt.show()


def plot(data_frame):
    # sns.set_theme(style="ticks")
    # 设置样式
    plt.figure(figsize=(300, 10), dpi=200)  # 设置图形大小
    sns.set(style="darkgrid")
    colors = ["#E44C4A", "#E17A1D", "#007EB1"]  # 自定义颜色列表
    sns.lineplot(data=data_frame.drop('ep', axis=1), dashes=False, palette=colors)  # 绘制折线图
    plt.xlabel('Episode')  # 设置x轴标签
    plt.ylabel('Loss')  # 设置y轴标签
    plt.title('Comparison of Loss by Algorithm')  # 设置标题
    # 使用Seaborn绘制折线图
    plt.legend(loc='upper right')  # 设置图例位置
    plt.show()


if __name__ == '__main__':
    scale = MinMaxScaler(feature_range=(0, 1))
    # df = get_sc_data()

    fmri = sns.load_dataset("fmri")
    sns.relplot(x="timepoint", y="signal", kind="line", data=fmri)

    # line_plot(df)



