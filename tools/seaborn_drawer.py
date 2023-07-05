# -*- coding: utf-8 -*-
# @Time    : 2023/6/1 16:38
# @Author  : hxc
# @File    : seaborn_drawer.py
# @Software: PyCharm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
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


def plot(data_frame):
    # sns.set_theme(style="ticks")
    # 设置样式
    plt.figure(figsize=(10, 6), dpi=200)  # 设置图形大小
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
    df = pd.read_csv("../data/reward.csv")
    df.iloc[:, [0, 1, 2]] = np.log(df.iloc[:, [0, 1, 2]])
    df.iloc[:, [0, 1, 2]] = scale.fit_transform(df.iloc[:, [0, 1, 2]])
    # plot(df)
    sub_plots(df)



