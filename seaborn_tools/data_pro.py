# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 16:09
# @Author  : hxc
# @File    : data_pro.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import statsmodels.api as sm


def get_sc_data():
    df = pd.DataFrame()
    for i in range(10):
        temp_df = pd.read_csv(f"./data/adjust_sc/d_idle_{i}.csv")
        temp_df["Sample_num"] = [(10 - i) for _ in range(temp_df.shape[0])]
        temp_df = temp_df.rename(columns={f'd_idle_0': 'Idle time of doctors',
                                          'ep': 'Episodes'})
        window_length = 801  # 窗口大小必须是奇数
        polyorder = 2  # 多项式次数
        # 应用 Savitzky-Golay 滤波
        # filtered_values = savgol_filter(values, window_length, polyorder)
        temp_df["Idle time of doctors"] = savgol_filter(temp_df["Idle time of doctors"].values, window_length, polyorder)
        temp_df["Idle time of doctors"] = np.log(temp_df["Idle time of doctors"].values)
        # 使用 drop() 方法删除指定列
        temp_df = temp_df.drop(columns=['sum_d_idle', 'mean_d_idle'])
        if i == 0:
            df = temp_df
        else:
            df = pd.concat([df, temp_df], ignore_index=True)
    # df["Idle time of doctors"] = df["Idle time of doctors"] / df["Idle time of doctors"].max()
    return df


if __name__ == '__main__':
    a = get_sc_data()
    b = 1
