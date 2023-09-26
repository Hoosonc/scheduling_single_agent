# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 13:31
# @Author  : hxc
# @File    : results_statistic.py
# @Software: PyCharm
import scipy.stats as stats
import numpy as np
import pandas as pd
import os


# if p_value < 0.05:
#     print("两种方法的结果存在显著差异")
# else:
#     print("两种方法的结果没有显著差异")

def get_med_iqr(data):
    # 计算中位数
    med = float(np.median(data))
    # 保留三位小数
    med = "{:.3f}".format(med)
    # 计算四分位数
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # 计算四分位数间距
    iqr = q3 - q1
    # 保留三位小数
    iqr = "{:.3f}".format(iqr)
    return med, iqr


if __name__ == '__main__':
    files = os.listdir("../data/simulation_instances")
    for file in files:
        print(file)
        target_df = pd.read_csv(f"../data/simulation_results/result_ppo_{file}")
        target_median, target_iqr = get_med_iqr(target_df["d_idle"].values)
        print("median:", target_median)
        print("iqr:", target_iqr)
        for method in ["ACO"]:
            print(method)
            df = pd.read_csv(f"../data/simulation_results/result_{method}_{file}")
            temp_med, temp_iqr = get_med_iqr(df["d_idle"].values)
            diff = np.array(df["d_idle"].values) - np.array(target_df["d_idle"].values)
            # 进行威尔科克森符号秩检验
            w_stat, p_value = stats.wilcoxon(diff)

            # 输出检验结果
            print("median:", temp_med)
            print("iqr:", temp_iqr)
            print("w_stat：", "{:.3f}".format(w_stat))
            print("p value：", "{:.3f}".format(p_value))
            print("=======================")
