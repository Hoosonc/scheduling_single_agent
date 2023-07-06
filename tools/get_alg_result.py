# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 12:39
# @Author  : hxc
# @File    : get_alg_result.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from check_time import check_time


def get_solution(solution, path):
    df = pd.read_csv(path)
    data = df.values
    d_num = df.groupby("did").count().shape[0]
    p_num = df.groupby("pid").count().shape[0]
    d_last_time = np.zeros((d_num,))
    p_last_time = np.zeros((p_num,))
    sc_list = []
    for index in solution:
        pid = int(data[index][0])
        did = int(data[index][1])
        pro_time = data[index][2]
        if d_last_time[did] >= p_last_time[pid]:
            start_time = d_last_time[did]
        else:
            start_time = p_last_time[pid]
        finish_time = start_time + pro_time
        d_last_time[did] = finish_time
        p_last_time[pid] = finish_time
        sc_list.append([did, pid, start_time, pro_time, finish_time])
    df = pd.DataFrame(data=sc_list, columns=["did", "pid", "start_time", "pro_time", "finish_time"])
    total_time_d, total_time_p, d_idle, p_idle, total_idle = check_time(file=df)
    return total_time_d, total_time_p, d_idle, p_idle, total_idle
