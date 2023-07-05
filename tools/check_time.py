# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 15:41
# @Author  : hxc
# @File    : check_time.py
# @Software: PyCharm
import pandas as pd


def check_time(path=None, file=None):
    total_time_d = 0
    total_time_p = 0
    d_idle = 0
    p_idle = 0
    if path is not None:
        file = pd.read_csv(path, encoding='utf-8-sig').fillna('')
    cla_by_did = file.groupby("did")
    cla_by_pid = file.groupby("pid")
    for sc_d in cla_by_did:
        s = sc_d[1].sort_values("start_time").values
        total_time_d += s[len(s)-1][4]
        for i in range(len(s)):
            if i == 0:
                d_idle += s[i][2]
            else:
                assert (s[i][2] - s[i-1][[4]]) >= 0
                d_idle += (s[i][2] - s[i-1][4])
    for sc_p in cla_by_pid:
        s_p = sc_p[1].sort_values("start_time").values
        if len(s_p) > 1:
            total_time_p += s_p[len(s_p)-1][4] - s_p[0][2]
        for i in range(len(s_p)):
            if i == 0:
                # p_idle += s_p[i][2]
                p_idle += 0
            else:
                assert (s_p[i][2] - s_p[i-1][[4]]) >= 0
                p_idle += (s_p[i][2] - s_p[i-1][4])
    return total_time_d, total_time_p, d_idle, p_idle, d_idle+p_idle


if __name__ == '__main__':
    d, p, d_idle = check_time(f"../data/save_data/0_3568.csv")
    print(d, p, d+p, d_idle)
