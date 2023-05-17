# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 16:34
# @Author  : hxc
# @File    : get_data_txt.py
# @Software: PyCharm
import numpy as np
import pandas as pd


def get_data_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values("did")
    df["id"] = [i for i in range(df.shape[0])]
    max_time_op = 0
    sum_op = df["pro_time"].sum()
    jobs = df.groupby("pid").count().shape[0]

    machines = df.groupby("did").count().shape[0]
    jobs_length = np.zeros(machines, dtype=int)
    d_reg_num = np.zeros(machines, dtype=int)
    for m in df.groupby('did'):
        jobs_length[m[0]] = m[1]["pro_time"].sum()
        d_reg_num[m[0]] = m[1].groupby("pid").count().shape[0]
    return df.values, jobs, machines, max_time_op, jobs_length, sum_op, d_reg_num


def get_data(path):
    instance_file = open(path, "r")
    line_str = instance_file.readline()
    line_cnt = 1
    machines = 0
    jobs = 0
    jobs_length = None
    max_time_op = 0
    sum_op = 0
    all_job_list = []
    while line_str:
        split_data = line_str.split()
        if line_cnt == 1:
            jobs, machines = int(split_data[0]), int(split_data[1])
            # matrix which store tuple of (machine, length of the job)
            # instance_matrix = np.zeros((jobs, machines), dtype=(int, 2))
            # contains all the time to complete jobs
            jobs_length = np.zeros(jobs, dtype=int)
        else:
            # couple (machine, time)
            assert len(split_data) % 2 == 0
            # each jobs must pass a number of operation equal to the number of machines
            assert len(split_data) / 2 == machines
            i = 0
            # we get the actual jobs
            job_nb = line_cnt - 2
            while i < len(split_data):
                machine, time = int(split_data[i]), int(split_data[i + 1])
                # instance_matrix[job_nb][i // 2] = (machine, time)
                all_job_list.append([job_nb, machine, time])
                max_time_op = max(max_time_op, time)
                jobs_length[job_nb] += time
                sum_op += time
                i += 2
        line_str = instance_file.readline()
        line_cnt += 1
    instance_file.close()
    all_job_list = np.matrix(all_job_list)
    return all_job_list, jobs, machines, max_time_op, jobs_length, sum_op


if __name__ == '__main__':
    # get_data("../data/instances/ta01")
    get_data_csv("../data/10_60_78.csv")
