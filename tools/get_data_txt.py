# -*- coding: utf-8 -*-
# @Time    : 2023/5/11 16:34
# @Author  : hxc
# @File    : get_data_txt.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from tools.gen_data import gen_data


def get_data_csv(env_id, path=None):
    if path is not None:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(f"./data/simulation_instances/{int(env_id + 1)}.csv")
    df = df.sort_values("did")
    df["id"] = [i for i in range(df.shape[0])]
    max_time_op = 0
    sum_op = df["pro_time"].sum()
    jobs = df.groupby("pid").count().shape[0]
    j_all_task_list = [[] for _ in range(jobs)]
    machines = df.groupby("did").count().shape[0]
    m_all_task_list = [[] for _ in range(machines)]
    jobs_length = np.zeros(machines, dtype=int)
    d_reg_num = np.zeros(machines, dtype=int)
    for m in df.groupby('did'):
        jobs_length[m[0]] = m[1]["pro_time"].sum()
        d_reg_num[m[0]] = m[1].groupby("pid").count().shape[0]
        m_all_task_list[m[0]].extend(m[1]["id"].values.tolist())
    is_multi = np.zeros((df.shape[0],))
    multi_task = []
    multi_job_num = 0
    multi_pid = []
    for j in df.groupby('pid'):
        if j[1].shape[0] > 1:
            multi_job_num += 1
            multi_pid.append(j[0])
            multi_task.extend(j[1]["id"].values.tolist())
            is_multi[j[1]["id"].values] = 1
        j_all_task_list[j[0]].extend(j[1]["id"].values.tolist())

    return (df.values, jobs, machines, max_time_op, jobs_length,
            sum_op, d_reg_num, j_all_task_list, m_all_task_list,
            multi_task, multi_job_num, multi_pid, is_multi)


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


def get_data_txt(path):
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
    df = pd.DataFrame(data=all_job_list, columns=["pid", "did", "pro_time"])
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


if __name__ == '__main__':
    # get_data("../data/instances/ta01")
    get_data_csv("../data/10_60_78.csv")
