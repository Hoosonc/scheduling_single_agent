# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 16:21
# @Author  : hxc
# @File    : random_scheduling.py
# @Software: PyCharm
import pandas as pd
import numpy as np

import os
from check_time import check_time
np.random.seed(2023)


class RandomScheduling:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.machines = self.data.groupby("did").count().shape[0]
        self.jobs = self.data.groupby("pid").count().shape[0]

        self.p_sc_list = None
        self.d_sc_list = None

        self.p_last_schedule = None

        self.p_sc_jobs = None
        self.d_sc_jobs = None
        self.d_tasks = None
        self.p_tasks = None
        self.tasks = None
        self.d_idle_list = []

    def reset(self):

        self.p_last_schedule = np.zeros((2, self.jobs))  # 上一个号的结束时间
        """
             [[已处理号数]
             [该病人上一个号结束的时间]]
        """

        self.d_sc_list = [[] for _ in range(self.machines)]
        self.p_sc_list = [[] for _ in range(self.jobs)]
        self.d_tasks = [[] for _ in range(self.machines)]
        self.p_tasks = [[] for _ in range(self.jobs)]
        self.tasks = []

    def step(self):
        self.reset()
        for i in range(100):
            indices = np.random.permutation(self.data.shape[0])
            data = self.data.values[indices]
            for row in data:
                pid = int(row[0])
                did = int(row[1])
                pro_time = row[2]
                last_schedule_list = self.p_last_schedule
                insert_data = self.find_position(pid, did, pro_time)
                self.d_sc_list[did].append(insert_data)
                self.d_tasks[did].append([insert_data[2], insert_data[4]])
                self.p_tasks[pid].append([insert_data[2], insert_data[4]])
                self.p_sc_list[pid].append([did, insert_data[2], insert_data[4]])
                last_schedule_list[0][pid] += 1
                if last_schedule_list[0][pid] != 0:
                    self.cal_p_idle(pid)
            sc_list = []
            for sc in self.d_sc_list:
                sc_list.extend(sc)
            df1 = pd.DataFrame(data=sc_list, columns=["did", "pid", "start_time", "pro_time", "finish_time"])
            total_time_d, total_time_p, d_idle, p_idle, total_idle = check_time(file=df1)
            self.d_idle_list.append([p_idle, d_idle, total_idle, total_time_d])
            self.reset()

    def find_position(self, pid, did, pro_time):

        self.tasks = []
        # 该医生的排班列表
        schedule_list = self.d_sc_list[int(did)]
        # 排班列表的最后结束时间
        last_time = 0
        if schedule_list:
            schedule_list = np.array(schedule_list)
            last_time = schedule_list[:, 4].max()
            self.tasks.extend(self.d_tasks[did])

        # 该病人的排班情况
        last_schedule_list = self.p_last_schedule

        # 如果该病人被分配过
        if last_schedule_list[0][pid] != 0:
            self.tasks.extend(self.p_tasks[pid])
        holes = self.find_idle_times(pro_time)
        if holes:
            insert_data = [did, pid, holes[0][0], pro_time, holes[0][0] + pro_time]

        else:
            if last_time <= last_schedule_list[1][pid]:
                insert_data = [did, pid, last_schedule_list[1][pid], pro_time, last_schedule_list[1][pid] + pro_time]
            else:
                insert_data = [did, pid, last_time, pro_time, last_time + pro_time]

        return insert_data

    def cal_p_idle(self, pid):
        sc = np.array(self.p_sc_list[pid])
        idx = sc[:, 1].argsort(axis=0).reshape(1, -1)  # 按照start_time的值排序
        sc = sc[idx, :][0]
        last_schedule_list = self.p_last_schedule
        last_schedule_list[1][pid] = sc[len(sc) - 1][2]

    def find_idle_times(self, pro_time):
        # 如果任务列表为空，返回空闲时间列表 []
        if not self.tasks:
            return []

        # 转换为 NumPy 数组并按开始时间排序
        tasks = np.array(self.tasks)
        tasks = tasks[np.argsort(tasks[:, 0])]

        idle_times = []
        prev_end_time = tasks[0, 1]

        if tasks[0, 0] >= pro_time:
            # 如果第一个任务的开始时间大于等于 Pro_time，那么将时间轴从 0 开始
            idle_times.insert(0, [0, tasks[0, 0]])
            return idle_times

        for i in range(1, len(tasks)):
            curr_start_time, curr_end_time = tasks[i]
            if curr_start_time > prev_end_time and curr_start_time - prev_end_time >= pro_time:
                # 如果当前任务的开始时间在上一个任务结束时间之后，并且它们之间的空闲时间大于 Pro_time
                idle_times.append([prev_end_time, curr_start_time])
                return idle_times
            prev_end_time = max(prev_end_time, curr_end_time)

        return idle_times


if __name__ == '__main__':
    files = os.listdir("../data/simulation_instances")
    for file in files:
        rs = RandomScheduling(f"../data/simulation_instances/{file}")
        rs.step()
        df = pd.DataFrame(data=rs.d_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
        df.to_csv(f"../data/simulation_results/result_random_{file}", index=False)
        print(file)
