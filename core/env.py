# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
# import random
# import matplotlib.pyplot as plt
import random

import numpy as np
import torch
# import torch.nn.functional as f
import pandas as pd
from tools.get_data_txt import get_data_txt, get_data_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:
    def __init__(self, args):
        # init origin attributes
        self.args = args
        # (self.all_job_list, self.jobs, self.machines,
        #  self.max_time_op, self.jobs_length, self.sum_op, self.d_reg_num) = get_data_txt(args.reg_path)
        (self.all_job_list, self.jobs, self.machines,
         self.max_time_op, self.jobs_length, self.sum_op, self.d_reg_num) = get_data_csv(args.reg_path, 300, 10)
        self.reg_num = self.all_job_list.shape[0]
        self.j_edge_matrix = None
        self.m_edge_matrix = None
        self.state = None
        self.p_sc_list = None
        self.d_sc_list = None

        self.p_last_schedule = None
        self.d_position = None
        self.d_total_idle_time = None
        self.p_total_idle_time = None

        self.done = False
        self.pos = []
        self.hole_total_time = 0.
        self.total_idle_time_p = 0.
        self.render_data = []
        self.color = []
        self.d_total_time = 0
        self.p_total_time = 0
        self.p_s_f = None

        self.edge_idle = None
        self.edge_connect = None
        self.edge_mask = None
        self.p_sc_jobs = None
        self.d_sc_jobs = None
        self.d_tasks = None
        self.p_tasks = None
        self.tasks = None
        self.reward = None
        self.idle_total = []

    def reset(self, ep):
        if (ep+1) % 100 == 0:
            (self.all_job_list, self.jobs, self.machines,
             self.max_time_op, self.jobs_length, self.sum_op, self.d_reg_num) = get_data_csv(self.args.reg_path, 300, 10)
        self.state = self.all_job_list.copy()
        self.state = np.concatenate([self.state, np.ones((self.state.shape[0], 1))], axis=1)  # 添加“是否处理”
        self.state = np.concatenate([self.state, np.zeros((self.state.shape[0], 2))], axis=1)  # add “开始时间” 和 ”最早开始时间“
        self.j_edge_matrix = np.eye(self.reg_num, dtype="int64")
        self.m_edge_matrix = np.eye(self.reg_num, dtype="int64")
        self.init_edge_matrix()
        # add a column 'id'
        # self.state = np.concatenate([self.state, np.arange(self.state.shape[0]).reshape(-1, 1)], axis=1)

        self.p_last_schedule = np.zeros((2, self.jobs))  # 上一个号的结束时间
        """
             [[已处理号数]
             [该病人上一个号结束的时间]]
        """
        self.d_position = np.zeros((self.machines,))
        self.p_sc_list = [[] for _ in range(self.jobs)]
        self.d_sc_list = [[] for _ in range(self.machines)]
        self.d_total_idle_time = np.zeros((self.machines,))
        self.p_total_idle_time = np.zeros((self.jobs,))
        self.d_total_time = 0
        self.p_total_time = 0
        self.p_s_f = np.zeros((2, self.jobs))

        self.done = False

        self.hole_total_time = 0

        self.d_tasks = [[] for _ in range(self.machines)]
        self.p_tasks = [[] for _ in range(self.jobs)]
        self.tasks = []
        self.reward = 1.
        self.idle_total = []

    def step(self, action, step):
        reward = 0.
        if self.state[action][4] == 0:
            pass
        else:

            process_id = action
            did = int(self.state[action, 1])
            pid = int(self.state[process_id, 0])
            last_schedule_list = self.p_last_schedule

            pro_time = self.state[process_id, 2]
            insert_data = self.find_position(pid, did, process_id, pro_time)

            insert_data.append(step)
            insert_data.append(process_id)

            self.d_sc_list[did].append(insert_data)
            self.d_tasks[did].append([insert_data[2], insert_data[4]])
            self.p_tasks[pid].append([insert_data[2], insert_data[4]])
            # 算空隙时间和收集空隙
            hole_total_time = self.cal_hole(did)

            reward += (hole_total_time - self.d_total_idle_time[did])

            self.d_total_idle_time[did] = hole_total_time

            self.p_sc_list[pid].append([did, insert_data[2],
                                        insert_data[4], insert_data[6]])
            last_schedule_list[0][pid] += 1
            if last_schedule_list[0][pid] != 0:

                patient_idle_time = self.cal_p_idle(pid)

                # reward += (patient_idle_time - self.p_total_idle_time[pid])
                self.p_total_idle_time[pid] = patient_idle_time
                total_idle_time_p = np.sum(self.p_total_idle_time)
                self.total_idle_time_p = total_idle_time_p

            reward = reward/self.jobs_length.max()
            self.reward -= reward

            self.update_states(insert_data[2], pid, did, process_id)
        # print(reward)
        if sum(self.state[:, 4]) == 0:
            self.done = True

        return self.done, self.reward

    def update_states(self, start_time, pid, did, process_id):

        self.state[process_id, 4] = 0  # 改为不可分配状态
        self.state[process_id, 5] = start_time
        self.state[process_id, 6] = start_time
        idx_list = []
        temp_state = self.state[self.state[:, 4] == 1]
        idx_list.extend(temp_state[temp_state[:, 1] == did][:, 3].tolist())
        idx_list.extend(temp_state[temp_state[:, 0] == pid][:, 3].tolist())
        idx_list = list(set(idx_list))
        for idx in idx_list:
            insert_data = self.find_position(pid, did, int(idx), self.state[int(idx), 2])
            self.state[int(idx), 6] = insert_data[2]
        self.get_edge(pid, 0)
        self.get_edge(did, 1)
        # self.action_mask[did] -= 1
        # if self.action_mask[did] == 0:
        #     pass
        # else:
        #     self.candidate[did] = self.random_sort[did][int(self.action_mask[did])]

    # def get_total_time(self):
    #     total_time = 0
    #     for i in range(self.doctor.player_num):
    #         total_time += self.doctor.schedule_list[i][int(self.doctor.free_pos[i] - 1)][3]
    #     return total_time
    def get_edge(self, idx, col):
        temp_mtx = self.state[self.state[:, col] == idx].copy()
        earliest_start_times = np.unique(temp_mtx[:, 6])
        job_id_list = temp_mtx[:, 3].astype("int64")
        if col == 0:
            edge_mtx = self.j_edge_matrix
        else:
            edge_mtx = self.m_edge_matrix
        edge_mtx[job_id_list, :] = 0
        if len(earliest_start_times) == 1:
            # 生成所有可能的组合
            grid = np.meshgrid(job_id_list, job_id_list)
            combinations = np.vstack(grid).reshape(2, -1).T.astype("int64")
            edge_mtx[combinations[:, 0], combinations[:, 1]] = 1
        else:
            for i in range(1, len(earliest_start_times)):
                prev_job_list = temp_mtx[temp_mtx[:, 6] == earliest_start_times[i - 1]][:, 3]
                curr_job_list = temp_mtx[temp_mtx[:, 6] == earliest_start_times[i]][:, 3]
                grid = np.meshgrid(prev_job_list, curr_job_list)
                combinations = np.vstack(grid).reshape(2, -1).T.astype("int64")
                edge_mtx[combinations[:, 0], combinations[:, 1]] = 1

    def init_edge_matrix(self):
        for m_idx in range(self.machines):
            self.get_edge(m_idx, 1)
        for j_idx in range(self.jobs):
            self.get_edge(j_idx, 0)

    def cal_hole(self, did):
        hole_total_time = 0
        # ['pid', 'start_time', 'pro_time', 'finish_time', "step", "job_id"]
        sc = np.array(self.d_sc_list[int(did)])
        if len(sc) >= 1:
            idx = sc[:, 2].argsort(axis=0).reshape(1, -1)  # 按照start_time的值排序
            sc = sc[idx, :][0]
            hole_total_time += sc[0][2]
            hole_total_time += (sc[1:, 2] - sc[0:-1, 4]).sum()
        return hole_total_time

    def cal_p_idle(self, pid):
        total_idle_time = 0.
        # ["did", "start_time", "finish_time", "job_id"]
        sc = np.array(self.p_sc_list[pid])
        idx = sc[:, 1].argsort(axis=0).reshape(1, -1)  # 按照start_time的值排序
        sc = sc[idx, :][0]
        last_schedule_list = self.p_last_schedule
        total_idle_time += (sc[1:, 1] - sc[0:-1, 2]).sum()

        last_schedule_list[1][pid] = sc[len(sc) - 1][2]
        return total_idle_time

    def find_position(self, pid, did, job_id, pro_time):

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
    pass
