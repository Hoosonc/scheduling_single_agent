# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
# import random
# import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from sklearn.utils import shuffle
from core.doctor import Doctor
from core.patient import Patient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment:
    def __init__(self, args):
        # init origin attributes
        self.args = args
        self.doc_file = pd.read_csv(args.doc_path, encoding='utf-8-sig').fillna('')
        self.reg_file = pd.read_csv(args.reg_path, encoding='utf-8-sig').fillna('')
        self.cla_by_did = None
        self.cla_by_pid = None
        self.reg_num = self.reg_file.shape[0]
        self.doctor = Doctor(self.doc_file, args)
        self.doctor.init_doc_info()
        self.patients = Patient(self.reg_file, args, self.doctor.player_num)
        self.patients.init_patient_info()
        self.edge = None
        self.edge_attr = None
        self.reg_detail = None
        self.done = False
        self.pos = []
        self.hole_total_time = 0.
        self.total_idle_time_p = 0.
        self.render_data = []
        self.color = []
        self.start_nodes = []
        self.directed_edges = []
        self.done_node = []
        self.d_total_time = 0
        self.p_total_time = 0
        self.p_s_f = np.zeros((2, self.patients.player_num))
        self.max_time = self.reg_file["pro_time"].sum()
        self.edge_idle = None
        self.edge_connect = None
        self.p_sc_jobs = None
        self.d_sc_jobs = None

    def reset(self):
        self.d_total_time = 0
        self.p_total_time = 0
        self.p_s_f = np.zeros((2, self.patients.player_num))
        self.doctor.reset()
        self.patients.reset()
        self.done = False
        self.reg_detail = shuffle(self.reg_file)
        self.reg_detail["id"] = np.arange(0, self.reg_detail.shape[0])
        self.patients.state = np.concatenate([self.patients.state,
                                              self.reg_detail["pro_time"].values.reshape(-1, 1) / self.max_time],
                                             axis=1)
        # self.patients.state = np.concatenate([self.patients.state,
        #                                       self.reg_detail["pid"].values.reshape(-1, 1)],
        #                                      axis=1)
        self.cla_by_did = self.reg_detail.groupby("did")
        self.cla_by_pid = self.reg_detail.groupby("pid")
        self.patients.get_job_id_list(self.cla_by_pid)
        self.doctor.get_job_id_list(self.cla_by_did)
        self.p_sc_jobs = self.patients.reg_job_id_list.copy()
        self.d_sc_jobs = self.doctor.reg_job_id_list.copy()
        self.hole_total_time = 0
        self.edge_idle = np.zeros((self.reg_num, self.reg_num))
        self.edge_connect = np.zeros((self.reg_num, self.reg_num), dtype=bool)
        self.init_edge_connect(self.patients.reg_job_id_list)
        self.init_edge_connect(self.doctor.reg_job_id_list)
        self.edge = self.get_edge()

    def init_edge_connect(self, all_reg_job_list):
        for job_id_list in all_reg_job_list:
            for job_id_start in job_id_list:
                for job_id_end in job_id_list:
                    if job_id_start != job_id_end:
                        self.edge_connect[job_id_start, job_id_end] = True

    def step(self, action, step):
        reward = 0.
        p_action = self.reg_detail.values[action][0]
        d_action = self.reg_detail.values[action][1]
        if self.patients.mask_matrix[d_action][p_action]:

            doc = self.doctor
            last_schedule_list = self.patients.last_schedule

            insert_data = self.find_position(p_action, d_action, action)

            if self.doctor.free_pos[d_action] == 0:
                d_last_time = 0
            else:
                sc_d = pd.DataFrame(np.array(doc.schedule_list[int(d_action)]),
                                    columns=['pid', 'start_time', 'pro_time', 'finish_time',
                                             "step", "job_id"]).sort_values("start_time").values
                d_last_time = sc_d[int(self.doctor.free_pos[d_action]) - 1][3]
            if insert_data[3] > d_last_time:
                self.d_total_time += (insert_data[3] - d_last_time)

            if last_schedule_list[0][p_action] == 0:
                self.p_s_f[0][p_action] = insert_data[1]
                self.p_s_f[1][p_action] = insert_data[3]
            else:
                if last_schedule_list[0][p_action] == 1:
                    if insert_data[3] <= self.p_s_f[0][p_action]:
                        self.p_s_f[0][p_action] = insert_data[1]
                    elif insert_data[3] > self.p_s_f[1][p_action]:
                        self.p_s_f[1][p_action] = insert_data[3]
                    self.p_total_time += self.p_s_f[1][p_action] - self.p_s_f[0][p_action]
                elif last_schedule_list[0][p_action] > 1:
                    old_time = self.p_s_f[1][p_action] - self.p_s_f[0][p_action]
                    if insert_data[3] <= self.p_s_f[0][p_action]:
                        self.p_s_f[0][p_action] = insert_data[1]
                    elif insert_data[3] > self.p_s_f[1][p_action]:
                        self.p_s_f[1][p_action] = insert_data[3]
                    assert self.p_s_f[1][p_action] - self.p_s_f[0][p_action] - old_time >= 0
                    self.p_total_time += self.p_s_f[1][p_action] - self.p_s_f[0][p_action] - old_time

            insert_data.append(step)
            insert_data.append(action)
            doc.insert_patient(insert_data, d_action)
            self.patients.reg_num_list[p_action] -= 1
            # self.done_node.append(action)
            # self.render_data.append([d_action, p_action, insert_data[1], insert_data[2], insert_data[3]])
            # self.render()
            # 算空隙时间和收集空隙
            hole_total_time = self.cal_hole(d_action)
            # reward += hole_total_time
            # self.hole_total_time = hole_total_time

            # reward += (hole_total_time - self.doctor.total_idle_time[d_action])

            self.doctor.total_idle_time[d_action] = hole_total_time
            self.patients.schedule_info[p_action].append([d_action, insert_data[1],
                                                          insert_data[3], insert_data[5]])

            last_schedule_list[0][p_action] += 1
            if last_schedule_list[0][p_action] != 0:
                patient_idle_time = self.cal_p_idle(
                    self.patients.schedule_info[p_action], p_action)

                # reward += (patient_idle_time - self.patients.total_idle_time[p_action])
                self.patients.total_idle_time[p_action] = patient_idle_time
                total_idle_time_p = np.sum(self.patients.total_idle_time)
                self.total_idle_time_p = total_idle_time_p

            reward = 1 - ((self.p_total_time + self.d_total_time) / (self.max_time * 2))
            # reward = 1 - (self.p_total_time / self.max_time)
            # reward = 1 - (self.p_total_time / self.max_time)
            reward = max(0., reward)

            self.update_states(action, insert_data[1], p_action, d_action)
        if sum(self.patients.reg_num_list) == 0:
            self.done = True

        return self.done, reward

    def update_states(self, job_id, start_time, p_index, d_index):
        did = d_index
        self.patients.state[job_id][1] = start_time / self.max_time
        # self.patients.state[job_id][2] = finish / self.max_time * 2
        self.patients.action_mask[job_id] = False
        if did in self.patients.reg_list[p_index]:
            self.patients.reg_list[p_index].remove(did)

        self.patients.mask_matrix[d_index][p_index] = 0

    def get_total_time(self):
        total_time = 0
        for i in range(self.doctor.player_num):
            total_time += self.doctor.schedule_list[i][int(self.doctor.free_pos[i] - 1)][3]
        return total_time

    def get_edge(self):
        # 获取第一个矩阵中值为True的元素的索引
        true_indices = np.argwhere(self.edge_connect)
        # 提取行列索引
        rows, cols = true_indices[:, 0], true_indices[:, 1]
        edge = true_indices.T.astype("int64")
        edge_attr = self.edge_idle[rows, cols].astype("float32")

        return edge, edge_attr

    def cal_hole(self, did):
        doc = self.doctor
        hole_total_time = 0
        sc = pd.DataFrame(np.array(doc.schedule_list[int(did)]),
                          columns=['pid', 'start_time',
                                   'pro_time', 'finish_time',
                                   "step", "job_id"]).sort_values("start_time").values
        for index in range(int(doc.free_pos[did])):
            start_time = sc[index][1]
            if index == 0:
                hole_total_time += start_time
            else:
                last_time = sc[index - 1][3]
                assert (start_time - last_time) >= 0
                hole_total_time += (start_time - last_time)
        assert hole_total_time >= 0
        return hole_total_time

    def cal_p_idle(self, sc, pid):
        total_idle_time = 0.
        patients = self.patients
        sc = pd.DataFrame(np.array(sc), columns=["did", "start_time", "finish_time",
                                                 "job_id"]).sort_values("start_time").values
        last_schedule_list = patients.last_schedule

        for i in range(len(sc) - 1):
            total_idle_time += sc[i + 1][1] - sc[i][2]
            assert (sc[i + 1][1] - sc[i][2]) >= 0
        last_schedule_list[1][pid] = sc[len(sc) - 1][2]
        return total_idle_time

    def find_position(self, pid, did, job_id):
        doc = self.doctor
        # 当前医生的排班列表位置
        patient = self.patients
        # 处理时间
        pro_time = self.reg_detail.values[job_id][2]

        p_reg_list = self.p_sc_jobs[pid]
        d_reg_list = self.d_sc_jobs[did]
        p_reg_list.remove(job_id)
        d_reg_list.remove(job_id)

        self.edge_connect[:, job_id] = False
        self.edge_connect[job_id, :] = False
        self.edge_connect[job_id, d_reg_list] = True
        self.edge_connect[job_id, p_reg_list] = True
        # 该医生的排班列表
        schedule_list = doc.schedule_list[int(did)]
        # 排班列表的最后结束时间
        last_time = 0
        if schedule_list:
            schedule_list = np.array(schedule_list)
            last_time = schedule_list[:, 3].max()
            scheduled_job = schedule_list[:, 5]

            self.edge_connect[scheduled_job, :][:, d_reg_list] = False

            self.edge_connect[np.repeat(job_id, len(scheduled_job)), scheduled_job] = False
            last_job = np.where(schedule_list[:, 3] == schedule_list[:, 3].max())[0][0]
            self.edge_connect[last_job, job_id] = True

        # 该病人的排班情况
        last_schedule_list = self.patients.last_schedule

        finish_time = 0

        # 如果该病人被分配过
        if last_schedule_list[0][pid] != 0:
            # 获取该病人以排班情况的列表
            sch_info = patient.schedule_info[pid]
            sch_info = np.array(sch_info)
            scheduled_job = sch_info[:, 3]
            self.edge_connect[scheduled_job, :][:, d_reg_list] = False
            last_job_id = np.where(sch_info[:, 2] == sch_info[:, 2].max())[0][0]
            self.edge_connect[last_job_id, job_id] = True

            finish_time = sch_info[:, 2].max()
            if last_time > finish_time:
                last_job = np.where(sch_info[:, 2] == sch_info[:, 2].max())[0][0]
                self.edge_idle[int(last_job)][job_id] = ((last_time-finish_time)/self.max_time)
            elif last_time < finish_time:
                # if isinstance(schedule_list, list):
                #     schedule_list = np.array(schedule_list)
                if last_time == 0:
                    pass
                else:
                    last_job = np.where(schedule_list[:, 3] == schedule_list[:, 3].max())[0][0]
                    self.edge_idle[int(last_job)][job_id] = ((finish_time - last_time)/self.max_time)
        start_time = max(last_time, finish_time)
        insert_data = [pid, start_time, pro_time, start_time + pro_time]

        return insert_data


if __name__ == '__main__':
    pass
