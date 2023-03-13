# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
# import random
# import matplotlib.pyplot as plt
import random
from torch.distributions.categorical import Categorical
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
        self.doctor = Doctor(self.reg_file, args)
        self.doctor.init_doc_info()
        self.patients = Patient(self.reg_file, args, self.doctor.player_num)
        self.patients.init_patient_info()
        # self.p_nodes = np.zeros((self.patients.player_num, self.))
        # self.d_nodes = np.ones((self.doctor.player_num, 1))
        # self.nodes = np.concatenate([self.p_nodes, self.d_nodes])
        self.nodes = np.zeros((self.patients.player_num + self.doctor.player_num, self.reg_num))

        # self.reg_nodes = np.zeros((self.reg_num, 4))

        self.edge = None
        self.edge_attr = np.zeros((self.reg_num, 4))
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
        self.edge_mask = None
        self.p_sc_jobs = None
        self.d_sc_jobs = None
        self.d_tasks = None
        self.p_tasks = None
        self.tasks = None

        self.idle_total = []

    def reset(self):
        torch.manual_seed(random.randint(1000, 5000))
        self.d_total_time = 0
        self.p_total_time = 0
        self.p_s_f = np.zeros((2, self.patients.player_num))
        self.doctor.reset()
        self.patients.reset()
        self.done = False
        self.reg_detail = shuffle(self.reg_file)
        self.reg_detail["id"] = np.arange(0, self.reg_detail.shape[0])

        self.cla_by_did = self.reg_detail.groupby("did")
        self.cla_by_pid = self.reg_detail.groupby("pid")
        self.patients.get_job_id_list(self.cla_by_pid)
        # self.patients.get_multi_reg_edge()
        # self.patients.edge = np.array(self.patients.edge).T
        # self.patients.edge[0, :] += self.reg_num
        self.doctor.get_job_id_list(self.cla_by_did)
        # self.doctor.get_edge()
        # self.doctor.edge = np.array(self.doctor.edge).T
        # self.doctor.edge[0, :] += self.reg_num + len(self.patients.multi_reg_pid)
        self.p_sc_jobs = self.patients.reg_job_id_list.copy()
        self.d_sc_jobs = self.doctor.reg_job_id_list.copy()
        self.hole_total_time = 0

        # self.reg_nodes = np.zeros((self.reg_num, 4))
        # self.reg_nodes[:, 3] = self.reg_detail["id"].values
        # self.reg_nodes[:, 2] = self.reg_detail["pro_time"].values / (self.max_time/self.doctor.player_num)
        # self.reg_nodes[:, 0] = 1
        self.edge_attr = np.zeros((self.reg_num, 4), dtype="float32")
        self.edge_attr[:, 0] = 1
        self.nodes = np.zeros((self.patients.player_num + self.doctor.player_num, self.reg_num))
        self.get_edge()
        self.d_tasks = [[] for _ in range(self.doctor.player_num)]
        self.p_tasks = [[] for _ in range(self.patients.player_num)]
        self.tasks = []
        self.idle_total = []

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
                # self.doctor.state[d_action][0] = insert_data[3] / (self.max_time/self.doctor.player_num)  # 修改医生工作总时间
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
            self.d_tasks[d_action].append([insert_data[1], insert_data[3]])
            self.p_tasks[p_action].append([insert_data[1], insert_data[3]])
            self.patients.reg_num_list[p_action] -= 1
            # self.done_node.append(action)
            # self.render_data.append([d_action, p_action, insert_data[1], insert_data[2], insert_data[3]])
            # self.render()
            # 算空隙时间和收集空隙
            hole_total_time = self.cal_hole(d_action)
            # reward += hole_total_time
            # self.hole_total_time = hole_total_time

            reward += (hole_total_time - self.doctor.total_idle_time[d_action])

            self.doctor.total_idle_time[d_action] = hole_total_time
            # self.doctor.state[d_action][1] = self.doctor.total_idle_time[d_action] /
            # (self.max_time/self.doctor.player_num)

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

            # reward = 1 - (reward / (self.max_time/self.doctor.player_num))
            reward = - reward
            # reward = 1 - ((sum(self.doctor.total_idle_time) + sum(self.patients.total_idle_time)) / self.max_time)
            # reward = max(0., reward)

            self.update_states(action, insert_data[1], p_action, d_action)
        if sum(self.patients.reg_num_list) == 0:
            self.done = True

        return self.done, reward

    def update_states(self, job_id, start_time, p_index, d_index):
        self.edge_attr[job_id][0] = False
        self.edge_attr[job_id][1] = start_time
        self.nodes[p_index][job_id] = 0
        self.nodes[p_index+d_index][job_id] = 0
        # self.reg_nodes[job_id][0] = False
        # self.reg_nodes[job_id][1] = start_time / (self.max_time/self.doctor.player_num)

        if p_index in self.patients.multi_reg_pid:
            index = np.where(self.patients.multi_reg_pid == p_index)[0][0]
            self.patients.multi_patient_state[index][0] -= 1

        self.patients.action_mask[job_id] = False
        if d_index in self.patients.reg_list[p_index]:
            self.patients.reg_list[p_index].remove(d_index)

        self.patients.mask_matrix[d_index][p_index] = 0

    def get_total_time(self):
        total_time = 0
        for i in range(self.doctor.player_num):
            total_time += self.doctor.schedule_list[i][int(self.doctor.free_pos[i] - 1)][3]
        return total_time

    def get_edge(self):
        edge = []
        # p_edge = np.array(self.patients.edge).T
        # p_edge[0, :] += self.reg_num
        # d_edge = np.array(self.doctor.edge).T
        # d_edge[0, :] += self.reg_num + len(self.patients.multi_reg_pid)
        # self.edge = np.concatenate([p_edge, d_edge], axis=1).astype("int64")
        reg_data = self.reg_detail.values
        for i in range(reg_data.shape[0]):
            job_id = reg_data[i][3]
            pid = reg_data[i][0]
            did = reg_data[i][1]
            pro_time = reg_data[i][2]
            self.nodes[pid][job_id] = 1
            self.nodes[pid+did][job_id] = 1
            edge.append([reg_data[i][0], reg_data[i][1]])
            self.edge_attr[i][2] = pro_time
            self.edge_attr[i][3] = job_id
        self.edge = np.array(edge, dtype="int64").T
        self.edge[1] += self.patients.player_num

    def cal_hole(self, did):
        doc = self.doctor
        hole_total_time = 0
        # ['pid', 'start_time', 'pro_time', 'finish_time', "step", "job_id"]
        sc = np.array(doc.schedule_list[int(did)])
        if len(sc) >= 1:
            idx = sc[:, 1].argsort(axis=0).reshape(1, -1)  # 按照start_time的值排序
            sc = sc[idx, :][0]
            hole_total_time += sc[0][1]
            hole_total_time += (sc[1:, 1] - sc[0:-1, 3]).sum()
        return hole_total_time

    def cal_p_idle(self, sc, pid):
        total_idle_time = 0.
        patients = self.patients
        # ["did", "start_time", "finish_time", "job_id"]
        sc = np.array(sc)
        idx = sc[:, 1].argsort(axis=0).reshape(1, -1)  # 按照start_time的值排序
        sc = sc[idx, :][0]
        last_schedule_list = patients.last_schedule
        total_idle_time += (sc[1:, 1] - sc[0:-1, 2]).sum()

        last_schedule_list[1][pid] = sc[len(sc) - 1][2]
        return total_idle_time

    def find_position(self, pid, did, job_id):
        doc = self.doctor
        # 当前医生的排班列表位置
        # d_free_pos = int(doc.free_pos[did])
        # patient = self.patients
        # 处理时间
        pro_time = self.reg_detail.values[job_id][2]

        self.tasks = []
        # 该医生的排班列表
        schedule_list = doc.schedule_list[int(did)]
        # 排班列表的最后结束时间
        last_time = 0
        if schedule_list:
            schedule_list = np.array(schedule_list)
            last_time = schedule_list[:, 3].max()
            self.tasks.extend(self.d_tasks[did])

        # 该病人的排班情况
        last_schedule_list = self.patients.last_schedule

        # 如果该病人被分配过
        if last_schedule_list[0][pid] != 0:
            self.tasks.extend(self.p_tasks[pid])
        holes = self.find_idle_times(pro_time)
        if holes:
            insert_data = [pid, holes[0][0], pro_time, holes[0][0] + pro_time]
        else:
            if last_time <= last_schedule_list[1][pid]:
                insert_data = [pid, last_schedule_list[1][pid], pro_time, last_schedule_list[1][pid] + pro_time]
            else:
                insert_data = [pid, last_time, pro_time, last_time + pro_time]
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

    def choose_action(self, data, edge, edge_attr, model):

        prob, value = model(data, edge, edge_attr)
        mask = torch.from_numpy(self.patients.action_mask).view(1, -1).to(device)
        prob[~mask] = 0
        prob = prob / prob.sum()
        policy_head = Categorical(probs=prob)
        action = policy_head.sample()

        return action.item(), value, policy_head.log_prob(action)


if __name__ == '__main__':
    pass
