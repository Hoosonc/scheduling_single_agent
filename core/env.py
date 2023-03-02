# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:13
# @Author : hxc
# @File : env.py
# @Software : PyCharm
# import random
import matplotlib.pyplot as plt
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
        self.patients.state = np.concatenate([self.patients.state,
                                              self.reg_detail["pid"].values.reshape(-1, 1)],
                                             axis=1)
        self.cla_by_did = self.reg_detail.groupby("did")
        self.cla_by_pid = self.reg_detail.groupby("pid")
        self.patients.get_job_id_list(self.cla_by_pid)
        self.doctor.get_job_id_list(self.cla_by_did)
        self.hole_total_time = 0
        self.render_data = []
        self.color = []
        self.edge = self.get_edge()
        # color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        # for j in range(self.patients.player_num):
        #     col = "#"
        #     c = np.random.choice(a=color_arr, size=6, replace=True)
        #     for i in range(6):
        #         col = col + c[i]
        #     self.color.append(col)

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

            reward += (hole_total_time - self.doctor.total_idle_time[d_action])

            self.doctor.total_idle_time[d_action] = hole_total_time
            self.patients.schedule_info[p_action].append([d_action, insert_data[1],
                                                          insert_data[3], insert_data[5]])

            last_schedule_list[0][p_action] += 1
            if last_schedule_list[0][p_action] != 0:
                patient_idle_time = self.cal_p_idle(
                    self.patients.schedule_info[p_action], p_action)

                reward += (patient_idle_time - self.patients.total_idle_time[p_action])
                self.patients.total_idle_time[p_action] = patient_idle_time
                total_idle_time_p = np.sum(self.patients.total_idle_time)
                self.total_idle_time_p = total_idle_time_p

            # reward = 1 - ((self.p_total_time + self.d_total_time) / (self.max_time * 2))
            # reward = 1 - (self.p_total_time / self.max_time)
            reward = 1 - (self.p_total_time / self.max_time)
            # reward = 1 -
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

    def get_edge_by_reg(self, all_reg_job_id_list, d_p):
        edge = []
        edge_attr = []
        for list_id in range(len(all_reg_job_id_list)):
            tree = []
            if len(all_reg_job_id_list[list_id]) > 1:
                reg_list = all_reg_job_id_list[list_id].copy()
                if d_p == 1:
                    doc = self.doctor
                    did = list_id
                    if self.doctor.schedule_list[did]:
                        sc = pd.DataFrame(np.array(doc.schedule_list[int(did)]),
                                          columns=['pid', 'start_time',
                                                   'pro_time', 'finish_time',
                                                   "step", "job_id"]).sort_values("start_time").values
                    else:
                        sc = []
                else:
                    pid = list_id
                    if self.patients.schedule_info[pid]:
                        sc = pd.DataFrame(np.array(self.patients.schedule_info[pid]),
                                          columns=["did", "start_time", "finish_time",
                                                   "job_id"]).sort_values("start_time").values
                    else:
                        sc = []
                if len(sc) == 1:
                    for i in range(len(sc)):
                        if d_p == 1:
                            tree.append([int(sc[i][5])])
                            reg_list.remove(int(sc[i][5]))
                        else:
                            tree.append([int(sc[i][3])])
                            reg_list.remove(int(sc[i][3]))
                elif len(sc) > 1:
                    for i in range(len(sc)):
                        if d_p == 1:
                            if i == 0:
                                reg_list.remove(int(sc[i][5]))
                            else:
                                edge.append([int(sc[i-1][5]), int(sc[i][5])])
                                edge_attr.append((sc[i][1]-sc[i-1][3]) / self.max_time)
                                reg_list.remove(int(sc[i][5]))
                        else:
                            if i == 0:
                                reg_list.remove(int(sc[i][3]))
                            else:
                                edge.append([int(sc[i-1][3]), int(sc[i][3])])
                                edge_attr.append((sc[i][1]-sc[i-1][2])/self.max_time)
                                reg_list.remove(int(sc[i][3]))
                    if d_p == 1:
                        tree.append([int(sc[-1][5])])
                    else:
                        tree.append([int(sc[-1][3])])
                if reg_list:
                    tree.append(reg_list)
                for d in range(len(tree)):
                    if len(tree[d]) > 1:
                        for s in tree[d]:
                            for end in tree[d]:
                                if s != end and [s, end] not in edge:
                                    edge.append([s, end])
                                    edge_attr.append(0)
                    if d != 0:
                        for s in tree[d - 1]:
                            for end in tree[d]:
                                assert s != end
                                if s != end and [s, end] not in edge:
                                    edge.append([s, end])
                                    edge_attr.append(0)
        return edge, edge_attr

    def get_edge(self):
        edge = []
        edge_attr = []
        p_edge, p_edge_attr = self.get_edge_by_reg(self.patients.reg_job_id_list, 0)
        d_edge, d_edge_attr = self.get_edge_by_reg(self.doctor.reg_job_id_list, 1)
        edge.extend(p_edge)
        edge.extend(d_edge)
        edge_attr.extend(p_edge_attr)
        edge_attr.extend(d_edge_attr)

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
        d_free_pos = int(doc.free_pos[did])
        patient = self.patients
        # 处理时间
        pro_time = self.reg_detail.values[job_id][2]

        tasks = []
        # 该医生的排班列表
        schedule_list = doc.schedule_list[int(did)]
        # 排班列表的最后结束时间
        last_time = 0
        if schedule_list:
            for sc in schedule_list:
                tasks.append([sc[1], sc[3]])
                last_time = max(last_time, sc[3])

        # 该病人的排班情况
        last_schedule_list = self.patients.last_schedule
        # 获取该病人以排班情况的列表
        sch_info = patient.schedule_info[pid]

        # 如果该病人被分配过
        if last_schedule_list[0][pid] != 0:
            for sc in sch_info:
                tasks.append([sc[1], sc[2]])
        holes = self.find_idle_times(tasks, pro_time)
        if holes:
            insert_data = [pid, holes[0][0], pro_time, holes[0][0] + pro_time]
        else:
            if last_time <= last_schedule_list[1][pid]:
                insert_data = [pid, last_schedule_list[1][pid], pro_time, last_schedule_list[1][pid]+pro_time]
            else:
                insert_data = [pid, last_time, pro_time, last_time + pro_time]
        return insert_data

    def find_idle_times(self, tasks, pro_time):
        # 如果任务列表为空，返回空闲时间列表 []
        if not tasks:
            return []

        # 转换为 NumPy 数组并按开始时间排序
        tasks = np.array(tasks)
        tasks = tasks[np.argsort(tasks[:, 0])]

        idle_times = []
        prev_end_time = tasks[0, 1]
        for i in range(1, len(tasks)):
            curr_start_time, curr_end_time = tasks[i]
            if curr_start_time > prev_end_time and curr_start_time - prev_end_time >= pro_time:
                # 如果当前任务的开始时间在上一个任务结束时间之后，并且它们之间的空闲时间大于 Pro_time
                idle_times.append([prev_end_time, curr_start_time])
            prev_end_time = max(prev_end_time, curr_end_time)
        if tasks[0, 0] >= pro_time:
            # 如果第一个任务的开始时间大于等于 Pro_time，那么将时间轴从 0 开始
            idle_times.insert(0, [0, tasks[0, 0]])
        return idle_times

    def render(self):
        # data = pd.read_csv(path, header=None).drop([0]).values
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 画布设置，大小与分辨率
        plt.figure(figsize=(100, 30), dpi=30)
        # barh-柱状图换向，循坏迭代-层叠效果
        pm = [2, 4, 7, 9]
        for i in range(len(self.render_data)):
            a = 0
            if int(self.render_data[i][0]) in pm:
                a = 0
            plt.barh(int(self.render_data[i][0]), height=0.8, width=int(float(self.render_data[i][3])),
                     left=int(float(self.render_data[i][2])) + a,
                     color=self.color[int(float(self.render_data[i][1])) - 1])
            plt.text(int(float(self.render_data[i][2])) + a + (int(float(self.render_data[i][3])) / 2),
                     int(self.render_data[i][0]),
                     int(float(self.render_data[i][1])),
                     fontsize=30,
                     verticalalignment="center",
                     horizontalalignment="center"
                     )
        plt.title("甘特图", fontsize=165)
        plt.xlabel("time", fontsize=100)
        plt.tick_params(labelsize=60)
        y_labels = [i for i in range(self.doctor.player_num)]
        plt.yticks(range(self.doctor.player_num), y_labels, rotation=0, fontsize=100)
        plt.show()


if __name__ == '__main__':
    pass
