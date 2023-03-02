# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 10:11
# @Author  : hxc
# @File    : patient.py
# @Software: PyCharm
import torch
import numpy as np
# from net.controller import Cnn
from core.player import Player
# from net.controller import MLP
# from net.controller import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Patient(Player):
    def __init__(self, reg_file, args, d_num):
        super(Patient, self).__init__()
        self.args = args
        self.file = reg_file
        self.reg_num = None  # 挂号数量, 不变
        self.reg_num_list = None  # 挂号数量, 变
        self.reg_list = None  # 挂号列表
        self.last_schedule = None
        self.mask_matrix = None
        self.schedule_info = None
        self.d_num = d_num
        self.state = None
        # self.reg_job_id_list = None

    def init_patient_info(self):
        self.player_num = self.file.groupby("pid").count().shape[0]
        self.state = np.zeros((self.file.shape[0], 2), dtype="float32")
        self.reg_num = np.zeros((self.player_num,))
        self.reg_job_id_list = [[] for _ in range(self.player_num)]
        self.schedule_info = [[] for _ in range(self.player_num)]
        for patient in self.file.values:
            pid = patient[0]
            self.reg_num[pid] += 1

        self.reg_list = [[] for _ in range(self.player_num)]  # [[每个人挂的半天号]]
        self.time_length = np.zeros((self.player_num,))
        for patient in self.file.values:
            pid = patient[0]
            did = patient[1]
            pro_time = patient[2]
            self.reg_list[pid].append(did)
            self.time_length[pid] += pro_time
            self.max_time_op = max(self.max_time_op, pro_time)
        self.max_time_total = self.time_length.max(initial=0)
        self.action_mask = np.ones((self.player_num,), dtype=bool)

    def reset(self):
        self.schedule_info = [[] for _ in range(self.player_num)]
        self.state = np.zeros((self.file.shape[0], 2), dtype="float32")
        self.action_mask = np.ones((self.file.shape[0],), dtype=bool)
        self.mask_matrix = np.zeros((self.d_num, self.player_num), dtype=bool)
        self.total_idle_time = np.zeros((self.player_num,))
        self.reg_num_list = np.array(self.reg_num.tolist()).reshape((self.player_num,))
        self.last_schedule = np.zeros((2, self.player_num))  # 上一个号的结束时间
        """
             [[已处理号数]
             [该病人上一个号结束的时间]]
        """
        self.reg_list = [[] for _ in range(self.player_num)]  # [[每个人挂的半天号]]
        for patient in self.file.values:
            pid = patient[0]
            did = patient[1]
            self.reg_list[pid].append(did)

        for i in range(len(self.reg_list)):
            for did in self.reg_list[i]:
                self.mask_matrix[int(did)][i] = 1
        self.state[:, :1] = self.action_mask.reshape(-1, 1)

        self.reset_()
