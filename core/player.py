# -*- coding: utf-8 -*-
# @Time    : 2022/10/8 10:14
# @Author  : hxc
# @File    : player.py
# @Software: PyCharm
import torch
import torch.nn.functional as f
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Player:
    def __init__(self):
        self.file = None
        self.args = None
        self.time_length = None  # 每个工作需要的总处理时长
        self.max_time_op = 0  # 最大的处理时长
        self.max_time_total = 0  # 总时间最长的工作时间
        self.nb_legal_actions = 0  # 合法动作的数量
        self.total_idle_time = None  # 总间隔时间
        self.op_time = None  # 操作时间和
        self.state = None
        self.action_mask = None
        self.player_num = 0  # 角色数量
        self.cx = None
        self.hx = None
        self.pi_list = list()
        self.entropy_list = list()
        self.v_list = list()
        self.pi_old_list = list()
        self.rewards = list()
        self.cnn_state_dict = None
        self.mlp_state_dict = None
        self.mask_matrix = None
        self.idle_time_a = None
        self.idle_time_p = None
        self.reg_job_id_list = None
        self.edge = []

    def reset_(self):
        self.total_idle_time = np.zeros((self.player_num,))  # 总间隔时间
        self.op_time = np.zeros((self.player_num,))  # 操作时间和

        # self.idle_time_a = np.zeros((self.player_num,))
        # self.idle_time_p = np.zeros((self.player_num,))

    def get_job_id_list(self, d_p):
        self.reg_job_id_list = [[] for _ in range(self.player_num)]
        file = self.file.values
        for i in range(self.player_num):
            if d_p == 0:
                job_id_list = file[:, 3][file[:, 0] == i]
            else:
                job_id_list = file[:, 3][file[:, 1] == i]
            self.reg_job_id_list[i].extend(job_id_list.tolist())

    def load_param(self):
        pass

    def get_edge(self):
        self.edge = [[] for _ in range(self.player_num)]
        for i in range(self.player_num):
            if len(self.reg_job_id_list[i]) > 1:
                for s in self.reg_job_id_list[i]:
                    for end in self.reg_job_id_list[i]:
                        if s != end and [s, end] not in self.edge[i]:
                            self.edge[i].append([s, end])
