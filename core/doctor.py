# -*- coding : utf-8 -*-
# @Time :  2022/7/27 15:30
# @Author : hxc
# @File : doctor.py
# @Software : PyCharm
import torch
import numpy as np
# import pandas as pd
# from net.controller import Cnn
from core.player import Player
# from net.controller import MLP
# from net.controller import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Doctor(Player):
    def __init__(self, file, args):
        super(Doctor, self).__init__()
        self.args = args
        self.file = file
        self.reg_num = list()  # 不变  挂号数量
        self.reg_num_list = None  # 变
        self.schedule_list = None
        self.free_pos = None
        self.total_time = 0

    def init_doc_info(self):
        self.player_num = self.file.groupby("did").count().shape[0]
        all_doc = self.file.sort_values('did', ascending=True).groupby('did')
        self.reg_job_id_list = [[] for _ in range(self.player_num)]
        self.max_time_op = 0
        self.state = np.zeros((self.player_num, 2))

        for doc in all_doc:
            doc_info = doc[1]
            reg_num = doc_info.count()[0]
            self.reg_num.append(reg_num)

    def reset(self, reg_file):
        self.file = reg_file
        self.schedule_list = [[] for _ in range(self.player_num)]
        self.total_idle_time = np.zeros((self.player_num,))
        self.free_pos = np.zeros((self.player_num,))
        # self.state = np.zeros((self.player_num, 2))
        self.reg_num_list = np.array(self.reg_num.copy()).reshape((self.player_num,))
        self.get_job_id_list(1)
        self.get_edge()
        self.reset_()

    def insert_patient(self, insert_data, d_index):
        # axis = 0 删除选中行；axis = 1 删除选中列；
        # self.schedule_list[d_index] = np.delete(
        #     self.schedule_list[d_index],
        #     [delete_index], axis=1
        # )

        self.schedule_list[d_index].append(insert_data)
        self.free_pos[d_index] += 1
