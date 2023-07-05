# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 21:37
# @Author  : hxc
# @File    : verification3.py
# @Software: PyCharm

import os

import pandas as pd

from params import Params
from verification_env import Environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from net.AC_model import AC
from net.DQN_model import DQN
import time

from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from buffers import BatchBuffer
from net.AC_GCN import AC_GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.policy = args.policy
        self.files = os.listdir("../data/simulation_instances")
        torch.manual_seed(args.seed)
        self.env = Environment(args)

        self.algorithm = None
        self.net_name = "GAT"
        if self.policy == "dqn":
            self.model = DQN().to(device)
        else:

            if self.net_name == "GCN":
                self.model = AC_GCN().to(device)
            elif self.net_name == "GAT":
                self.model = AC().to(device)

        self.scheduled_data = []
        self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []
        self.idle_total = []
        self.episode = 0
        self.sum_reward = []
        # self.model_name = f"{self.jobs}_{self.machines}"
        self.model_name = f"300_10_{self.policy}_GAT"
        self.load_params(self.model_name)
        self.time_count = 0
        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda)

    def train(self):

        for i in range(len(self.files)):
            self.time_count = 0
            idle_list = []

            self.env.reset(f"../data/simulation_instances/{self.files[i]}")
            for episode in range(100):
                # 记录程序开始时间
                start_time = time.perf_counter()
                self.step(i)
                self.sum_reward = []

                # 记录程序开始时间
                end_time = time.perf_counter()
                self.time_count += end_time - start_time

                p_idle = np.sum(self.env.p_total_idle_time)
                d_idle = np.sum(self.env.d_total_idle_time)

                d_total_time = self.env.get_total_time()

                total_idle_time = int(p_idle + d_idle)
                # total_time = env.d_total_time + env.p_total_time
                idle_list.append([p_idle, d_idle, total_idle_time, d_total_time])
                self.env.reset(f"../data/simulation_instances/{self.files[i]}")
            print(self.policy, self.files[i], self.time_count)
            df = pd.DataFrame(data=idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
            df.to_csv(f"../data/simulation_results/result_{self.policy}_{self.files[i]}", index=False)
            # print("Mean:", np.mean(idle_list))
            # print("Std:", np.std(idle_list))
            # confidence_interval = np.percentile(np.array(idle_list), [2.5, 97.5])
            # print("CI:", confidence_interval)

    def step(self, i):
        # env = Environment(self.args)
        # env.reset(f"../data/simulation_instances/{self.files[i]}")
        for step in range(100000):
            data = self.env.state[:, [2, 4, 5, 6]].copy()
            data[:, [0, 2, 3]] = data[:, [0, 2, 3]] / (self.env.jobs_length.mean()*2)
            m_edge_index = coo_matrix(self.env.m_edge_matrix)
            m_edge_index = np.array([m_edge_index.row, m_edge_index.col])
            np.fill_diagonal(self.env.j_edge_matrix, 0)
            j_edge_index = coo_matrix(self.env.j_edge_matrix)
            j_edge_index = np.array([j_edge_index.row, j_edge_index.col])
            edge_index = np.concatenate([m_edge_index, j_edge_index], axis=1)
            data = torch.tensor(data, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index.astype("int64")).to(device)
            # candidate = torch.tensor(env.candidate.copy().astype("int64")).to(device)
            data = Data(x=data, edge_index=edge_index, num_nodes=len(data))
            if self.policy == "dqn":
                value, log_prob = 0, 0
                action, q, _ = self.choose_action(data)
            else:
                q = 0
                action, value, log_prob = self.choose_action(data)

            done, reward = self.env.step(action, step)
            if done:
                break

    def choose_action(self, data):
        if self.policy == "dqn":
            logits, prob = self.model(data)
            mask = (self.env.state[:, 4] == 1)
            mask = torch.from_numpy(mask).to(device).view(1, -1)
            # 将无效动作对应的概率值设置为0
            masked_probs = prob * mask

            # 将有效动作的概率值归一化
            valid_probs = masked_probs / masked_probs.sum(dim=1)

            action = torch.argmax(valid_probs, dim=1)
            q = logits[0][action]
            return action.item(), q, 0
        else:
            prob, value, log_probs = self.model(data)
            mask = (self.env.state[:, 4] == 1)
            mask = torch.from_numpy(mask).to(device).view(1, -1)
            # 将无效动作对应的概率值设置为0
            masked_probs = prob * mask

            # 将有效动作的概率值归一化
            valid_probs = masked_probs / masked_probs.sum(dim=1)

            policy_head = Categorical(probs=valid_probs.view(1, -1))
            # action = policy_head.sample()
            action = torch.argmax(valid_probs, dim=1)

            # log_prob = log_probs.view(self.jobs,)[action]

            return action.item(), value, policy_head.log_prob(action)

    def load_params(self, model_name):
        self.model.load_state_dict(torch.load(f"../net/params/{model_name}.pth"))


if __name__ == '__main__':
    args = Params().args
    print("The RL program starts training...")
    for alg in ["dqn"]:
        args.policy = alg
        trainer = Trainer(args)
        trainer.train()
    # trainer.save_model(trainer.model_name)
    # # trainer.save_reward_loss("r_l")
    # # trainer.save_data("result")
    print("training finished!")