# -*- coding: utf-8 -*-
# @Time    : 2023/5/25 18:17
# @Author  : hxc
# @File    : verification.py
# @Software: PyCharm
import os

from params import Params
from verification_env import Environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical
# import gc
# import torch.optim as opt
import csv
from net.AC_model import AC
from net.DQN_model import DQN
# from multiprocessing import Queue
from threading import Thread
# from multiprocessing import Process

# from net.cnn import CNN
# from net.gcn_new import GCN
# from net.utils import get_now_date as hxc
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
        self.envs = [Environment(args) for _ in range(10)]

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
        self.model_name = f"300_10_AC_GAT"
        self.load_params(self.model_name)

        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda)

    def train(self):

        for i in range(len(self.files)):
            idle_list = []
            for env in self.envs:
                env.reset(f"../data/simulation_instances/{self.files[i]}")
            for episode in range(10):
                self.sum_reward = []
                t_list = []
                for j in range(self.args.env_num):
                    t = Thread(target=self.step, args=(self.envs[j], j))
                    t.start()
                    t_list.append(t)
                for thread in t_list:
                    thread.join()
                for env in self.envs:
                    # p_idle = np.sum(env.p_total_idle_time)
                    d_idle = np.sum(env.d_total_idle_time)

                    # total_idle_time = int(p_idle + d_idle)
                    # total_time = env.d_total_time + env.p_total_time
                    idle_list.append(d_idle)
                    env.reset(f"../data/simulation_instances/{self.files[i]}")
            print(self.files[i])
            print("Mean:", np.mean(idle_list))
            print("Std:", np.std(idle_list))
            confidence_interval = np.percentile(np.array(idle_list), [2.5, 97.5])
            print("CI:", confidence_interval)

    def step(self, env, i):
        for step in range(100000):
            data = env.state[:, [2, 4, 5, 6]].copy()
            data[:, [0, 2, 3]] = data[:, [0, 2, 3]] / (env.jobs_length.mean()*2)
            m_edge_index = coo_matrix(env.m_edge_matrix)
            m_edge_index = np.array([m_edge_index.row, m_edge_index.col])
            np.fill_diagonal(env.j_edge_matrix, 0)
            j_edge_index = coo_matrix(env.j_edge_matrix)
            j_edge_index = np.array([j_edge_index.row, j_edge_index.col])
            edge_index = np.concatenate([m_edge_index, j_edge_index], axis=1)
            data = torch.tensor(data, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index.astype("int64")).to(device)
            # candidate = torch.tensor(env.candidate.copy().astype("int64")).to(device)
            data = Data(x=data, edge_index=edge_index, num_nodes=len(data))
            if self.policy == "dqn":
                value, log_prob = 0, 0
                action, q, _ = self.choose_action(data, env)
            else:
                q = 0
                action, value, log_prob = self.choose_action(data, env)

            done, reward = env.step(action, step)
            if done:
                break

    def choose_action(self, data, env):
        if self.policy == "dqn":
            logits, prob = self.model(data)
            mask = (env.state[:, 4] == 1)
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
            mask = (env.state[:, 4] == 1)
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
    trainer = Trainer(args)
    trainer.train()
    # trainer.save_model(trainer.model_name)
    # # trainer.save_reward_loss("r_l")
    # # trainer.save_data("result")
    print("training finished!")