# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:13
# @Author : hxc
# @File : trainer.py
# @Software : PyCharm
# import numpy as np
# import math

from core.env import Environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical
# import gc
# import torch.optim as opt
import csv
from net.gcn import GCN
# from multiprocessing import Queue
from threading import Thread
# from multiprocessing import Process
from torch.optim.lr_scheduler import StepLR
# from net.cnn import CNN
# from net.gcn_new import GCN
# from net.utils import get_now_date as hxc
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from core.buffers import BatchBuffer
from core.rl_algorithms import PPOClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.envs = [Environment(args) for _ in range(args.env_num)]
        for env in self.envs:
            env.reset()
        self.jobs = self.envs[0].jobs
        self.machines = self.envs[0].machines
        self.model = GCN(self.machines, self.machines).to(device)

        self.ppo = PPOClip(self.model, device, args)
        self.total_time = 0
        self.min_idle_time = 1
        self.min_total_time = 1800
        self.min_d_idle = 1
        self.scheduled_data = []
        self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []
        self.idle_total = []
        self.episode = 0
        self.sum_reward = []
        self.model_name = f"{self.jobs}_{self.machines}"
        # self.load_params(self.model_name)
        # self.scheduler = StepLR(self.ppo.optimizer, step_size=240, gamma=0.81)
        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda)

    def train(self):
        # env = self.env
        for episode in range(1, self.args.episode + 1):
            self.sum_reward = []
            t_list = []
            for i in range(self.args.env_num):
                t = Thread(target=self.step, args=(self.envs[i], i))
                t.start()
                t_list.append(t)
            for thread in t_list:
                thread.join()
            idle_total_list = []
            for env in self.envs:

                p_idle = np.sum(env.p_total_idle_time)
                d_idle = np.sum(env.d_total_idle_time)

                total_idle_time = int(p_idle + d_idle)
                # total_time = env.d_total_time + env.p_total_time
                idle_total_list.append(d_idle)
                idle_total_list.append(p_idle)
                idle_total_list.append(total_idle_time)

                env.reset()
            idle_total_list.append(episode)
            self.idle_total.append(idle_total_list)

            # update net
            self.buffer.get_data()
            mini_buffer = self.buffer.get_mini_batch(self.args.mini_size, self.args.update_num)
            loss = 0
            for i in range(0, self.args.update_num):
                # self.env.reset()

                buf = mini_buffer[i]
                loss = self.ppo.learn(buf)
            self.buffer.reset()

            self.r_l.append([self.sum_reward[0], self.sum_reward[1], loss.item(), episode])

            # self.scheduler.step()

            # print("episode:", episode)
            # print("总时间：", self.env.get_total_time())
            if episode % 1 == 0:
                print("loss:", loss.item())
                print("d_idle:", d_idle)
                print("mean_reward:", self.sum_reward[0], episode)
            if episode % 100 == 0:
                self.episode = episode
                self.save_model(self.model_name)
                self.save_info(self.r_l, f"r_l_{self.model_name}",
                               ['reward1', 'reward2', 'loss', 'ep'], "r_l")
                self.save_info(self.idle_total, f"i_t_{self.model_name}",
                               ['d_idle', 'p_idle', 'idle', 'd_idle2', 'p_idle2', 'idle2', 'ep'], "i_t")
                self.r_l = []
                self.idle_total = []

    def step(self, env, i):
        buffer = self.buffer.buffer_list[i]
        done = False
        for step in range(self.jobs * self.machines * 5):
            data = env.state[:, [2, 4, 5]]

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

            action, value, log_prob = self.choose_action(data, env)

            done, reward = env.step(action, step)
            self.buffer.buffer_list[i].add_data(data, action, reward, done, value, log_prob)
            if done:
                break

        if done:
            buffer.value_list.append(torch.tensor([0]).view(1, 1).to(device))
        else:
            data = env.state
            edge_index = coo_matrix(env.edge_matrix)
            edge_index = np.array([edge_index.row, edge_index.col])
            data = torch.tensor(data, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index.astype("int64")).to(device)

            data = Data(x=data, edge_index=edge_index, num_nodes=len(data))
            _, value, _ = self.model(data)
            buffer.value_list.append(value.view(1, 1).detach().to(device))
        buffer.compute_reward_to_go_returns_adv()
        self.sum_reward.append(buffer.reward_list[-1])

    def choose_action(self, data, env):

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

    def save_model(self, file_name):
        # torch.save(self.model.actor.state_dict(), f'./net/params/actor.pth')
        # torch.save(self.model.critic.state_dict(), f'./net/params/critic.pth')
        torch.save(self.model.state_dict(), f'./net/params/{file_name}.pth')

    def load_params(self, model_name):
        self.model.load_state_dict(torch.load(f"./net/params/{model_name}.pth"))

    def save_data(self, file_name):
        with open(f'./data/save_data/{file_name}.csv', mode='w+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            headers = ['did', 'pid', 'start_time', 'pro_time', 'finish_time', "step", "job_id"]
            csv_writer.writerow(headers)
            csv_writer.writerows(self.scheduled_data)
            print(f'{file_name}')

    def save_info(self, data_list, file_name, headers, path):
        with open(f'./data/{path}/{file_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            if self.episode == 100:
                csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            print(f'{file_name}')
