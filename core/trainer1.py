# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 0:13
# @Author  : hxc
# @File    : trainer1.py
# @Software: PyCharm
from core.env import Environment
import numpy as np
import torch

# import gc
# import torch.optim as opt
import csv
from torch.distributions.categorical import Categorical
from net.gcn import GCN
# from multiprocessing import Queue
from threading import Thread
# from multiprocessing import Process
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
# from net.cnn import CNN
# from net.gcn_new import GCN
# from net.utils import get_now_date as hxc
from core.buffers import BatchBuffer
from scipy.sparse import coo_matrix
from core.rl_algorithms import PPOClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.env = Environment(args)
        self.env.reset()
        self.jobs = self.env.jobs
        self.machines = self.env.machines
        self.model = GCN(self.jobs, self.machines).to(device)

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
        self.scheduler = StepLR(self.ppo.optimizer, step_size=240, gamma=0.81)
        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda)

    def train(self):
        for episode in range(1, self.args.episode + 1):
            self.env.reset()
            self.sum_reward = []
            buffer = self.buffer.buffer_list[0]
            done = False
            for step in range(self.jobs * self.machines * 5):
                data = self.env.state
                edge_index = coo_matrix(self.env.edge_matrix)
                edge_index = np.array([edge_index.row, edge_index.col])
                data = torch.tensor(data, dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_index.astype("int64")).to(device)

                data = Data(x=data, edge_index=edge_index, num_nodes=len(data))

                action, value, log_prob = self.choose_action(data)

                done, reward = self.env.step(action, step)
                self.buffer.buffer_list[0].add_data(data, action, reward, done, value, log_prob)
                if done:
                    break

            if done:
                buffer.value_list.append(torch.tensor([0]).view(1, 1).to(device))
            else:
                data = self.env.state
                edge_index = coo_matrix(self.env.edge_matrix)
                edge_index = np.array([edge_index.row, edge_index.col])
                data = torch.tensor(data, dtype=torch.float32).to(device)
                edge_index = torch.tensor(edge_index.astype("int64")).to(device)

                data = Data(x=data, edge_index=edge_index, num_nodes=len(data))
                _, value = self.model(data)
                buffer.value_list.append(value.view(1, 1).detach().to(device))
            buffer.compute_reward_to_go_returns_adv()
            self.sum_reward.append(sum(buffer.reward_list))

            # update net
            self.buffer.get_data()
            mini_buffer = self.buffer.get_mini_batch(128, self.args.update_num)
            loss = 0
            for i in range(0, self.args.update_num):
                # self.env.reset()

                buf = mini_buffer[i]
                loss = self.ppo.learn(buf)
            self.buffer.reset()

            self.r_l.append([np.mean(self.sum_reward), loss.item(), episode])

            self.scheduler.step()

            # print("episode:", episode)
            # print("总时间：", self.env.get_total_time())
            if episode % 1 == 0:
                print("loss:", loss.item())
                print("mean_reward:", np.mean(self.sum_reward), episode)
            if episode % 120 == 0:
                self.episode = episode
                # self.save_model(self.model_name)
                # self.save_info(self.r_l, f"r_l_{self.model_name}",
                #                ['reward', 'loss', 'ep'], "r_l")
                # self.save_info(self.idle_total, f"i_t_{self.model_name}",
                #                ['d_idle', 'p_idle', 'idle', 'total_d', 'total_p', 'total', 'ep'], "i_t")
                self.r_l = []
                self.idle_total = []

    def choose_action(self, data):

        prob, value = self.model(data)

        mask = (self.env.action_mask != self.machines)
        mask = torch.from_numpy(mask).to(device)
        # 将无效动作对应的概率值设置为0
        masked_probs = prob * mask

        # 将有效动作的概率值归一化
        valid_probs = masked_probs / masked_probs.sum()

        policy_head = Categorical(probs=valid_probs)

        action = policy_head.sample()

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
            if self.episode == 120:
                csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            print(f'{file_name}')
