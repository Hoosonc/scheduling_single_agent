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
from core.buffers import BatchBuffer
from core.rl_algorithms import PPOClip
from core.DQN import DQN_update
from core.Actor_critic import AC_update
from net.AC_GCN import AC_GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.policy = self.args.policy
        torch.manual_seed(self.args.seed)
        self.envs = [Environment(self.args, env_id) for env_id in range(self.args.env_num)]
        for env in self.envs:
            env.reset()
        self.jobs = self.envs[0].jobs
        self.machines = self.envs[0].machines
        self.algorithm = None
        self.net_name = "GAT"
        if self.policy == "dqn":
            self.model = DQN().to(device)
            self.algorithm = DQN_update(self.model, device, self.args)
        else:

            if self.net_name == "GCN":
                self.model = AC_GCN().to(device)
            elif self.net_name == "GAT":
                self.model = AC().to(device)
            if self.policy == "ppo2":
                self.algorithm = PPOClip(self.model, device, self.args)
            else:
                self.algorithm = AC_update(self.model, device, self.args)

        self.scheduler = None

        self.scheduled_data = []
        # self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []

        self.idle_total = []
        self.episode = 0
        self.sum_reward = []
        self.model_name = f"{self.args.file_name}"
        # self.load_params(self.model_name)

        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda)

    def train(self):

        for episode in range(self.args.episode):

            self.sum_reward = []
            t_list = []
            # for i in range(self.args.env_num):
            #     self.step(self.envs[i], i)
            for i in range(self.args.env_num):
                t = Thread(target=self.step, args=(self.envs[i], i))
                t.start()
                t_list.append(t)
            for thread in t_list:
                thread.join()

            self.get_idle_time(episode)
            # update net
            if self.policy == "dqn":
                loss = self.algorithm.learn(self.buffer)
            else:
                if self.policy == "ppo2":
                    self.buffer.get_data()

                    mini_buffer = self.buffer.get_mini_batch(self.args.mini_size, self.args.update_num)
                    loss = 0
                    for i in range(0, self.args.update_num):
                        # self.env.reset()
                        buf = mini_buffer[i]
                        loss = self.algorithm.learn(buf)
                else:
                    loss = self.algorithm.learn(self.buffer.buffer_list[0])
            self.buffer.reset()

            # self.r_l.append([self.sum_reward[0], self.sum_reward[1], loss.item(), episode])
            self.r_l.append([self.sum_reward[0], loss.item(), episode])

            if self.scheduler is not None and (episode + 1) % 20 == 0:
                self.scheduler.step()

            # print("episode:", episode)
            # print("总时间：", self.env.get_total_time())
            if episode % 1 == 0:
                print("loss:", loss.item())
                # print("d_idle:", d_idle)
                print("sum_reward:", self.sum_reward[0], episode)
            if (episode + 1) % 5 == 0:
                self.episode = episode
                self.save_model(self.model_name)

                # self.save_info(self.r_l, f"r_l_{self.model_name}",
                #                ['reward', 'loss', 'ep'], "r_l")
                #
                # self.save_info(self.idle_total, f"i_t_{self.model_name}",
                #                ['d_idle', 'p_idle', 'idle', 'ep'], "i_t")
                self.r_l = []
                self.idle_total = []

    def step(self, env, i):
        buffer = self.buffer.buffer_list[i]
        done = False
        for step in range(300):
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
            if self.policy == "dqn":
                self.buffer.buffer_list[i].add_data(state_t=data, action_t=action, reward_t=reward,
                                                    terminal_t=done, q=q.view(1, -1))
            else:
                self.buffer.buffer_list[i].add_data(state_t=data, action_t=action, reward_t=reward,
                                                    terminal_t=done, value_t=value, log_prob_t=log_prob)
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
        if self.policy != "dqn":
            buffer.compute_reward_to_go_returns_adv()
        self.sum_reward.append(np.sum(buffer.reward_list))

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

    def get_idle_time(self, episode):
        p_idle_list = []
        d_idle_list = []
        total_idle_time_list = []
        total_idle = []
        for env in self.envs:
            p_idle = int(np.sum(env.p_total_idle_time))
            d_idle = int(np.sum(env.d_total_idle_time))

            total_idle_time = int(p_idle + d_idle)
            # total_time = env.d_total_time + env.p_total_time
            d_idle_list.append(d_idle)
            p_idle_list.append(p_idle)
            total_idle_time_list.append(total_idle_time)
            env.reset()
        p_sum_idle = sum(p_idle_list)
        p_mean_idle = np.mean(p_idle_list)
        d_sum_idle = sum(d_idle_list)
        d_mean_idle = np.mean(d_idle_list)
        p_idle_list.append(p_sum_idle)
        p_idle_list.append(p_mean_idle)
        d_idle_list.append(d_sum_idle)
        d_idle_list.append(d_mean_idle)
        total_idle.append(p_idle_list)
        total_idle.append(d_idle_list)
        total_idle.append(episode)
        self.idle_total.append(total_idle)

    def save_model(self, file_name):
        # torch.save(self.model.actor.state_dict(), f'./net/params/actor.pth')
        # torch.save(self.model.critic.state_dict(), f'./net/params/critic.pth')
        torch.save(self.model.state_dict(), f'./net/params/{file_name}_{self.policy}_{self.net_name}.pth')

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
        with open(f'./data/{path}/{file_name}_{self.policy}_{self.net_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            if (self.episode + 1) == 5:
                csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            # print(f'{file_name}')
