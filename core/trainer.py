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
            max_len = 0
            min_len = 999999
            for env in self.envs:
                for d_sc in env.d_sc_list:
                    sc = np.array(d_sc)
                    final_len = sc[:, 4].max()
                    max_len = max(final_len, max_len)
                    min_len = min(final_len, min_len)
                # p_idle = np.sum(env.patients.total_idle_time)
                # d_idle = np.sum(env.doctor.total_idle_time)
                # total_idle_time = int(p_idle + d_idle)
                # total_time = env.d_total_time + env.p_total_time
                # self.idle_total.append([d_idle, p_idle, total_idle_time,
                #                         env.d_total_time, env.p_total_time, total_time, episode])
                # # self.scheduled_data = []
                # # for did in range(self.doctor.player_num):
                # #     sc = env.doctor.schedule_list[did]
                # #     self.scheduled_data.extend(sc)
                # # self.file_name = f"/10_60/{int(d_idle)}_{int(env.d_total_time)}"
                # # self.save_data(self.file_name)
                #
                # if d_idle < self.min_idle_time:
                #     self.min_idle_time = d_idle
                #     self.scheduled_data = []
                #     for did in range(self.doctor.player_num):
                #         sc = env.doctor.schedule_list[did]
                #         self.scheduled_data.extend(sc)
                #     self.file_name = f"{int(d_idle)}_{int(env.d_total_time)}"
                #     self.save_data(self.file_name)
                env.reset()

            # update net
            self.buffer.get_data()
            mini_buffer = self.buffer.get_mini_batch(self.args.mini_size, self.args.update_num)
            loss = 0
            for i in range(0, self.args.update_num):
                # self.env.reset()

                buf = mini_buffer[i]
                loss = self.ppo.learn(buf)
            self.buffer.reset()

            self.r_l.append([np.mean(self.sum_reward), loss.item(), episode])

            # self.scheduler.step()

            # print("episode:", episode)
            # print("总时间：", self.env.get_total_time())
            if episode % 1 == 0:
                print("loss:", loss.item())
                print("max:", max_len)
                print("mean_reward:", np.mean(self.sum_reward), episode)
            if episode % 100 == 0:
                self.episode = episode
                self.save_model(self.model_name)
                # self.save_info(self.r_l, f"r_l_{self.model_name}",
                #                ['reward', 'loss', 'ep'], "r_l")
                # self.save_info(self.idle_total, f"i_t_{self.model_name}",
                #                ['d_idle', 'p_idle', 'idle', 'total_d', 'total_p', 'total', 'ep'], "i_t")
                self.r_l = []
                self.idle_total = []

    def step(self, env, i):
        buffer = self.buffer.buffer_list[i]
        done = False
        for step in range(self.jobs * self.machines * 5):
            data = env.state[:, 2:]

            edge_index = coo_matrix(env.edge_matrix)
            edge_index = np.array([edge_index.row, edge_index.col])
            data = torch.tensor(data, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index.astype("int64")).to(device)
            candidate = torch.tensor(env.candidate.copy().astype("int64")).to(device)
            data = Data(x=data, edge_index=edge_index, num_nodes=len(data))

            action, value, log_prob = self.choose_action(data, candidate, env)

            done, reward = env.step(action, step)
            self.buffer.buffer_list[i].add_data(data, action, reward, done, value, log_prob, candidate)
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
        self.sum_reward.append(sum(buffer.reward_list))

    def choose_action(self, data, candidate, env):

        prob, value, log_probs = self.model(data, candidate)

        mask = (env.action_mask < self.machines)
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
            if self.episode == 120:
                csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            print(f'{file_name}')
