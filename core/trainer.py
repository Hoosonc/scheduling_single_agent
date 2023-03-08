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

# import gc
# import torch.optim as opt
import csv
from net.gcn import GCN
# from multiprocessing import Queue
from threading import Thread
from torch.optim.lr_scheduler import StepLR
# from net.cnn import CNN
# from net.gcn_new import GCN
# from net.utils import get_now_date as hxc
# from core.buffers import BatchBuffer
from core.rl_algorithms import PPOClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        # self.env = Environment(args)
        self.envs = [Environment(args) for _ in range(args.env_num)]
        for env in self.envs:
            env.reset()
        self.patient = self.envs[0].patients
        self.doctor = self.envs[0].doctor
        self.reg_num = self.envs[0].reg_num
        node_num = self.patient.player_num + self.doctor.player_num

        self.model = GCN(self.envs[0].reg_num, node_num).to(device)

        self.ppo = PPOClip(self.model, device, args)

        # self.batch_buffer = BatchBuffer(buffer_num=self.args.env_num, gamma=args.gamma, lam=args.gae_lambda)
        # self.param = (self.model.parameters())
        # self.optimizer = opt.Adam(self.param, lr=args.lr)
        self.total_time = 0
        self.min_idle_time = 999999
        self.min_total_time = 1800
        self.min_d_idle = 1
        self.scheduled_data = []
        self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []
        self.idle_total = []
        self.episode = 0
        self.state_list = []
        self.edge_list = []
        self.action_list = []
        self.value_list = []
        self.log_prob_list = []
        self.sum_reward = []
        self.model_name = f"{self.doctor.player_num}_{self.patient.player_num}_{self.reg_num}"
        # self.load_params(self.model_name)
        self.scheduler = StepLR(self.ppo.optimizer, step_size=200, gamma=0.8)

    def train(self):
        # env = self.env
        for episode in range(1, self.args.episode + 1):
            self.sum_reward = []
            # queue = Queue()
            t_list = []
            for i in range(self.args.env_num):
                t = Thread(target=self.step, args=(self.envs[i], i))
                t.start()
                t_list.append(t)
            for thread in t_list:
                thread.join()

            # update net
            loss = 0
            for i in range(0, self.args.update_num, self.args.mini_batch):
                # self.env.reset()
                # batch_states, batch_edges, batch_actions, batch_returns, batch_values, batch_log_prob, batch_adv = \
                #     self.batch_buffer.get_mini_batch(i, self.args.mini_batch)
                buffer_list = self.envs[i:i + self.args.mini_batch]
                # loss = self.ppo.learn(batch_states, batch_edges, batch_actions,
                #                       batch_returns, batch_values, batch_log_prob, batch_adv)
                loss = self.ppo.learn(buffer_list)

            self.r_l.append([np.mean(self.sum_reward), loss.item(), episode])

            for env in self.envs:
                p_idle = np.sum(env.patients.total_idle_time)
                d_idle = np.sum(env.doctor.total_idle_time)
                total_idle_time = p_idle + d_idle
                total_time = env.d_total_time + env.p_total_time
                self.idle_total.append([d_idle, p_idle, total_idle_time,
                                        env.d_total_time, env.p_total_time, total_time, episode])
                if d_idle < self.min_d_idle:
                    self.min_d_idle = d_idle
                    scheduled_data = []
                    for did in range(self.doctor.player_num):
                        sc = env.doctor.schedule_list[did]
                        scheduled_data.extend(sc)
                    self.file_name = f"{int(d_idle)}_{int(env.d_total_time)}"
                    self.save_data(self.file_name)
                env.reset()

            self.scheduler.step()

            # print("episode:", episode)
            # print("总时间：", self.env.get_total_time())
            if episode % 30 == 0:
                print("loss:", loss)
                print("mean_reward:", np.mean(self.sum_reward))
            if episode % 120 == 0:
                self.episode = episode
                self.save_model(self.model_name)
                self.save_info(self.r_l, f"r_l_{self.model_name}",
                               ['reward', 'loss', 'ep'], "r_l")
                self.save_info(self.idle_total, f"i_t_{self.model_name}",
                               ['d_idle', 'p_idle', 'idle', 'total_d', 'total_p', 'total', 'ep'], "i_t")
                self.r_l = []
                self.idle_total = []

    def step(self, env, i):
        for step in range(self.reg_num):
            data, edge = env.nodes, env.edge
            edge_attr = env.edge_attr

            env.state_list.append(data)
            env.edge_list.append(edge)
            env.edge_attr_list.append(edge_attr)

            action, value, log_prob = self.model.choose_action(data, edge, edge_attr, env)
            env.action_list.append(action)
            env.value_list.append(value.detach())
            env.log_prob_list.append(log_prob.detach())

            done, reward = env.step(action, step)
            env.terminal_list.append(done)
            env.reward_list.append(reward)
        self.compute_reward_to_go_returns_adv(env)
        self.sum_reward.append(sum(env.reward_list))
        # data_tuple = [i, env]
        # que.put(data_tuple)

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

    def get_data_edge(self):
        all_data = []
        data = self.patient.state.astype("float32")
        edge = self.env.get_edge()
        edge = edge.astype("int64")
        all_data.append(data)
        all_data.append(self.patient.multi_patient_state)
        all_data.append(self.doctor.state)
        return all_data, edge

    def compute_reward_to_go_returns_adv(self, env):
        """
            the env will reset directly once it ends and return a new state
            st is only one more than at and rt at the end of the episode
            state:    s1 s2 s3 ... st-1 -
            action:   a1 a2 a3 ... at-1 -
            reward:   r1 r2 r3 ... rt-1 -
            terminal: t1 t2 t3 ... tt-1 -
            value:    v1 v2 v3 ... vt-1 vt
        """
        # (N,T) -> (T,N)   N:n_envs   T:trajectory_length

        rewards = torch.from_numpy(np.array(env.reward_list)).to(device).detach().view(1, -1)
        values = torch.cat([value for value in env.value_list], dim=0).view(1, -1)
        log_prob = torch.cat([log_p for log_p in env.log_prob_list], dim=0).view(1, -1)
        env.log_prob = log_prob
        terminals = torch.from_numpy(np.array(env.terminal_list, dtype=int)).to(device).detach().view(1, -1)
        rewards = torch.transpose(rewards, 1, 0)
        values = torch.transpose(values, 1, 0)
        terminals = torch.transpose(terminals, 1, 0)
        r = values[-1]
        returns = []
        deltas = []
        for i in reversed(range(rewards.shape[0])):
            r = rewards[i] + (1. - terminals[i]) * self.args.gamma * r
            returns.append(r.view(-1, 1))

            v = rewards[i] + (1. - terminals[i]) * self.args.gamma * values[i + 1]
            delta = v - values[i]
            deltas.append(delta.view(1, -1))
        env.returns = torch.cat(list(reversed(returns)), dim=1)

        deltas = torch.cat(list(reversed(deltas)), dim=0)
        advantage = deltas[-1, :]
        advantages = [advantage.view(1, -1)]
        for i in reversed(range(rewards.shape[0] - 1)):
            advantage = deltas[i] + (1. - terminals[i]) * self.args.gamma * self.args.gae_lambda * advantage
            advantages.append(advantage.view(1, -1))
        advantages = torch.cat(list(reversed(advantages)), dim=0).view(-1, rewards.shape[0])
        mean = torch.cat(
            [torch.full((1, advantages.shape[1]), m.item()) for m in torch.mean(advantages, dim=1)]).to(device)
        std = torch.cat(
            [torch.full((1, advantages.shape[1]), m.item() + 1e-8) for m in torch.std(advantages, dim=1)]).to(device)
        env.adv = ((advantages - mean) / (std + 1e-8))
