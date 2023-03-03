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
# from net.cnn import CNN
# from net.gcn_new import GCN
# from net.utils import get_now_date as hxc
from core.buffers import BatchBuffer
from core.rl_algorithms import PPOClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.env = Environment(args)
        self.env.reset()
        self.patient = self.env.patients
        self.doctor = self.env.doctor
        self.model = GCN(self.env.reg_num, self.patient, self.doctor).to(device)

        # self.load_params()
        self.ppo = PPOClip(self.model, device, args)

        self.batch_buffer = BatchBuffer(buffer_num=args.batch_num, gamma=args.gamma, lam=args.gae_lambda)
        # self.param = (self.model.parameters())
        # self.optimizer = opt.Adam(self.param, lr=args.lr)
        self.total_time = 0
        self.min_idle_time = 999999
        self.min_total_time = 1770
        self.scheduled_data = []
        self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []
        self.idle_total = []
        self.episode = 0

    def train(self):
        env = self.env

        steps = 0
        update_episode = 0
        for episode in range(1, self.args.episode + 1):

            sum_reward = []
            for step in range(self.args.num_steps):
                steps += 1

                data, edge = self.get_data_edge()
                process_state = data[0]
                multi_reg_state = data[1]
                d_state = data[2]
                action = self.model.choose_action(process_state, multi_reg_state, d_state, edge)

                done, reward = env.step(action, step)
                # print(reward, step)

                self.terminal_list.append(done)
                self.reward_list.append(reward)

                max_steps = (steps >= self.args.max_steps)

                if done or max_steps:
                    if done:
                        self.model.value_list.append(torch.tensor([0]).view(1, 1).to(device))
                    else:
                        data, edge = self.get_data_edge()

                        process_state = torch.tensor(data[0], dtype=torch.float32).to(device)
                        multi_reg_state = torch.tensor(data[1], dtype=torch.float32).to(device)
                        d_state = torch.tensor(data[2], dtype=torch.float32).to(device)
                        # data = torch.tensor(data).to(device)
                        edge_index = torch.tensor(edge).to(device)
                        _, value, _, _ = self.model(process_state, multi_reg_state, d_state, edge_index)
                        self.model.value_list.append(value)

                    self.batch_buffer.add_batch_data(self.model.state_list,
                                                     self.model.edge_list,
                                                     self.model.action_list, self.reward_list,
                                                     self.terminal_list, self.model.value_list,
                                                     self.model.log_prob_list, update_episode)
                    sum_reward.append(sum(self.reward_list))
                    update_episode += 1
                    self.model.reset()
                    self.reward_list = []
                    self.terminal_list = []

                    p_idle = np.sum(self.patient.total_idle_time)
                    d_idle = np.sum(self.doctor.total_idle_time)
                    total_idle_time = p_idle + d_idle

                    total_time = self.env.d_total_time+self.env.p_total_time
                    self.idle_total.append([d_idle, p_idle, total_idle_time,
                                            self.env.d_total_time, self.env.p_total_time, total_time, episode])

                    if total_time < self.min_total_time:
                        self.min_total_time = total_time
                        self.scheduled_data = []
                        for did in range(self.doctor.player_num):
                            sc = self.env.doctor.schedule_list[did]
                            for i in range(int(self.doctor.free_pos[did])):
                                item = [int(did), sc[i][0], sc[i][1], sc[i][2], sc[i][3], sc[i][4]]
                                self.scheduled_data.append(item)
                        self.file_name = f"{int(self.env.d_total_time)}_{int(self.env.p_total_time)}_{int(total_time)}"
                        self.save_data(self.file_name)

                    if update_episode == self.args.batch_num:

                        self.batch_buffer.get_data()
                        loss = 0.
                        for _ in range(0, self.args.batch_num, self.args.update_size):
                            # self.env.reset()
                            batch_states, batch_edges, batch_actions, batch_returns, \
                                batch_values, batch_log_prob, batch_adv = \
                                self.batch_buffer.get_mini_batch(self.args.update_size)
                            loss = self.ppo.learn(batch_states, batch_edges, batch_actions,
                                                  batch_returns, batch_values, batch_log_prob, batch_adv)
                            # print("return:", torch.sum(batch_returns))
                        if episode % 30 == 0:
                            print("mean_reward:", np.mean(np.array(sum_reward)), episode)
                            print("loss:", loss.item())
                        self.r_l.append([np.mean(np.array(sum_reward)), loss.item(), episode])
                        self.batch_buffer.__init__(buffer_num=self.args.batch_num,
                                                   gamma=self.args.gamma, lam=self.args.gae_lambda)
                        update_episode = 0
                        # print("episode:", episode)
                        # print("总时间：", self.env.get_total_time())
                    if episode % 120 == 0:
                        self.episode = episode
                        self.save_model()
                        self.save_info(self.r_l, f"r_l", ['reward', 'loss', 'ep'], "r_l")
                        self.save_info(self.idle_total, f"i_t", ['d_idle', 'p_idle', 'idle',
                                                                 'total_d', 'total_p', 'total', 'ep'], "i_t")
                        self.r_l = []
                        self.idle_total = []

                        # self.save_reward_loss("r_l")
                    steps = 0

                    # self.total_time = self.env.get_total_time()
                    self.env.reset()

                    break

    def save_model(self):
        # torch.save(self.model.actor.state_dict(), f'./net/params/actor.pth')
        # torch.save(self.model.critic.state_dict(), f'./net/params/critic.pth')
        torch.save(self.model.state_dict(), f'./net/params/gat.pth')

    def load_params(self):
        self.model.load_state_dict(torch.load("./net/params/gat.pth"))

    def save_data(self, file_name):
        with open(f'./data/save_data/{file_name}.csv', mode='w+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            headers = ['did', 'pid', 'start_time', 'pro_time', 'finish_time', "step"]
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
