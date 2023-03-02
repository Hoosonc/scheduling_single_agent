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
        # self.model = CNN(self.env.reg_num, self.patient.player_num,
        #                  self.doctor.player_num, self.patient, self.doctor).to(device)
        # self.load_params()
        self.ppo = PPOClip(self.model, device, args)

        self.batch_buffer = BatchBuffer(buffer_num=args.batch_num, gamma=args.gamma, lam=args.gae_lambda)
        # self.param = (self.model.parameters())
        # self.optimizer = opt.Adam(self.param, lr=args.lr)
        self.total_time = 0
        self.min_idle_time = 999999
        self.min_total_time = 1943
        self.scheduled_data = []
        self.file_name = ""
        self.reward_list = []
        self.terminal_list = []
        self.r_l = []
        self.idle_total = []

    def train(self):
        env = self.env

        steps = 0
        update_episode = 0
        for episode in range(1, self.args.episode + 1):

            sum_reward = []
            for step in range(self.args.num_steps):
                steps += 1

                data, edge, edge_attr = self.get_data_edge()
                action = self.model.choose_action(data, edge, edge_attr)
                # CNN 方案
                # data, edge = self.get_data_edge()
                # action = self.model.choose_action(data)

                done, reward = env.step(action, step)
                # print(reward, step)

                self.terminal_list.append(done)
                self.reward_list.append(reward)

                max_steps = (steps >= self.args.max_steps)

                if done or max_steps:
                    if done:
                        self.model.value_list.append(torch.tensor([0]).view(1, 1).to(device))
                    else:
                        data, edge, edge_attr = self.get_data_edge()
                        data = torch.tensor(data).to(device)
                        edge_index = torch.tensor(edge).to(device)
                        edge_attr = torch.tensor(edge_attr).to(device)
                        _, value, _, = self.model(data, edge_index, edge_attr)
                        self.model.value_list.append(value)

                    self.batch_buffer.add_batch_data(self.model.state_list,
                                                     self.model.edge_list,
                                                     self.model.edge_attr_list,
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
                    # assert total_idle_time == self.env.edge_idle.sum()*self.env.max_time
                    # if total_idle_time < self.min_idle_time:
                    #     self.min_idle_time = total_idle_time
                    #     self.scheduled_data = []
                    #     for did in range(self.doctor.player_num):
                    #         sc = self.env.doctor.schedule_list[did]
                    #         for i in range(int(self.doctor.free_pos[did])):
                    #             item = [int(did), sc[i][0], sc[i][1], sc[i][2], sc[i][3], sc[i][4], sc[i][5]]
                    #             self.scheduled_data.append(item)
                    #     self.file_name = f"{int(float(total_idle_time))}_" \
                    #                      f"{int(float(p_idle))}_{int(float(d_idle))}_{episode}"
                    #     self.save_data(self.file_name)
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
                        self.file_name = f"{int(self.env.d_total_time)}_{int(self.env.p_total_time)}_{int(total_time)}_{episode}"
                        self.save_data(self.file_name)

                    # print("剩余任务:", np.sum(self.patient.reg_num_a_list) + np.sum(self.patient.reg_num_p_list),
                    #       update_episode)

                    # a = np.sum(self.patient.reg_num_a_list) + np.sum(self.patient.reg_num_p_list)
                    # print(np.sum(self.patient.mask_matrix))
                    if update_episode == self.args.batch_num:
                        print("mean_reward:", np.mean(np.array(sum_reward)), episode)
                        # print("总间隙:", total_idle_time)
                        # print("病人空隙：", p_idle)
                        # print("医生空隙：", d_idle)
                        # print("episode:", episode)
                        self.batch_buffer.get_data()
                        loss = 0.
                        # v_loss = 0.
                        # p_loss = 0.
                        # entropy = 0.
                        for _ in range(0, self.args.batch_num, self.args.update_size):
                            # self.env.reset()
                            batch_states, batch_edges, batch_edge_attrs, batch_actions, batch_returns, \
                                batch_values, batch_log_prob, batch_adv = \
                                self.batch_buffer.get_mini_batch(self.args.update_size)
                            loss = self.ppo.learn(batch_states, batch_edges, batch_edge_attrs, batch_actions,
                                                  batch_returns, batch_values, batch_log_prob, batch_adv)
                            # v_loss, p_loss, entropy = self.ppo.learn(batch_states, batch_edges, batch_actions,
                            #                                          batch_returns, batch_values, batch_log_prob,
                            #                                          batch_adv)
                            # print("return:", torch.sum(batch_returns))
                        print("loss:", loss)
                        self.r_l.append([np.mean(np.array(sum_reward)), loss.item(), episode])
                        # print("v_loss:", v_loss)
                        # print("p_loss:", p_loss)
                        # print("entropy:", entropy)
                        self.batch_buffer.__init__(buffer_num=self.args.batch_num,
                                                   gamma=self.args.gamma, lam=self.args.gae_lambda)
                        update_episode = 0
                        # print("episode:", episode)
                        # print("总时间：", self.env.get_total_time())
                    if episode % 120 == 0:
                        self.save_model()
                        self.save_info(self.r_l, f"r_l", ['reward', 'loss', 'ep'], "r_l")
                        self.save_info(self.idle_total, f"i_t", ['d_idle', 'p_idle', 'idle',
                                                                 'total_d', 'total_p', 'total', 'ep'], "i_t")
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
            print(f'保存结果文件')

    def save_info(self, data_list, file_name, headers, path):
        with open(f'./data/{path}/{file_name}.csv', mode='w+', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)

            csv_writer.writerow(headers)
            csv_writer.writerows(data_list)
            print(f'保存结果文件')

    def get_data_edge(self):
        data = self.patient.state.astype("float32")
        edge, edge_attr = self.env.get_edge()
        return data, edge, edge_attr
