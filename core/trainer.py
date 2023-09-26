# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:13
# @Author : hxc
# @File : trainer.py
# @Software : PyCharm
import time

from core.env import Environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from tools.csv_tools import csv_writer
from net.AC_model import AC
from net.DQN_model import DQN
from threading import Thread
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from core.buffers import BatchBuffer
from core.rl_algorithms import PPOClip
from core.DQN import DQN_update
from core.Actor_critic import AC_update
from core.ddpg import DDPG_update
from net.AC_GCN import AC_GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args):
        self.args = args
        self.csv_writer = csv_writer()
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
        elif self.policy == "ddpg":
            self.model = AC().to(device)
            self.algorithm = DDPG_update(self.model, device, self.args)
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
        self.rewards_list = []
        self.terminal_list = []
        self.loss = []

        self.idle_total = []
        self.p_total_idle = []
        self.d_total_idle = []
        self.episode = 0
        self.sum_reward = []
        self.returns = []
        self.model_name = f"{self.args.file_name}"
        self.w_header = True
        # self.load_params(self.model_name)
        self.action_dim = args.action_dim

        self.buffer = BatchBuffer(self.args.env_num, self.args.gamma, self.args.gae_lambda, self.policy)

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

            self.get_results(episode)
            # update net
            self.buffer.get_data()
            loss = 0
            if self.policy == "ppo2":
                mini_buffer = self.buffer.get_mini_batch(self.args.update_num)

                for i in range(0, self.args.update_num):
                    # self.env.reset()
                    buf = mini_buffer[i]
                    loss = self.algorithm.learn(buf)
            else:
                loss = self.algorithm.learn(self.buffer)

            self.buffer.reset()

            self.loss.append([loss.item(), episode])

            if self.scheduler is not None:
                self.scheduler.step()

            # if episode % 1 == 0:
            #     print("loss:", loss.item())
                # print("sum_reward:", self.rewards_list[episode][0])
                # print("returns:", self.returns[episode][0], episode)
                # print("total_idle", self.p_total_idle[episode][0]+self.d_total_idle[episode][0])
                # print("p_idle:", self.p_total_idle[episode][0])
                # print("d_idle:", self.d_total_idle[episode][0])
            if (episode + 1) % 60 == 0:
                self.save_data(episode)
                self.save_model(self.model_name)

    def step(self, env, i):
        buffer = self.buffer.buffer_list[i]
        done = False
        for step in range(300):
            data = env.state[:, [0, 1, 2, 4, 5, 6]].copy()
            data[:, [2, 4]] = data[:, [2, 4]] / (env.jobs_length.max())
            m_edge_index = coo_matrix(env.m_edge_matrix)
            m_edge_index = np.array([m_edge_index.row, m_edge_index.col])
            np.fill_diagonal(env.j_edge_matrix, 0)
            j_edge_index = coo_matrix(env.j_edge_matrix)
            j_edge_index = np.array([j_edge_index.row, j_edge_index.col])
            edge_index = np.concatenate([m_edge_index, j_edge_index], axis=1)
            data = torch.tensor(data, dtype=torch.float32).to(device)
            edge_index = torch.tensor(edge_index.astype("int64")).to(device)

            data = Data(x=data, edge_index=edge_index, num_nodes=len(data))
            if self.policy == "dqn":
                value, log_prob = 0, 0
                action, q, _ = self.choose_action(data, env)
            else:
                q = 0
                action, value, log_prob = self.choose_action(data, env)
            temp_s = env.state[env.state[:, 4] == 1]
            # job_id = temp_s[temp_s[:, 7] == 0][action, 3]
            job_id = temp_s[action, 3]

            job_id = job_id.astype("int").reshape(job_id.shape[1],)
            done, reward = env.step(job_id, step)
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
        # if self.policy != "dqn":
        buffer.compute_reward_to_go_returns_adv()
        env.sum_reward = np.sum(buffer.reward_list)
        if self.policy == "dqn" or self.policy == "ddpg":
            env.returns = buffer.q_returns[0][0].item()
        else:
            env.returns = buffer.returns[0][0].item()

    def choose_action(self, data, env):
        if self.policy == "dqn":
            logits, prob = self.model(data)
            # policy_head = Categorical(probs=prob.view(1, -1))
            if env.state[:, 4].sum() >= self.action_dim:
                action_dim = self.action_dim
            else:
                action_dim = int(env.state[:, 4].sum())
            # 找到前两个最大值及其索引
            # top_values, top_indices = torch.topk(prob, k=action_dim)
            # actions = policy_head.sample((action_dim,))
            top_values, top_indices = torch.topk(prob, k=action_dim)
            actions = top_indices
            q = torch.sum(logits[0, actions.view(1, -1)])
            # ac = np.append(actions.cpu().numpy().reshape(1, -1), multi_task)
            # ac = np.random.shuffle(ac)
            return actions.cpu().numpy().reshape(1, -1), q, 0
        else:
            prob, value, log_probs = self.model(data)

            policy_head = Categorical(probs=prob.view(1, -1))

            if env.state[:, 4].sum() >= self.action_dim:
                action_dim = self.action_dim
            else:
                action_dim = int(env.state[:, 4].sum())
            # actions = policy_head.sample((action_dim, ))
            # 使用torch.topk函数获取前N个最大值及其索引
            top_values, top_indices = torch.topk(prob, k=action_dim)
            actions = top_indices
            log_prob = torch.mean(policy_head.log_prob(actions))
            # ac = np.append(actions.cpu().numpy().reshape(1, -1), multi_task)
            # ac = np.random.shuffle(ac)
            return actions.cpu().numpy().reshape(1, -1), value, log_prob

    def get_results(self, episode):
        p_idle_list = []

        d_idle_list = []
        total_idle_time_list = []
        rewards = []
        returns = []
        self.episode = episode

        for env in self.envs:
            p_idle = int(np.sum(env.p_total_idle_time))
            d_idle = int(np.sum(env.d_total_idle_time))

            rewards.append(env.sum_reward)
            returns.append(env.returns)

            total_idle_time = int(p_idle + d_idle)
            # total_time = env.d_total_time + env.p_total_time
            d_idle_list.append(d_idle)
            p_idle_list.append(p_idle)
            total_idle_time_list.append(total_idle_time)
            env.reset()
        self.p_total_idle.append(self.get_result(p_idle_list))
        self.d_total_idle.append(self.get_result(d_idle_list))
        self.rewards_list.append(self.get_result(rewards))
        self.returns.append(self.get_result(returns))

    def get_result(self, result_list):
        sum_ = sum(result_list)
        mean_ = np.mean(result_list)
        result_list.extend([sum_, mean_, self.episode])
        return result_list

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), f'./net/params/{file_name}.pth')

    def load_params(self, model_name):
        self.model.load_state_dict(torch.load(f"./net/params/{model_name}.pth"))

    def save_data(self, episode):
        if self.w_header:
            self.csv_writer.write_headers(f"loss_{self.model_name}", ['loss', 'ep'], "./data/loss")
            rewards_header = [f"rewards_{i}" for i in range(len(self.envs))]
            rewards_header.extend(["sum_rewards", "mean_rewards", "ep"])
            self.csv_writer.write_headers(f"rewards_{self.model_name}", rewards_header, "./data/rewards")
            returns_header = [f"returns_{i}" for i in range(len(self.envs))]
            returns_header.extend(["sum_returns", "mean_returns", "ep"])
            self.csv_writer.write_headers(f"returns_{self.model_name}", returns_header, "./data/returns")

            p_idle_header = [f"p_idle_{i}" for i in range(len(self.envs))]
            p_idle_header.extend(["sum_p_idle", "mean_p_idle", "ep"])
            self.csv_writer.write_headers(f"p_idle_{self.model_name}", p_idle_header, "./data/p_idle")

            d_idle_header = [f"d_idle_{i}" for i in range(len(self.envs))]
            d_idle_header.extend(["sum_d_idle", "mean_d_idle", "ep"])
            self.csv_writer.write_headers(f"d_idle_{self.model_name}", d_idle_header, "./data/d_idle")

            self.w_header = False
        self.episode = episode
        self.save_model(self.model_name)

        self.csv_writer.write_result(self.loss, f"loss_{self.model_name}", "./data/loss")
        self.csv_writer.write_result(self.rewards_list, f"rewards_{self.model_name}", "./data/rewards")
        self.csv_writer.write_result(self.returns, f"returns_{self.model_name}", "./data/returns")
        self.csv_writer.write_result(self.p_total_idle, f"p_idle_{self.model_name}", "./data/p_idle")
        self.csv_writer.write_result(self.d_total_idle, f"d_idle_{self.model_name}", "./data/d_idle")
        self.loss = []
        self.rewards_list = []
        self.returns = []
        self.p_total_idle = []
        self.d_total_idle = []
