import numpy as np
from threading import Thread
import torch.nn.functional as f
# need to speed up, such as put them on numpy directly or on Tensor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer:
    def __init__(self, gamma, lam):
        self.state_list = []
        self.edge_list = []
        self.edge_attr_list = []
        self.action_list = []
        self.reward_list = []
        self.terminal_list = []
        self.value_list = []
        self.log_prob_list = []
        self.candidate_list = []
        self.adv = None
        self.returns = None
        self.gamma = gamma
        self.lam = lam
        self.log_prob = None

    def add_data(self, state_t=None, action_t=None, reward_t=None,
                 terminal_t=None, value_t=None, log_prob_t=None):

        self.state_list.append(state_t)
        # self.candidate_list.append(candidate)
        # self.edge_list.append(edge_t)
        # self.edge_attr_list.append(edge_attr_t)
        self.action_list.append(action_t)
        self.reward_list.append(reward_t)
        self.terminal_list.append(terminal_t)
        self.value_list.append(value_t.detach())
        self.log_prob_list.append(log_prob_t.detach())

    def compute_reward_to_go_returns_adv(self):
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

        rewards = torch.from_numpy(np.array(self.reward_list)).to(device).detach().view(1, -1)
        values = torch.cat([value for value in self.value_list], dim=0).view(1, -1)
        log_prob = torch.cat([log_p for log_p in self.log_prob_list], dim=0).view(1, -1)
        self.log_prob = log_prob
        terminals = torch.from_numpy(np.array(self.terminal_list, dtype=int)).to(device).detach().view(1, -1)
        rewards = torch.transpose(rewards, 1, 0)
        values = torch.transpose(values, 1, 0)
        terminals = torch.transpose(terminals, 1, 0)
        r = values[-1]
        returns = []
        deltas = []
        for i in reversed(range(rewards.shape[0])):
            r = rewards[i] + (1. - terminals[i]) * self.gamma * r
            returns.append(r.view(-1, 1))

            v = rewards[i] + (1. - terminals[i]) * self.gamma * values[i + 1]
            delta = v - values[i]
            deltas.append(delta.view(1, -1))
        self.returns = torch.cat(list(reversed(returns)), dim=1)

        deltas = torch.cat(list(reversed(deltas)), dim=0)
        advantage = deltas[-1, :]
        advantages = [advantage.view(1, -1)]
        for i in reversed(range(rewards.shape[0] - 1)):
            advantage = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * advantage
            advantages.append(advantage.view(1, -1))
        advantages = torch.cat(list(reversed(advantages)), dim=0).view(-1, rewards.shape[0])
        self.adv = f.normalize(advantages, p=2, dim=1)
        del self.value_list[-1]


class BatchBuffer:
    def __init__(self, buffer_num, gamma, lam):
        self.buffer_num = buffer_num
        self.buffer_list = [Buffer(gamma, lam) for _ in range(self.buffer_num)]
        # self.mini_buffer = None
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = None
        self.edges = None
        # self.candidate = None
        self.actions = None
        self.returns = None
        self.rewards = None
        self.values = None
        self.log_prob = None
        self.adv = None

    def reset(self):
        self.buffer_list = [Buffer(self.gamma, self.lam) for _ in range(self.buffer_num)]
        # self.mini_buffer = None
        self.states = None
        self.edges = None
        # self.candidate = None
        self.actions = None
        self.returns = None
        self.rewards = None
        self.values = None
        self.log_prob = None
        self.adv = None

    def add_batch_data(self, states_t=None, actions_t=None,
                       rewards_t=None, terminals_t=None, values_t=None, log_prob_t=None):
        assert len(states_t) == len(actions_t) == len(rewards_t) == len(terminals_t) \
               == (len(values_t) - 1) == len(log_prob_t)
        for i in range(self.buffer_num):
            self.buffer_list[i].add_data(states_t, actions_t, rewards_t,
                                         terminals_t, values_t, log_prob_t)

    def get_data(self):
        state_list = []
        edge_list = []
        # candidate_list = []
        action_list = []
        reward_list = []
        # terminal_list = []
        value_list = []
        log_prob_list = []
        adv_list = []
        return_list = []
        for buf in self.buffer_list:
            state_list.extend(buf.state_list)
            edge_list.extend(buf.edge_list)
            # candidate_list.extend(buf.candidate_list)
            action_list.extend(buf.action_list)
            value_list.extend(buf.value_list)
            log_prob_list.extend(buf.log_prob_list)
            reward_list.extend(buf.reward_list)
            # terminal_list.extend(buf.terminal_list)
            return_list.append(buf.returns)
            adv_list.append(buf.adv)

        self.states = state_list
        # self.edges = np.array(edge_list)
        self.edges = edge_list
        # self.edges_attr = np.array(edge_attr_list)
        self.actions = np.array(action_list)
        # self.candidate = candidate_list
        self.log_prob = torch.cat([log_prob.detach() for log_prob in log_prob_list], dim=0).view(1, -1)
        self.values = torch.cat([value.detach() for value in value_list], dim=0).view(1, -1)
        # self.value_list = np.array(self.value_list)
        # self.log_prob_list = np.array(self.log_prob_list)
        self.rewards = np.array(reward_list)
        self.returns = torch.cat(return_list, dim=1)
        self.adv = torch.cat(adv_list, dim=1)

    def get_mini_batch(self, mini_size, update_num):
        mini_buffer = [Buffer(self.gamma, self.lam) for _ in range(update_num)]
        t_list = []
        for i in range(update_num):
            t = Thread(target=self.get_batch, args=(mini_buffer[i], mini_size, i))
            t.start()
            t_list.append(t)
        for thread in t_list:
            thread.join()
        return mini_buffer

    def get_batch(self, buf, mini_size, idx):
        assert (idx*mini_size) < len(self.states)
        if ((idx+1)*mini_size) >= len(self.states):
            select_index = np.arange(start=(idx*mini_size), stop=len(self.states))
        else:
            select_index = np.arange(start=(idx*mini_size), stop=(idx+1)*mini_size)
        # select_index = np.random.choice(a=len(self.states), size=mini_size, replace=False, p=None)
        # buf.state_list = self.states[select_index]
        # buf.edge_list = self.edges[select_index]
        for index in select_index:
            buf.state_list.append(self.states[index])
            # buf.candidate_list.append(self.candidate[index])
        # buf.edge_attr_list = self.edges_attr[select_index]
        buf.action_list = self.actions[select_index]
        buf.value_list = torch.index_select(self.values.view(1, -1), dim=1,
                                            index=torch.tensor(select_index).to(device))
        buf.log_prob_list = torch.index_select(self.log_prob.view(1, -1), dim=1,
                                               index=torch.tensor(select_index).to(device))
        buf.reward_list = self.rewards[select_index]
        buf.returns = torch.index_select(self.returns, dim=1, index=torch.tensor(select_index).to(device))
        buf.adv = torch.index_select(self.adv, dim=1, index=torch.tensor(select_index).to(device))
