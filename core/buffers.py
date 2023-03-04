import numpy as np


# need to speed up, such as put them on numpy directly or on Tensor
import torch


class Buffer:
    def __init__(self):
        self.state_list = []
        self.edge_list = []
        self.action_list = []
        self.reward_list = []
        self.terminal_list = []
        self.value_list = []
        self.log_prob_list = []

    def add_data(self, state_t=None, edge_t=None, action_t=None, reward_t=None,
                 terminal_t=None, value_t=None, log_prob_t=None):
        if state_t and action_t and reward_t and terminal_t \
                and value_t and log_prob_t:
            self.state_list.extend(state_t)
            self.edge_list.extend(edge_t)
            self.action_list.extend(action_t)
            self.reward_list.extend(reward_t)
            self.terminal_list.extend(terminal_t)
            self.value_list.extend(value_t)
            self.log_prob_list.extend(log_prob_t)


class BatchBuffer:
    def __init__(self, buffer_num, gamma, lam):
        self.buffer_num = buffer_num
        self.buffer_list = [Buffer() for _ in range(self.buffer_num)]
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = None
        self.edges = None
        self.actions = None
        self.returns = None
        self.values = None
        self.log_prob = None
        self.adv = None

    def add_batch_data(self, states_t=None, edge_t=None, actions_t=None,
                       rewards_t=None, terminals_t=None, values_t=None, log_prob_t=None):
        assert len(states_t) == len(actions_t) == len(rewards_t) == len(terminals_t) \
               == (len(values_t)-1) == len(log_prob_t)
        for i in range(self.buffer_num):
            self.buffer_list[i].add_data(states_t, edge_t, actions_t, rewards_t,
                                         terminals_t, values_t, log_prob_t)

    def buffer_list_to_array(self):
        states = []
        edges = []
        actions = []
        rewards = []
        terminals = []
        values = []
        log_prob = []
        min_len = 999999
        for buf in self.buffer_list:
            min_len = min(len(buf.state_list), min_len)

        for buffer in self.buffer_list:
            states.extend(buffer.state_list)
            edges.extend(buffer.edge_list)
            actions.extend(buffer.action_list)
            rewards.extend(buffer.reward_list)
            terminals.extend(buffer.terminal_list)
            values.extend(buffer.value_list)
            log_prob.extend(buffer.log_prob_list)

        values = torch.cat([value for value in values], dim=0).view(1, -1)
        log_prob = torch.cat([lg for lg in log_prob], dim=0).view(1, -1)
        rewards = torch.from_numpy(np.array(rewards)).to(self.device).detach().view(1, -1)
        terminals = torch.from_numpy(np.array(terminals, dtype=int)).to(self.device).detach().view(1, -1)

        return states, edges, actions, rewards, terminals, values, log_prob

    def compute_reward_to_go_returns(self, rewards, values, terminals):
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
        rewards = torch.transpose(rewards, 1, 0)
        values = torch.transpose(values, 1, 0)
        terminals = torch.transpose(terminals, 1, 0)
        r = values[-1]
        returns = []

        for i in reversed(range(rewards.shape[0])):
            r = rewards[i] + (1. - terminals[i]) * self.gamma * r
            returns.append(r.view(-1, 1))
        returns = torch.cat(list(reversed(returns)), dim=1)
        # (T,N) -> (N,T)
        return returns

    def compute_GAE(self, rewards, values, terminals):
        # (N,T) -> (T,N)
        # rewards = np.transpose(rewards, [1, 0])
        # values = np.transpose(values, [1, 0])
        terminals = torch.transpose(terminals, 1, 0)
        rewards = torch.transpose(rewards, 1, 0)
        values = torch.transpose(values, 1, 0)
        length = rewards.shape[0]
        # print('reward:{},value:{},terminal{}'.format(rewards.shape,values.shape,terminals.shape))
        deltas = []
        for i in reversed(range(length)):
            v = rewards[i] + (1. - terminals[i]) * self.gamma * values[i + 1]
            delta = v - values[i]
            deltas.append(delta.view(1, -1))
        deltas = torch.cat(list(reversed(deltas)), dim=0)

        advantage = deltas[-1, :]
        advantages = [advantage.view(1, -1)]
        for i in reversed(range(length - 1)):
            advantage = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * advantage
            advantages.append(advantage.view(1, -1))
        advantages = torch.cat(list(reversed(advantages)), dim=0).view(-1, length)
        # (T,N) -> (N,T)
        # advantages = np.transpose(list(advantages), [1, 0])
        # print(advantages)
        return advantages

    def get_data(self):
        states, edges, actions, rewards, terminals, values, log_prob = self.buffer_list_to_array()
        # rew_mean = torch.cat(
        #     [torch.full((1, rewards.shape[1]), m.item()) for m in torch.mean(rewards, dim=1)]).to(self.device)
        # rew_std = torch.cat(
        #     [torch.full((1, rewards.shape[1]), m.item() + 1e-8) for m in torch.std(rewards, dim=1)]).to(self.device)
        # rewards = ((rewards - rew_mean) / rew_std)
        adv = self.compute_GAE(rewards, values, terminals)

        mean = torch.cat([torch.full((1, adv.shape[1]), m.item()) for m in torch.mean(adv, dim=1)]).to(self.device)
        std = torch.cat([torch.full((1, adv.shape[1]), m.item() + 1e-8) for m in torch.std(adv, dim=1)]).to(self.device)

        adv = ((adv - mean) / (std + 1e-8))

        returns = self.compute_reward_to_go_returns(rewards, values, terminals)

        self.states, self.edges, self.actions, self.returns, self.values, \
            self.log_prob, self.adv = states, edges, actions, returns, values, log_prob, adv

    def get_mini_batch(self, mini_size):
        select_index = np.random.choice(a=len(self.states), size=mini_size, replace=False, p=None)
        batch_states = []
        batch_edges = []
        batch_actions = []
        batch_returns = self.returns.view(-1,)[select_index].view(1, -1)
        batch_values = self.values.view(-1,)[select_index].view(1, -1)
        batch_log_prob = self.log_prob.view(-1,)[select_index].view(1, -1)
        batch_adv = self.adv.view(-1,)[select_index].view(1, -1)

        for index in select_index:
            batch_states.append(self.states[index])
            batch_edges.append(self.edges[index])
            batch_actions.append(self.actions[index])

        return batch_states, batch_edges, batch_actions, batch_returns, \
            batch_values, batch_log_prob, batch_adv
