import numpy as np


# need to speed up, such as put them on numpy directly or on Tensor
import torch


class Buffer:
    def __init__(self):
        self.state_list = []
        self.edge_list = []
        self.edge_attr_list = []
        self.action_list = []
        self.reward_list = []
        self.terminal_list = []
        self.value_list = []
        self.log_prob_list = []
        self.adv = None
        self.returns = None

    def add_data(self, state_t=None, edge_t=None, action_t=None, reward_t=None,
                 terminal_t=None, value_t=None, log_prob_t=None, edge_attr_t=None):
        if state_t and action_t and reward_t and terminal_t \
                and value_t and log_prob_t:
            self.state_list.extend(state_t)
            self.edge_list.extend(edge_t)
            self.action_list.extend(action_t)
            self.reward_list.extend(reward_t)
            self.terminal_list.extend(terminal_t)
            self.value_list.extend(value_t)
            self.log_prob_list.extend(log_prob_t)
            self.edge_attr_list.extend(edge_attr_t)


class BatchBuffer:
    def __init__(self, buffer_num, gamma, lam):
        self.buffer_num = buffer_num
        self.buffer_list = [Buffer() for _ in range(self.buffer_num)]
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = None
        self.edges = None
        self.edges_attr = None
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
        for buf in self.buffer_list:
            rewards = torch.from_numpy(np.array(buf.reward_list)).to(self.device).detach().view(1, -1)
            values = torch.cat([value for value in buf.value_list], dim=0).view(1, -1)
            terminals = torch.from_numpy(np.array(buf.terminal_list, dtype=int)).to(self.device).detach().view(1, -1)
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
            buf.returns = torch.cat(list(reversed(returns)), dim=1)

            deltas = torch.cat(list(reversed(deltas)), dim=0)
            advantage = deltas[-1, :]
            advantages = [advantage.view(1, -1)]
            for i in reversed(range(rewards.shape[0] - 1)):
                advantage = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * advantage
                advantages.append(advantage.view(1, -1))
            advantages = torch.cat(list(reversed(advantages)), dim=0).view(-1, rewards.shape[0])
            mean = torch.cat(
                [torch.full((1, advantages.shape[1]), m.item()) for m in torch.mean(advantages, dim=1)]).to(self.device)
            std = torch.cat(
                [torch.full((1, advantages.shape[1]), m.item() + 1e-8) for m in torch.std(advantages, dim=1)]).to(
                self.device)
            buf.adv = ((advantages - mean) / (std + 1e-8))
            # (T,N) -> (N,T)

    # def compute_GAE(self, rewards, values, terminals):
    #     # (N,T) -> (T,N)
    #     # rewards = np.transpose(rewards, [1, 0])
    #     # values = np.transpose(values, [1, 0])
    #     for buf in self.buffer_list:
    #         rewards = torch.from_numpy(np.array(buf.reward_list)).to(self.device).detach().view(1, -1)
    #         values = torch.cat([value for value in buf.value_list], dim=0).view(1, -1)
    #         terminals = torch.from_numpy(np.array(buf.terminal_list, dtype=int)).to(self.device).detach().view(1, -1)
    #         rewards = torch.transpose(rewards, 1, 0)
    #         values = torch.transpose(values, 1, 0)
    #         terminals = torch.transpose(terminals, 1, 0)
    #         length = rewards.shape[0]
    #         # print('reward:{},value:{},terminal{}'.format(rewards.shape,values.shape,terminals.shape))
    #         deltas = []
    #         for i in reversed(range(length)):
    #             v = rewards[i] + (1. - terminals[i]) * self.gamma * values[i + 1]
    #             delta = v - values[i]
    #             deltas.append(delta.view(1, -1))
    #         deltas = torch.cat(list(reversed(deltas)), dim=0)
    #
    #         advantage = deltas[-1, :]
    #         advantages = [advantage.view(1, -1)]
    #         for i in reversed(range(length - 1)):
    #             advantage = deltas[i] + (1. - terminals[i]) * self.gamma * self.lam * advantage
    #             advantages.append(advantage.view(1, -1))
    #         advantages = torch.cat(list(reversed(advantages)), dim=0).view(-1, length)
    #         # (T,N) -> (N,T)
    #         # advantages = np.transpose(list(advantages), [1, 0])
    #         # print(advantages)
    #     return advantages

    def get_data(self):
        # states, edges, actions, rewards, terminals, values, log_prob = self.buffer_list_to_array()

        self.compute_reward_to_go_returns_adv()

        # self.states, self.edges, self.actions, self.returns, self.values, \
        #     self.log_prob, self.adv = states, edges, actions, returns, values, log_prob, adv

    def get_mini_batch(self, start_index, mini_size):
        # select_index = np.random.choice(a=len(self.states), size=mini_size, replace=False, p=None)
        buf_list = self.buffer_list[start_index:start_index+mini_size]
        # batch_states = []
        # batch_edges = []
        # batch_edges_attr = []
        # batch_actions = []
        # batch_returns = []
        # batch_values = []
        # batch_log_prob = []
        # batch_adv = []

        # batch_returns = self.returns.view(-1,)[select_index].view(1, -1)
        # batch_values = self.values.view(-1,)[select_index].view(1, -1)
        # batch_log_prob = self.log_prob.view(-1,)[select_index].view(1, -1)
        # batch_adv = self.adv.view(-1,)[select_index].view(1, -1)

        # for buf in buf_list:
        #     batch_states.append(buf.state_list)
        #     batch_edges.append(buf.edge_list)
        #     batch_edges_attr.append(buf.edge_attr_list)
        #     batch_actions.append(buf.action_list)
        #     batch_returns.append(buf.returns)
        #     batch_adv.append(buf.adv)
        #     batch_log_prob.append(buf.log_prob_list)

        return buf_list
