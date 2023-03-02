import torch
from torch.optim.lr_scheduler import StepLR


class PPOClip:
    def __init__(self, net, device, args):
        self.net = net
        self.device = device
        # parameters
        self.value_factor = args.value_loss_coefficient  # paper value
        self.entropy_factor = args.entropy_coefficient
        self.clip_epsilon = args.epsilon
        self.lr_v = args.lr_v
        self.max_grad_norm = args.max_grad_norm
        self.v_loss_no_clip = None
        self.v_loss = None
        self.pi_loss = None
        self.entropy = None

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr_v)
        self.scheduler = StepLR(self.optimizer, step_size=300, gamma=0.9)

    def learn(self, state_batch, batch_edges, batch_edge_attrs, action_batch, return_batch,
              old_value_batch, old_log_prob_batch, adv_batch):
        # lr = self.learning_rate

        value_batch, log_prob_batch, entropy = self.net.get_batch_p_v(state_batch, batch_edges,
                                                                      batch_edge_attrs, action_batch)

        # todo: not mentioned in paper, but used in openai baselines
        # self.value_loss_clip(value_batch, return_batch, old_value_batch)
        self.value_loss(value_batch, return_batch)

        self.pi_loss = self.policy_loss(log_prob_batch, old_log_prob_batch, adv_batch)
        # print('entropy :',policy_head.entropy())
        self.entropy = torch.mean(entropy)

        loss = self.v_loss_no_clip * self.value_factor - self.pi_loss + self.entropy * self.entropy_factor
        # loss = self.pi_loss - self.v_loss * self.value_factor + self.entropy * self.entropy_factor

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        return loss
        # return self.v_loss_no_clip, self.pi_loss, self.entropy

    def value_loss(self, value_batch, return_batch):
        self.v_loss_no_clip = 0.5 * torch.mean((value_batch - return_batch) ** 2)
        # value_loss = 0.5 * torch.mean((return_batch - value_batch) ** 2)
        # self.v_loss_no_clip = value_loss
        # others = {'r_square':r_square}
        # others = None
        # return value_loss

    def value_loss_clip(self, value_batch, return_batch, old_value_batch):  # value clip code level skill 1
        old_value_batch = old_value_batch[:, :-1].detach()
        value_clipped = old_value_batch + torch.clamp(value_batch - old_value_batch, -self.clip_epsilon,
                                                      self.clip_epsilon)
        value_loss_1 = (value_batch - return_batch) ** 2
        value_loss_2 = (return_batch - value_clipped) ** 2
        # print('value loss: ',value_loss_1.shape)
        # print('max: ',torch.max(value_loss_1, value_loss_2).shape)
        self.v_loss = .5 * torch.mean(torch.max(value_loss_1, value_loss_2))

        # others = {'r_square':r_square}
        # others = None
        # return value_loss

    def policy_loss(self, log_prob_batch, old_log_prob_batch, adv_batch):
        adv_batch = adv_batch.detach()
        ratio = torch.exp(log_prob_batch - old_log_prob_batch.detach())
        # p_ratio_average = p_ratio.cpu().detach().numpy().mean()

        # p_ratio_average = d_ratio.cpu().detach().numpy().mean()

        ratio = ratio.view(-1, 1)  # take care the dimension here!!!

        adv_batch = adv_batch.view(-1, 1)
        surrogate_1 = ratio * adv_batch
        surrogate_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_batch
        surrogate = torch.min(surrogate_1, surrogate_2)
        policy_loss = torch.mean(surrogate)

        # approxkl = .5 * torch.mean((old_log_prob_batch - log_prob_batch) ** 2)
        # # print('ratio : ', torch.gt(torch.abs(ratio-1.),self.clip_epsilon*self.alpha).float())
        # clipfrac = torch.mean(torch.gt(torch.abs(ratio - 1.), self.clip_epsilon * self.alpha).float())
        # # print('clipfrac :',clipfrac)
        #
        # args.clip_fraction = args.clip_fraction.append(
        #     pd.DataFrame({'run': [args.run], 'update_time': [args.update_time], 'clip_fraction': clipfrac.item()}),
        #     ignore_index=True)
        #
        # args.tempC.append(clipfrac.item())
        # # print(args.clip_fraction)
        # args.update_time += 1
        # others = {'approxkl': approxkl, 'clipfrac': clipfrac}
        return policy_loss
