# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:09
# @Author : hxc
# @File : run.py
# @Software : PyCharm
from core.params import Params
from core.trainer import Trainer
from torch.optim.lr_scheduler import StepLR


def adjust_params():
    params = Params()
    n = 0
    for algorithm in params.policy_list:
        params.args.policy = algorithm
        # for lr in params.lr_list:
        #     params.args.lr = lr
        #     for dr in params.discount_rate:
        #         print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} start training!")
        #         params.args.gamma = dr
        #         params.args.file_name = n
        #         trainer = Trainer(params.args)
        #         trainer.train()
        #         n += 1
        #         trainer.save_model(trainer.model_name)
        #         print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} training finished!")

        for lr in params.lr_decay_list:
            params.args.lr = lr
            for dr in params.discount_rate:
                print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} start training!")
                params.args.gamma = dr
                params.args.file_name = n
                trainer = Trainer(params.args)
                trainer.scheduler = StepLR(trainer.algorithm.optimizer, step_size=500, gamma=0.9)
                trainer.train()
                n += 1
                trainer.save_model(trainer.model_name)
                print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} training finished!")


if __name__ == '__main__':
    adjust_params()
    # trainer = Trainer()
    # trainer.train()
    # trainer.save_model(trainer.model_name)
    # # trainer.save_reward_loss("r_l")
    # # trainer.save_data("result")
    # print("training finished!")
