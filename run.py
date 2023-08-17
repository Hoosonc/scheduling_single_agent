# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:09
# @Author : hxc
# @File : run.py
# @Software : PyCharm
from core.params import Params
from core.trainer import Trainer


def adjust_params():
    params = Params()
    for algorithm in params.policy_list:
        params.args.policy = algorithm
        for lr in params.lr_list:
            params.args.lr = lr
            for dr in params.discount_rate:
                print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} start training!")
                params.args.gamma = dr
                trainer = Trainer(params.args)
                trainer.train()
                trainer.save_model(trainer.model_name)
                print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} training finished!")


if __name__ == '__main__':
    print("The RL program starts training...")
    adjust_params()
    # trainer = Trainer()
    # trainer.train()
    # trainer.save_model(trainer.model_name)
    # # trainer.save_reward_loss("r_l")
    # # trainer.save_data("result")
    # print("training finished!")
