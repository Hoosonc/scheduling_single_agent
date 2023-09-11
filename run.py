# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:09
# @Author : hxc
# @File : run.py
# @Software : PyCharm
import pandas as pd

from core.params import Params
from core.trainer import Trainer
from torch.optim.lr_scheduler import StepLR


def adjust_params():
    params = Params()
    n = 0
    file_list = []
    for file_id in range(5):
        params.args.file_id = file_id
        for algorithm in params.policy_list:
            params.args.policy = algorithm
            for lr in params.lr_decay_list:
                params.args.lr = lr
                for dr in params.discount_rate:
                    print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} start training!")
                    params.args.gamma = dr
                    params.args.file_name = n
                    trainer = Trainer(params.args)
                    trainer.scheduler = StepLR(trainer.algorithm.optimizer, step_size=232, gamma=0.9)
                    trainer.train()
                    n += 1
                    trainer.save_model(trainer.model_name)
                    file_list.append([n, file_id, algorithm, lr, dr])
                    print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} training finished!")
            for lr in params.lr_list:
                params.args.lr = lr
                for dr in params.discount_rate:
                    print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} start training!")
                    params.args.gamma = dr
                    params.args.file_name = n
                    trainer = Trainer(params.args)

                    trainer.train()
                    n += 1
                    trainer.save_model(trainer.model_name)
                    file_list.append([n, file_id, algorithm, lr, dr])
                    print(f"--algorithm:{algorithm} --lr:{lr} --discount rate:{dr} training finished!")

    df = pd.DataFrame(data=file_list, columns=["file", "file_id", "algorithm", "lr", "dr"])
    df.to_csv("./data/adjust_1.csv", index=False)


def adjust_sc():
    params = Params()
    params.args.policy = "ppo2"
    params.gamma = 0.99
    params.args.lr = 0.00001
    n = 0
    file_list = []
    for file_id in range(1):
        params.args.file_id = file_id
        # params.args.lr = 0.00001
        for single_num in params.single_sample:
            params.args.action_dim = single_num
            params.args.file_name = n
            trainer = Trainer(params.args)
            print(f"--single_num:{single_num} start training!")
            # trainer.scheduler = StepLR(trainer.algorithm.optimizer, step_size=232, gamma=0.9)
            trainer.train()
            n += 1
            trainer.save_model(trainer.model_name)

            file_list.append([n, file_id, single_num])

    df = pd.DataFrame(data=file_list, columns=["file", "file_id", "single_num"])
    df.to_csv("./data/adjust_sc_1.csv", index=False)


if __name__ == '__main__':
    # adjust_params()
    adjust_sc()
    # trainer = Trainer()
    # trainer.train()
    # trainer.save_model(trainer.model_name)
    # # trainer.save_reward_loss("r_l")
    # # trainer.save_data("result")
    # print("training finished!")
