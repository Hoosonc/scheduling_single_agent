# -*- coding : utf-8 -*-
# @Time :  2022/7/27 14:09
# @Author : hxc
# @File : run.py
# @Software : PyCharm
from core.params import Params
from core.trainer import Trainer

if __name__ == '__main__':
    args = Params().args
    print("The multi_agent RL program of Patient treatment sequence scheduling starts training...")
    trainer = Trainer(args)
    trainer.train()
    trainer.save_model(trainer.model_name)
    # trainer.save_reward_loss("r_l")
    # trainer.save_data("result")
    print("training end!")
