# -*- coding : utf-8 -*-
# @Time :  2022/7/29 11:00
# @Author : hxc
# @File : utils.py
# @Software : PyCharm
import torch
import numpy as np
import time
import datetime


def get_now_date():
    now_date = datetime.datetime.now()
    # year = str(now_date.year)
    # month = str(now_date.month)
    # day = str(now_date.day)
    # s = str(now_date.second)
    moment = time.strftime("%H-%M")
    # return year + '-' + month + '-' + day + '-' + moment + '-' + s
    return moment


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
