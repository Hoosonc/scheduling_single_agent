# -*- coding : utf-8 -*-
# @Time :  2022/8/18 15:55
# @Author : hxc
# @File : gen_data.py
# @Software : PyCharm
import math
import random

import numpy as np
import csv
import pandas as pd
# from net.utils import get_now_date as hxc


def distance_d(doc_num, file_name):
    dis_arr = np.random.choice(a=[1, 2, 3, 4, 5], size=doc_num*doc_num, replace=True).reshape(doc_num, doc_num)
    dis_arr = np.triu(dis_arr)
    for i in range(dis_arr.shape[0]):
        dis_arr[i][i] = 0
    dis_arr += dis_arr.T - np.diag(dis_arr.diagonal())
    dis_arr = pd.DataFrame(data=dis_arr.tolist(), index=None)
    dis_arr.to_csv(f"../data/{file_name}.csv", index=False, header=None)


def gen_data(num_patients, num_doctors):
    multi_reg_patient_num = int(num_patients*0.1)
    multi_3 = np.random.choice(a=num_patients, size=multi_reg_patient_num, replace=False)
    multi_2 = np.random.choice(a=multi_3, size=2, replace=False)
    mask = np.isin(multi_3, multi_2, invert=True)

    # 应用布尔掩码来删除数组b中的元素
    multi_3 = multi_3[mask]
    all_reg_list = []
    doc_reg_num = np.zeros((num_doctors,))
    did_list = [i for i in range(num_doctors)]
    for i in range(num_patients):
        if i in multi_3:
            size = 3
        elif i in multi_2:
            size = 2
        else:
            size = 1
        d_idx = np.random.choice(a=did_list, size=size, replace=False)
        for d in d_idx:
            all_reg_list.append([i, d, random.randint(5, 10)])
            doc_reg_num[d] += 1
            if doc_reg_num[d] == 37:
                did_list.remove(d)
    df = pd.DataFrame(data=all_reg_list, columns=["pid", "did", "pro_time"])
    # df.to_csv(f"../data/{num_doctors}_{num_patients}_{len(all_reg_list)}.csv", index=False)
    return df


if __name__ == '__main__':
    # distance_d(10, "distance")
    gen_data(300, 10)
    # doctors = gen_doctors()
    # doc_header = ['did', 'reg_num', 'start_time', 'avg_pro_time']
    # save_data(doc_header, doctors, "doc_am")
    # data_list = gen_patient()
    # print(len(data_list))
    # p_header = ['pid', 'did', 'start_time', 'pro_time']
    # save_data(p_header, data_list, "reg_am1")
    # df = pd.read_csv("../data/reg_new.csv")
    # a = df.groupby("pid").count()
    # print(a)
