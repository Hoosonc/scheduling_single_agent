# -*- coding : utf-8 -*-
# @Time :  2022/8/18 15:55
# @Author : hxc
# @File : gen_data.py
# @Software : PyCharm
import random

import numpy as np
import csv
import pandas as pd
# from net.utils import get_now_date as hxc


def save_data(header, data, file_name):
    with open(f'../data/{file_name}.csv', mode='a+', encoding='utf-8-sig', newline='') as f:
        csv_writer = csv.writer(f)
        header = header
        csv_writer.writerow(header)
        csv_writer.writerows(data)
        # print(f'保存文件')


def gen_doctors():
    doc_list = []
    for i in range(5):
        am = 0
        pm = 0
        did = i
        reg_num_list = [30, 35]
        op_time1 = [7, 8, 9, 10, 11]
        # op_time2 = [4, 5, 6, 7]
        reg_num = np.random.choice(a=reg_num_list, size=1, replace=False, p=None)[0]
        avg_pro_time = np.random.choice(a=op_time1, size=1, replace=False, p=None)[0]
        # if reg_num == 8:
        #     avg_pro_time = np.random.choice(a=op_time1, size=1, replace=False, p=None)[0]
        # else:
        #     avg_pro_time = np.random.choice(a=op_time2, size=1, replace=False, p=None)[0]
        # if am > 6:
        #     start_time = 1
        # elif pm > 6:
        #     start_time = 0
        # else:
        #     start_time = np.random.choice(a=[0, 1], size=1, replace=False, p=None)[0]
        #     if start_time == 0:
        #         am += 1
        #     else:
        #         pm += 1

        doc = [did, reg_num, 0, avg_pro_time]
        doc_list.append(doc)
    return doc_list


def gen_patient():
    patient_list = []
    pid_list = [pid for pid in range(0, 20)]
    p_reg_num = np.zeros((20,))
    d_reg_num = np.zeros((5,))
    d_p = [[] for did in range(5)]
    doc_file = pd.read_csv("../data/doc_am1.csv", encoding="utf-8-sig")
    max_num = 3

    while True:
        if len(patient_list) == 63:
            break
        if len(np.where(p_reg_num > 3)[0]) >= 40:
            max_num = 2
            up_id = np.where(p_reg_num > 3)[0]
            for j in up_id:
                if j in pid_list:
                    patient_list.remove(j)
        if len(np.where(p_reg_num > 2)[0]) >= 55:
            max_num = 1
            up_id = np.where(p_reg_num > 2)[0]
            for j in up_id:
                if j in pid_list:
                    patient_list.remove(j)
        for doc in doc_file.values:
            if d_reg_num[doc[0]] == doc[1]:
                continue
            temp_list = pid_list.copy()
            for i in d_p[doc[0]]:
                if i in temp_list:
                    temp_list.remove(i)
            if not temp_list:
                continue
            pid = np.random.choice(a=temp_list, size=1, replace=False, p=None)[0]
            p = [pid, doc[0], 0, doc[3]]
            d_p[doc[0]].append(pid)
            p_reg_num[pid] += 1
            d_reg_num[doc[0]] += 1
            patient_list.append(p)
            if p_reg_num[pid] > max_num:
                pid_list.remove(pid)

    return patient_list


def distance_d(doc_num, file_name):
    dis_arr = np.random.choice(a=[1, 2, 3, 4, 5], size=doc_num*doc_num, replace=True).reshape(doc_num, doc_num)
    dis_arr = np.triu(dis_arr)
    for i in range(dis_arr.shape[0]):
        dis_arr[i][i] = 0
    dis_arr += dis_arr.T - np.diag(dis_arr.diagonal())
    dis_arr = pd.DataFrame(data=dis_arr.tolist(), index=None)
    dis_arr.to_csv(f"../data/{file_name}.csv", index=False, header=None)


def gen_data(file_name, p_num, d_num, multi):
    data_list = []
    p_st = np.ones((1, p_num), dtype="int64")
    multi_reg = np.random.choice(a=p_num, p=None, size=multi, replace=False)
    p_st[0][multi_reg] = 3
    d_st = np.zeros((1, d_num), dtype="int64")
    p_a = [_ for _ in range(p_num)]
    d_a = [_ for _ in range(d_num)]
    job_num = 0
    legal_mask = np.ones((d_num, p_num), dtype=bool)
    while job_num < 78:
        pid = np.random.choice(a=p_a, p=None, size=1)[0]
        did = np.random.choice(a=d_a, p=None, size=1)[0]
        if legal_mask[did][pid]:
            pro_time = random.randint(10, 20)
            data_list.append([pid, did, pro_time])
            p_st[0][pid] -= 1
            d_st[0][did] += 1
            if p_st[0][pid] == 0:
                p_a.remove(pid)
            if d_st[0][did] == 8:
                d_a.remove(did)
            job_num += 1
            legal_mask[did][pid] = False
    data = pd.DataFrame(data=data_list, columns=["pid", "did", "pro_time"], index=None)
    data.to_csv(f"../data/{file_name}.csv", index=False)


if __name__ == '__main__':
    # distance_d(10, "distance")
    gen_data("reg_data_1", 60, 10, 9)
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
