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


def get_num(all_reg_num):
    import pulp
    # 创建整数线性规划问题
    problem = pulp.LpProblem("Equation_Solving", pulp.LpMinimize)

    # 定义变量
    x = pulp.LpVariable("x", lowBound=0, cat='Integer')
    y = pulp.LpVariable("y", lowBound=0, cat='Integer')
    z = pulp.LpVariable("z", lowBound=0, cat='Integer')

    # 添加约束
    problem += y + z == 0.1 * x
    problem += 3 * y + 2 * z + 0.9 * x == all_reg_num

    # 求解问题
    status = problem.solve()

    if status == pulp.LpStatusOptimal:
        # 输出结果
        x_ = pulp.value(x)
        y_ = pulp.value(y)
        z_ = pulp.value(z)
        # print("x =", pulp.value(x))
        # print("y =", pulp.value(y))
        # print("z =", pulp.value(z))
        return x_, y_, z_
    else:
        print("No optimal solution found.")


def gen_data(num_reg, num_doctors, seed):
    np.random.seed(seed)
    if num_reg % num_doctors != 0:
        max_pro_num = ((num_reg // num_doctors) + 1)
    else:
        max_pro_num = (num_reg // num_doctors)
    num_patients, num_multi3, num_multi2 = get_num(int(num_reg))
    num_patients, num_multi3, num_multi2 = int(num_patients), int(num_multi3), int(num_multi2)

    multi_3 = np.random.choice(a=num_patients, size=num_multi3, replace=False)
    multi_2 = np.random.choice(a=np.setdiff1d(np.arange(num_patients), multi_3), size=num_multi2, replace=False)

    pro_time = np.random.randint(7, 20 + 1, num_doctors)
    # 应用布尔掩码来删除数组b中的元素
    all_reg_list = []
    doc_reg_num = np.zeros((num_doctors,))
    d_list = [i for i in range(num_doctors)]

    for pid in multi_3:
        # did_list = np.random.permutation(d_list)
        d_idx = np.random.choice(a=d_list, size=3, replace=False)

        for d in d_idx:
            all_reg_list.append([pid, d, pro_time[d]])
            doc_reg_num[d] += 1
            if doc_reg_num[d] == max_pro_num:
                d_list.remove(d)

    for pid in multi_2:
        # did_list = np.random.permutation(d_list)
        d_idx = np.random.choice(a=d_list, size=2, replace=False)

        for d in d_idx:
            all_reg_list.append([pid, d, pro_time[d]])
            doc_reg_num[d] += 1
            if doc_reg_num[d] == max_pro_num:
                d_list.remove(d)

    for i in range(num_patients):
        np.random.seed(int(i))

        if i in multi_3:
            continue
        elif i in multi_2:
            continue
        else:
            d_idx = np.random.choice(a=d_list, size=1, replace=False)

            for d in d_idx:
                all_reg_list.append([i, d, pro_time[d]])
                doc_reg_num[d] += 1
                if doc_reg_num[d] == max_pro_num:
                    d_list.remove(d)
    df = pd.DataFrame(data=all_reg_list, columns=["pid", "did", "pro_time"])
    df.to_csv(f"../data/simulation_instances/{seed}.csv", index=False)
    # return df


if __name__ == '__main__':
    # distance_d(10, "distance")
    for i in range(1, 10+1):
        gen_data(232, 6, i)
    # for i in range(0, 10000, 100):
    #     gen_data(i, 300, 10)
