# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:54
# @Author  : hxc
# @File    : other_rules.py
# @Software: PyCharm
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv

from tools.check_time import check_time


class Rules:
    def __init__(self, path):
        self.path = path
        # 每个job的处理时间
        self.remaining_pro_time = None
        # 每个零件的剩余处理时间
        self.remaining_time = None
        self.file = None
        self.p_num = None
        self.d_num = None
        self.data_list = []
        self.reg_job_id = list()
        self.init()

    def init(self):
        self.file = pd.read_csv(self.path, encoding='utf-8-sig').fillna('')
        self.file = shuffle(self.file)
        self.p_num = self.file.groupby("pid").count().shape[0]
        self.d_num = self.file.groupby("did").count().shape[0]
        self.remaining_pro_time = np.zeros((self.file.shape[0],))
        self.remaining_time = np.zeros((self.p_num,))
        self.reg_job_id = [[] for _ in range(self.p_num)]
        for i in range(self.file.shape[0]):
            self.remaining_pro_time[i] += self.file.values[i][2]
            self.remaining_time[self.file.values[i][0]] += self.file.values[i][2]
            self.reg_job_id[self.file.values[i][0]].append(i)

    # 1 LWKR 选择剩余加工时间最短的工件
    def LWKR(self):
        remaining_time = np.array(self.remaining_time.tolist())
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.p_num):
            pid = np.argmin(remaining_time)
            for job_id in self.reg_job_id[pid]:
                did = self.file.values[job_id][1]
                pro_time = self.file.values[job_id][2]
                if not sc_list[self.file.values[job_id][1]]:
                    if record_finish[pid] == 0:
                        start_time = 0
                    else:
                        start_time = record_finish[pid]
                else:
                    if record_finish[pid] > sc_list[did][-1][4]:
                        start_time = record_finish[pid]
                    else:
                        start_time = sc_list[did][-1][4]
                info = [did, pid, start_time, pro_time, start_time + pro_time]
                sc_list[did].append(info)
                record_finish[pid] = start_time + pro_time
            remaining_time[pid] = 999999

        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        return self.save_data("lwkr")

    # 2 MWKR 选择剩余加工时间最长的工件
    def MWKR(self):
        remaining_time = np.array(self.remaining_time.tolist())
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.p_num):
            pid = np.argmax(remaining_time)
            for job_id in self.reg_job_id[pid]:
                did = self.file.values[job_id][1]
                pro_time = self.file.values[job_id][2]
                if not sc_list[self.file.values[job_id][1]]:
                    if record_finish[pid] == 0:
                        start_time = 0
                    else:
                        start_time = record_finish[pid]
                else:
                    if record_finish[pid] > sc_list[did][-1][4]:
                        start_time = record_finish[pid]
                    else:
                        start_time = sc_list[did][-1][4]
                info = [did, pid, start_time, pro_time, start_time + pro_time]
                sc_list[did].append(info)
                record_finish[pid] = start_time + pro_time
            remaining_time[pid] = 0

        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        return self.save_data("mwkr")

    # 3 SPT 选择工序加工时间最短的工件
    def SPT(self):
        remaining_pro_time = np.array(self.remaining_pro_time.tolist())
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.file.shape[0]):
            job_id = np.argmin(remaining_pro_time)
            pid = self.file.values[job_id][0]
            did = self.file.values[job_id][1]
            pro_time = self.file.values[job_id][2]
            if not sc_list[self.file.values[job_id][1]]:
                if record_finish[pid] == 0:
                    start_time = 0
                else:
                    start_time = record_finish[pid]
            else:
                if record_finish[pid] > sc_list[did][-1][4]:
                    start_time = record_finish[pid]
                else:
                    start_time = sc_list[did][-1][4]
            info = [did, pid, start_time, pro_time, start_time + pro_time]
            sc_list[did].append(info)
            remaining_pro_time[job_id] = 999999
            record_finish[pid] = start_time + pro_time
        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        return self.save_data("test_result")

    # 4 LPT 选择工序加工时间最长的工件
    def LPT(self):
        remaining_pro_time = np.array(self.remaining_pro_time.tolist())
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.file.shape[0]):
            job_id = np.argmax(remaining_pro_time)
            pid = self.file.values[job_id][0]
            did = self.file.values[job_id][1]
            pro_time = self.file.values[job_id][2]
            if not sc_list[self.file.values[job_id][1]]:
                if record_finish[pid] == 0:
                    start_time = 0
                else:
                    start_time = record_finish[pid]
            else:
                if record_finish[pid] > sc_list[did][-1][4]:
                    start_time = record_finish[pid]
                else:
                    start_time = sc_list[did][-1][4]
            info = [did, pid, start_time, pro_time, start_time + pro_time]
            sc_list[did].append(info)
            remaining_pro_time[job_id] = 0
            record_finish[pid] = start_time + pro_time
        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        return self.save_data("lpt")
    # 5 SPT/TWK 工序加工时间与总加工时间比值最小的工件
    # 6 LPT/TWK 工序加工时间与总加工时间比值最大的工件

    # 7 SPT/TWKR 工序加工时间与剩余加工时间比值最小的工件
    def spt_twkr(self):
        remaining_pro_time = np.array(self.remaining_pro_time.tolist())
        remaining_time = np.array(self.remaining_time.tolist())
        for i in range(self.p_num):
            for j_id in self.reg_job_id[i]:
                remaining_pro_time[j_id] = remaining_pro_time[j_id] / remaining_time[i]
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.file.shape[0]):
            job_id = np.argmin(remaining_pro_time)
            pid = self.file.values[job_id][0]
            did = self.file.values[job_id][1]
            pro_time = self.file.values[job_id][2]
            if not sc_list[self.file.values[job_id][1]]:
                if record_finish[pid] == 0:
                    start_time = 0
                else:
                    start_time = record_finish[pid]
            else:
                if record_finish[pid] > sc_list[did][-1][4]:
                    start_time = record_finish[pid]
                else:
                    start_time = sc_list[did][-1][4]
            info = [did, pid, start_time, pro_time, start_time + pro_time]
            sc_list[did].append(info)
            remaining_pro_time[job_id] = 999999
            remaining_time[pid] -= pro_time
            if remaining_time[pid] != 0:
                for j in self.reg_job_id[pid]:
                    remaining_pro_time[j] = remaining_pro_time[j] / remaining_time[pid]
            record_finish[pid] = start_time + pro_time
        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        self.save_data("spt_twkr")

    # 8 LPT/TWKR 工序加工时间与剩余加工时间比值最大的工件
    def lpt_twkr(self):
        remaining_pro_time = np.array(self.remaining_pro_time.tolist())
        remaining_time = np.array(self.remaining_time.tolist())
        for i in range(self.p_num):
            for j_id in self.reg_job_id[i]:
                remaining_pro_time[j_id] = remaining_pro_time[j_id] / remaining_time[i]
        record_finish = np.zeros((self.p_num,))
        sc_list = [[] for _ in range(self.d_num)]
        for step in range(self.file.shape[0]):
            job_id = np.argmax(remaining_pro_time)
            pid = self.file.values[job_id][0]
            did = self.file.values[job_id][1]
            pro_time = self.file.values[job_id][2]
            if not sc_list[self.file.values[job_id][1]]:
                if record_finish[pid] == 0:
                    start_time = 0
                else:
                    start_time = record_finish[pid]
            else:
                if record_finish[pid] > sc_list[did][-1][4]:
                    start_time = record_finish[pid]
                else:
                    start_time = sc_list[did][-1][4]
            info = [did, pid, start_time, pro_time, start_time + pro_time]
            sc_list[did].append(info)
            remaining_pro_time[job_id] = -1
            remaining_time[pid] -= pro_time
            if remaining_time[pid] != 0:
                for j in self.reg_job_id[pid]:
                    remaining_pro_time[j] = remaining_pro_time[j] / remaining_time[pid]
            record_finish[pid] = start_time + pro_time
        self.data_list = []
        for i in range(self.d_num):
            for s in sc_list[i]:
                self.data_list.append(s)
        self.save_data("lpt_twkr")
    # 9 SPT*TWK 工序加工时间与总加工时间乘积最小的工件
    # 10 LPT*TWK 工序加工时间与总加工时间乘积最大的工件
    # 11 SPT*TWKR 工序加工时间与剩余加工时间乘积最小的工件
    # 12 LPT*TWKR 工序加工时间与剩余加工时间乘积最大的工件
    # 13 SRM 除当前工序外所剩加工时间最短的工件
    # 14 LRM 除当前工序外所剩加工时间最长的工件
    # 15 SSO 后继工序加工时间最短的工件
    # 16 LSO 后继工序加工时间最长的工件
    # 17 SPT+SSO 当前工序加工时间与后继工序加工时间最短工件
    # 18 LPT+SSO 当前工序加工时间与后继工序加工时间最长工件
    # 19 SPT/LSO 当前工序加工时间与后继工序加工时间比值最小工件
    # 20 LPT/SSO 当前工序加工时间与后继工序加工时间比值最大工件

    def save_data(self, file_name):
        df = pd.DataFrame(data=self.data_list, columns=["did", "pid", "start_time", "pro_time", "finish_time"])
        d, p, d_idle = check_time(file=df)
        # print(d_idle)
        return d_idle
        # with open(f'../data/save_data/{file_name}.csv', mode='w+', encoding='utf-8-sig', newline='') as f:
        #     csv_writer = csv.writer(f)
        #     headers = ['did', 'pid', 'start_time', 'pro_time', 'finish_time']
        #     csv_writer.writerow(headers)
        #     csv_writer.writerows(self.data_list)
        #     print(f'保存结果文件')


if __name__ == '__main__':
    files = os.listdir("../data/simulation_instances")
    for file in files:
        # d_idle_list = []
        LWKR_idle_list = []
        MWKR_idle_list = []
        spt_idle_list = []
        lpt_idle_list = []
        rule = Rules(f"../data/simulation_instances/{file}")
        for i in range(100):
            LWKR_idle = rule.LWKR()
            MWKR_idle = rule.MWKR()
            spt_idle = rule.SPT()
            lpt_idle = rule.LPT()

            # d_idle = main(i, f"../data/simulation_instances/{file}")
            LWKR_idle_list.append(LWKR_idle)
            MWKR_idle_list.append(MWKR_idle)
            spt_idle_list.append(spt_idle)
            lpt_idle_list.append(lpt_idle)
        print(file)

        print("LWKR")
        print("Mean:", np.mean(LWKR_idle_list))
        print("Std:", np.std(LWKR_idle_list))
        confidence_interval = np.percentile(np.array(LWKR_idle_list), [2.5, 97.5])
        print("Confidence interval（95%）:", confidence_interval)

        print("MWKR")
        print("Mean:", np.mean(MWKR_idle_list))
        print("Std:", np.std(MWKR_idle_list))
        confidence_interval = np.percentile(np.array(MWKR_idle_list), [2.5, 97.5])
        print("Confidence interval（95%）:", confidence_interval)

        print("SPT")
        print("Mean:", np.mean(spt_idle_list))
        print("Std:", np.std(spt_idle_list))
        confidence_interval = np.percentile(np.array(spt_idle_list), [2.5, 97.5])
        print("Confidence interval（95%）:", confidence_interval)

        print("LPT")
        print("Mean:", np.mean(lpt_idle_list))
        print("Std:", np.std(lpt_idle_list))
        confidence_interval = np.percentile(np.array(lpt_idle_list), [2.5, 97.5])
        print("Confidence interval（95%）:", confidence_interval)
    # rule = Rules("../data/test_data.csv")
    # rule.SPT()
    # rule.LPT()
