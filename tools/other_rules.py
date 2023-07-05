# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 16:54
# @Author  : hxc
# @File    : other_rules.py
# @Software: PyCharm
import os
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import csv

# from tools.check_time import check_time
from check_time import check_time


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
        # return self.save_data("test_result")

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
        total_time_d, total_time_p, d_idle, p_idle, total_idle = check_time(file=df)
        # print(d_idle)
        return total_time_d, total_time_p, d_idle, p_idle, total_idle
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
        LWKR_time_count = 0
        MWKR_time_count = 0
        SPT_time_count = 0
        LPT_time_count = 0
        rule = Rules(f"../data/simulation_instances/{file}")
        for i in range(100):
            LWKR_start_time = time.perf_counter()
            LWKR_total_time_d, LWKR_total_time_p, LWKR_d_idle, LWKR_p_idle, LWKR_total_idle = rule.LWKR()
            LWKR_end_time = time.perf_counter()
            LWKR_time_count += LWKR_end_time - LWKR_start_time

            MWKR_start_time = time.perf_counter()
            MWKR_total_time_d, MWKR_total_time_p, MWKR_d_idle, MWKR_p_idle, MWKR_total_idle = rule.MWKR()
            MWKR_end_time = time.perf_counter()
            MWKR_time_count += MWKR_end_time - MWKR_start_time

            SPT_start_time = time.perf_counter()
            SPT_total_time_d, SPT_total_time_p, SPT_d_idle, SPT_p_idle, SPT_total_idle = rule.SPT()
            SPT_end_time = time.perf_counter()
            SPT_time_count += SPT_end_time - SPT_start_time

            LPT_start_time = time.perf_counter()
            LPT_total_time_d, LPT_total_time_p, LPT_d_idle, LPT_p_idle, LPT_total_idle = rule.LPT()
            LPT_end_time = time.perf_counter()
            LPT_time_count += LPT_end_time - LPT_start_time

            # d_idle = main(i, f"../data/simulation_instances/{file}")
            LWKR_idle_list.append([LWKR_p_idle, LWKR_d_idle, LWKR_total_idle, LWKR_total_time_d])
            MWKR_idle_list.append([MWKR_p_idle, MWKR_d_idle, MWKR_total_idle, MWKR_total_time_d])
            spt_idle_list.append([SPT_p_idle, SPT_d_idle, SPT_total_idle, SPT_total_time_d])
            lpt_idle_list.append([LPT_p_idle, LPT_d_idle, LPT_total_idle, LPT_total_time_d])
        print(file)
        print("LWKR_time_count", LWKR_time_count)
        print("MWKR_time_count", MWKR_time_count)
        print("SPT_time_count", SPT_time_count)
        print("LPT_time_count", LPT_time_count)
        pd.DataFrame(data=LWKR_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"]).to_csv(
            f"../data/simulation_results/result_LWKR_{file}", index=False)
        pd.DataFrame(data=MWKR_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"]).to_csv(
            f"../data/simulation_results/result_MWKR_{file}", index=False)
        pd.DataFrame(data=spt_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"]).to_csv(
            f"../data/simulation_results/result_SPT_{file}", index=False)
        pd.DataFrame(data=lpt_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"]).to_csv(
            f"../data/simulation_results/result_LPT_{file}", index=False)
    # rule = Rules("../data/test_data.csv")
    # rule.SPT()
    # rule.LPT()
    """
    5_150_180.csv
    LWKR_time_count 4.4017668925225735
    MWKR_time_count 4.39379720017314
    SPT_time_count 4.482285372912884
    LPT_time_count 4.4774572141468525
    5_150_179.csv
    LWKR_time_count 4.389691393822432
    MWKR_time_count 4.382357023656368
    SPT_time_count 4.467848066240549
    LPT_time_count 4.467322647571564
    30_900_1041.csv
    LWKR_time_count 25.57898547500372
    MWKR_time_count 25.502608075737953
    SPT_time_count 26.06779347732663
    LPT_time_count 26.03086845576763
    30_900_1039.csv
    LWKR_time_count 25.517048377543688
    MWKR_time_count 25.558426588773727
    SPT_time_count 26.019618965685368
    LPT_time_count 26.03498015552759
    25_750_878.csv
    LWKR_time_count 21.322479620575905
    MWKR_time_count 21.305060625076294
    SPT_time_count 21.73854999244213
    LPT_time_count 21.79307121410966
    25_750_875.csv
    LWKR_time_count 21.33624890819192
    MWKR_time_count 21.28655631840229
    SPT_time_count 21.759806890040636
    LPT_time_count 21.715739365667105
    20_600_715.csv
    LWKR_time_count 17.119035348296165
    MWKR_time_count 17.0957213640213
    SPT_time_count 17.470539581030607
    LPT_time_count 17.461254566907883
    15_450_535.csv
    LWKR_time_count 12.86738782003522
    MWKR_time_count 12.857424091547728
    SPT_time_count 13.118995420634747
    LPT_time_count 13.113761205226183
    15_450_534.csv
    LWKR_time_count 12.868749011307955
    MWKR_time_count 12.856474418193102
    SPT_time_count 13.115219946950674
    LPT_time_count 13.117991514503956
    10_300_357.csv
    LWKR_time_count 8.624977227300406
    MWKR_time_count 8.614013906568289
    SPT_time_count 8.789154700934887
    LPT_time_count 8.794267315417528
    10_300_351.csv
    LWKR_time_count 8.608302723616362
    MWKR_time_count 8.595200922340155
    SPT_time_count 8.767228912562132
    LPT_time_count 8.767459072172642
    """
