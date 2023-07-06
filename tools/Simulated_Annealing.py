# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 11:36
# @Author  : hxc
# @File    : Simulated_Annealing.py
# @Software: PyCharm
import numpy
import numpy as np
import time
import pandas as pd
import os
from get_alg_result import get_solution


files = os.listdir("../data/simulation_instances")

for file in files:
    start_time = time.perf_counter()
    d_idle_list = []
    for i in range(100):
        # 读取CSV文件
        data = pd.read_csv(f'../data/simulation_instances/{file}')

        # 提取数据
        job_ids = data['pid'].values
        machine_ids = data['did'].values
        processing_times = data['pro_time'].values

        # 参数设置
        num_jobs = len(job_ids)  # 零件数量
        num_machines = len(np.unique(machine_ids))  # 机器数量
        num_iterations = 1000  # 迭代次数
        initial_temperature = 100.0  # 初始温度
        final_temperature = 0.1  # 终止温度
        cooling_rate = 0.9  # 冷却速率

        # 初始化当前解和最优解
        current_solution = np.random.permutation(num_jobs)  # 随机生成初始解
        best_solution = current_solution.copy()
        best_fitness = np.inf

        # 计算初始适应度
        machine_finish_times = np.zeros(num_machines)  # 机器完成时间
        for i in range(num_jobs):
            job = current_solution[i]
            machine = machine_ids[job]
            start_time = max(machine_finish_times[machine], machine_finish_times[machine])
            finish_time = start_time + processing_times[job]
            machine_finish_times[machine] = finish_time

        current_fitness = np.max(machine_finish_times)

        # 模拟退火算法
        temperature = initial_temperature

        for iteration in range(num_iterations):
            # 生成新解
            new_solution = current_solution.copy()
            # 在当前解中随机选择两个位置进行交换
            swap_indices = np.random.choice(num_jobs, size=2, replace=False)
            new_solution[swap_indices[0]], new_solution[swap_indices[1]] = (
                new_solution[swap_indices[1]],
                new_solution[swap_indices[0]],
            )

            # 计算新解的适应度
            machine_finish_times = np.zeros(num_machines)
            for i in range(num_jobs):
                job = new_solution[i]
                machine = machine_ids[job]
                start_time = max(machine_finish_times[machine], machine_finish_times[machine])
                finish_time = start_time + processing_times[job]
                machine_finish_times[machine] = finish_time

            new_fitness = np.max(machine_finish_times)

            # 判断是否接受新解
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness

                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
            else:
                # 根据Metropolis准则以一定概率接受差解
                acceptance_probability = np.exp(-(new_fitness - current_fitness) / temperature)
                if np.random.rand() < acceptance_probability:
                    current_solution = new_solution
                    current_fitness = new_fitness

            # 降低温度
            temperature *= cooling_rate

        # 打印最优解
        # best_solution = numpy.sort(best_solution)
        total_time_d, total_time_p, d_idle, p_idle, total_idle = get_solution(best_solution, f"../data/simulation_instances/{file}")

        d_idle_list.append([p_idle, d_idle, total_idle, total_time_d])

    df = pd.DataFrame(data=d_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
    df.to_csv(f"../data/simulation_results/result_SA_{file}", index=False)
    end_time = time.perf_counter()
    # time_count = end_time - start_time
    print(file)
    # print("Best Solution:", best_solution)
    # print("Best Fitness:", best_fitness)
