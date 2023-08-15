# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 11:12
# @Author  : hxc
# @File    : ACO.py
# @Software: PyCharm
import numpy
import numpy as np
import pandas as pd
import os
from get_alg_result import get_solution
files = os.listdir("../data/simulation_instances")

for file in files:
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
        num_ants = 10  # 蚂蚁数量
        num_iterations = 20  # 迭代次数
        pheromone = np.ones((num_jobs, num_machines))  # 信息素矩阵
        alpha = 1.0  # 信息素重要程度
        beta = 2.0  # 启发式因子重要程度
        evaporation = 0.5  # 信息素蒸发率

        # 初始化最优解
        best_solution = None
        best_fitness = np.inf

        # 主循环
        for iteration in range(num_iterations):
            # 初始化解空间
            solutions = np.zeros((num_ants, num_jobs), dtype=int)

            # 蚂蚁构建解空间
            for ant in range(num_ants):
                visited = np.zeros(num_jobs, dtype=bool)  # 记录零件是否已安排
                solution = np.zeros(num_jobs, dtype=int)  # 解向量
                time_left = np.zeros(num_machines, dtype=int)  # 机器剩余处理时间

                # 选择下一个零件
                for i in range(num_jobs):
                    valid_choices = np.where(~visited)[0]  # 未安排的零件
                    machine_probs = pheromone[valid_choices, machine_ids[valid_choices]] ** alpha  # 信息素权重
                    time_probs = (1.0 / (time_left[machine_ids[valid_choices]] + 1)) ** beta  # 启发式因子权重
                    probabilities = machine_probs * time_probs  # 概率计算
                    probabilities /= np.sum(probabilities)  # 归一化

                    next_job = np.random.choice(valid_choices, p=probabilities)  # 选择下一个零件
                    solution[i] = next_job  # 更新解向量
                    visited[next_job] = True  # 更新已访问列表
                    time_left[machine_ids[next_job]] += processing_times[next_job]

                solutions[ant] = solution

            # 评估解空间中的解
            fitness_values = np.zeros(num_ants)

            for ant in range(num_ants):
                machine_finish_times = np.zeros(num_machines)  # 机器完成时间
                for i in range(num_jobs):
                    job = solutions[ant][i]
                    machine = machine_ids[job]
                    start_time = max(machine_finish_times[machine], machine_finish_times[machine])
                    finish_time = start_time + processing_times[job]
                    machine_finish_times[machine] = finish_time

                fitness_values[ant] = np.max(machine_finish_times)

            # 更新最优解
            min_fitness = np.min(fitness_values)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_solution = solutions[np.argmin(fitness_values)]

            # 更新信息素
            pheromone *= evaporation  # 信息素蒸发
            for ant in range(num_ants):
                for i in range(num_jobs):
                    job = solutions[ant][i]
                    machine = machine_ids[job]
                    pheromone[job][machine] += 1.0 / best_fitness
        total_time_d, total_time_p, d_idle, p_idle, total_idle = get_solution(best_solution,
                                                                              f"../data/simulation_instances/{file}")

        d_idle_list.append([p_idle, d_idle, total_idle, total_time_d])
    df = pd.DataFrame(data=d_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
    df.to_csv(f"../data/simulation_results/result_ACO_{file}", index=False)
    print(file)
# best_solution = numpy.sort(best_solution)

# 打印最优解
# print("Best Solution:", best_solution)
# print("Best Fitness:", best_fitness)
