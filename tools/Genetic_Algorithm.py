# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 10:45
# @Author  : hxc
# @File    : Genetic_Algorithm.py
# @Software: PyCharm
import os
import random
import csv
import numpy as np
from check_time import check_time
import pandas as pd

# 定义问题相关的参数
POPULATION_SIZE = 100
MAX_GENERATIONS = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1

# 读取CSV文件
def read_data_from_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            pid = int(row[0])
            did = int(row[1])
            pro_time = int(row[2])
            data.append((pid, did, pro_time))
    return data

# 初始化种群
def initialize_population(data):
    population = []
    for _ in range(POPULATION_SIZE):
        individual = random.sample(data, len(data))  # 随机打乱顺序
        population.append(individual)
    return population

# 计算适应度
def fitness_function(individual, machine_count):
    machine_end_times = [0] * machine_count  # 记录每个机器的当前完成时间  # 记录每个机器的当前完成时间
    total_time = 0
    for (pid, did, pro_time) in individual:
        machine_end_times[did] = max(machine_end_times[did], total_time) + pro_time
        total_time = max(machine_end_times)
    return total_time

# 选择操作
def selection(population, machine_count):
    selected = []
    fitness_values = [fitness_function(individual, machine_count) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    for _ in range(len(population)):
        selected.append(random.choices(population, probabilities)[0])
    return selected

# 交叉操作
def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1[crossover_point:] = [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        child2[crossover_point:] = [gene for gene in parent1 if gene not in parent2[:crossover_point]]
    return child1, child2

# 变异操作
def mutation(individual):
    if random.random() < MUTATION_RATE:
        index1 = random.randint(0, len(individual) - 1)
        index2 = random.randint(0, len(individual) - 1)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

# 更新种群
def update_population(population, selected, machine_count):
    population.extend(selected)
    fitness_values = [fitness_function(individual, machine_count) for individual in population]
    sorted_population = [individual for _, individual in sorted(zip(fitness_values, population))]
    return sorted_population[:POPULATION_SIZE]

# 主函数
def main():
    files = os.listdir("../data/simulation_instances")
    for file in files:
        d_idle_list = []
        for i in range(100):

            data = read_data_from_csv(f'../data/simulation_instances/{file}')
            d_num = int(file.split("_")[0])
            machine_count = d_num
            population = initialize_population(data)

            for generation in range(MAX_GENERATIONS):
                selected = selection(population, machine_count)
                offspring = []

                for i in range(0, len(selected), 2):
                    parent1 = selected[i]
                    parent2 = selected[i+1]
                    child1, child2 = crossover(parent1, parent2)
                    child1 = mutation(child1)
                    child2 = mutation(child2)
                    offspring.append(child1)
                    offspring.append(child2)

                population = update_population(population, offspring, machine_count)
                best_individual = population[0]
                best_fitness = fitness_function(best_individual, machine_count)

                # print(f"Generation {generation+1}: Best Fitness = {best_fitness}")

            best_individual = population[0]
            # print("Best Individual:")
            # print(len(best_individual))
            d = pd.read_csv(f"../data/simulation_instances/{file}")
            d_num = d.groupby("did").count().shape[0]
            p_num = d.groupby("pid").count().shape[0]
            d_last_time = np.zeros((d_num,))
            p_last_time = np.zeros((p_num,))
            sc_list = []

            for (pid, did, pro_time) in best_individual:
                if d_last_time[did] >= p_last_time[pid]:
                    start_time = d_last_time[did]
                else:
                    start_time = p_last_time[pid]
                finish_time = start_time + pro_time
                d_last_time[did] = finish_time
                p_last_time[pid] = finish_time
                sc_list.append([did, pid, start_time, pro_time, finish_time])
            df = pd.DataFrame(data=sc_list, columns=["did", "pid", "start_time", "pro_time", "finish_time"])
            total_time_d, total_time_p, d_idle, p_idle, total_idle = check_time(file=df)

            d_idle_list.append([p_idle, d_idle, total_idle, total_time_d])

        df = pd.DataFrame(data=d_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
        df.to_csv(f"../data/simulation_results/result_GA_{file}", index=False)
        print(file)


if __name__ == "__main__":
    main()
