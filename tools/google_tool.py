# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 16:21
# @Author  : hxc
# @File    : google_tool.py
# @Software: PyCharm

"""Minimal jobshop example."""
import collections
import os
import time

from tools.check_time import check_time
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import csv


def save_data(data_list):
    all_data = []
    for doc_id in range(len(data_list)):
        for task in data_list[doc_id]:
            all_data.append([doc_id, int(task[1]), task[0], int(task[3]), int(task[0] + int(task[3]))])
    df = pd.DataFrame(data=all_data, columns=["did", "pid", "start_time", "pro_time", "finish_time"])
    total_time_d, total_time_p, d_idle, p_idle, total_idle = check_time(file=df)
    return total_time_d, total_time_p, d_idle, p_idle, total_idle
    # with open(f"../data/save_data/or_tool_{i}.csv", mode="w+", newline="") as f:
    #     csv_w = csv.writer(f)
    #     header = ["did", "pid", "start_time", "pro_time", "finish_time"]
    #     csv_w.writerow(header)
    #     csv_w.writerows(all_data)


def get_data(path):
    reg = pd.read_csv(path)
    p_num = reg.groupby("pid").count().shape[0]
    data_list = [[] for _ in range(p_num)]
    for p in reg.values:
        did = p[1]
        pid = p[0]
        pro_time = p[2]
        data_list[pid].append((did, pro_time))
    return data_list


def main(path):
    """Minimal jobshop problem."""
    # Data.
    # jobs_data = [  # task = (machine_id, processing_time).
    #     [(0, 3), (1, 2), (2, 2)],  # Job0
    #     [(0, 2), (2, 1), (1, 4)],  # Job1
    #     [(1, 4), (2, 3)]  # Job2
    # ]
    jobs_data = get_data(path)

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)
    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Create the model.
    model = cp_model.CpModel()

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(start=solver.Value(
                        all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
        d_total_time, p_total_time, doc_idle, pat_idle, total_idl = save_data(assigned_jobs)
        return d_total_time, p_total_time, doc_idle, pat_idle, total_idl

    else:
        print('No solution found.')


if __name__ == '__main__':
    files = os.listdir("../data/simulation_instances")
    """
    5_150_180.csv 9.429400779306889
    5_150_179.csv 10.728971730917692
    30_900_1041.csv 62.70113477855921
    30_900_1039.csv 54.070056181401014
    25_750_878.csv 38.48305977135897
    25_750_875.csv 41.6258698888123
    20_600_715.csv 56.54425938427448
    20_600_704.csv 31.42456056177616
    15_450_535.csv 24.376219894737005
    15_450_534.csv 24.996688049286604
    10_300_357.csv 16.974918179214
    10_300_351.csv 29.052186463028193
    """
    for file in files:
        start_time = time.perf_counter()
        d_idle_list = []
        for i in range(1):
            total_time_d, total_time_p, d_idle, p_idle, total_idle = main(f"../data/simulation_instances/{file}")
            d_idle_list.append([p_idle, d_idle, total_idle, total_time_d])

        df = pd.DataFrame(data=d_idle_list, columns=["p_idle", "d_idle", "total_idle_time", "d_total_time"])
        df.to_csv(f"../data/simulation_results1/result_or_{file}", index=False)
        end_time = time.perf_counter()
        time_count = end_time - start_time
        print(file, time_count)
