# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 14:21
# @Author  : hxc
# @File    : plot_graph.py
# @Software: PyCharm
import networkx as nx
import matplotlib.pyplot as plt
import csv

# 创建一个空的有向图
graph = nx.DiGraph()

# 从CSV文件中读取节点信息
with open('../data/test_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    n = 0
    for row in reader:
        pid, did, pro_time = row  # 假设CSV文件的列顺序是 pid, did, pro_time

        # 添加节点到图中
        graph.add_node(n, pid=pid, did=did, pro_time=pro_time)
        n += 1

# 添加边连接相同pid的节点
for node in graph.nodes():
    pid = graph.nodes[node]['pid']
    nodes_with_same_pid = [n for n in graph.nodes() if graph.nodes[n]['pid'] == pid]
    for n in nodes_with_same_pid:
        if n != node:
            graph.add_edge(node, n)

# 添加边连接相同did的节点
for node in graph.nodes():
    did = graph.nodes[node]['did']
    nodes_with_same_did = [n for n in graph.nodes() if graph.nodes[n]['did'] == did]
    for n in nodes_with_same_did:
        if n != node:
            graph.add_edge(node, n)

# 为不同的did设置不同的颜色
pid_colors = {0: "red"}
for node in graph.nodes():
    pid = graph.nodes[node]['pid']
    if pid not in pid_colors:
        pid_colors[pid] = len(pid_colors)  # 使用不同的整数值来表示不同的颜色
    color = pid_colors[pid]
    graph.nodes[node]['color'] = color

# 可视化图结构
pos = nx.spring_layout(graph)  # 定义节点的布局方式
# 提取节点的颜色信息
node_colors = [graph.nodes[node]['color'] for node in graph.nodes()]

nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=500, node_color=node_colors, cmap='rainbow')
# nx.draw_networkx(graph, pos=pos, with_labels=True, node_size=500, node_color='lightblue')
plt.figure(figsize=(50, 50), dpi=200)
plt.show()
