from typing import List

import networkx as nx
import random
import torch
from matplotlib import pyplot as plt
from collections import defaultdict


def generate_G_from_edge():
    file = open("E:\keyan\GATNE\data\ml\ml_edgeList.txt", "w")
    with open("E:\keyan\GATNE\data\ml\\ratings.txt") as f:
        lines = f.readlines()
        for line in lines:
            s = line[2:]
            file.write(s)


def generate_ratings():
    file = open("E:\keyan\GATNE\data\ml\\ratings.txt", "w")
    with open("E:\keyan\part2\BiNE\data\ml-1m\\ratings") as f:
        lines = f.readlines()
        for line in lines:
            list = line.strip().split(" ")
            s = list[2] + ' ' + list[0] + ' ' + list[1] + '\n'
            print(s)
            file.write(s)
    file.close()


def load_graph(filename):
    G = nx.read_edgelist(filename,
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G


def graph_to_edge_list(G):
    edge_list = []
    for edge in G.edges():
        edge_list.append(edge)
    return edge_list


def edge_list_to_tensor(edge_list):
    edge_index = torch.LongTensor(edge_list).t()
    return edge_index


def sample_negative_edges(G, num_neg_samples):
    # 不用考虑num_neg_samples比所有不存在边的数量还高的边界条件
    # 不考虑自环
    # 注意，本来需要考虑逆边的问题，但是由于我采用的nx.non_edges函数不会出现两次重复节点对，所以不用考虑这个问题。

    neg_edge_list = []

    # 得到图中所有不存在的边（这个函数只会返回一侧，不会出现逆边）
    non_edges_one_side = list(enumerate(nx.non_edges(G)))
    neg_edge_list_indices = random.sample(range(0, len(non_edges_one_side)), num_neg_samples)
    # 取样num_neg_samples长度的索引
    for i in neg_edge_list_indices:
        neg_edge_list.append(non_edges_one_side[i][1])

    return neg_edge_list


def print_neg_edge():
    neg_edge_list = sample_negative_edges(G, 300062)
    l = []
    for i in neg_edge_list:
        l.append(i[0])
        l.append(i[1])
        s = str(l[0]) + ' ' + str(l[1]) + ' ' + '0'
        print(s)


def wite_neg_edge_to_File(target_file, file_name):
    file = open(target_file, "w")
    node_dic = create_dic(file_name)
    with open(file_name) as f:
        lines = f.readlines()
        for (k, v) in node_dic.items():
            repeat_count = 0
            for line in lines:
                l = line.strip().split(' ')
                if l[1] not in v:
                    s = '5' + ' ' + k + ' ' + l[1] + ' ' + '0' + '\n'
                    # print(s)
                    repeat_count += 1
                    file.write(s)
                if repeat_count == 4:
                    break
    f.close()
    file.close()


def create_dic(file_name):
    node_dic = dict()
    with open(file_name) as f:
        lines = f.readlines()
        target: List[str] = []
        for line in lines:
            l = line.strip().split(" ")
            if l[0] in node_dic.keys():
                target.append(l[1])
                node_dic[l[0]] = target
            else:
                target = [l[1]]
                node_dic[l[0]] = l[1]
        print(node_dic)
        print(len(node_dic.keys()))
    f.close()
    return node_dic


def decrease(file_name):
    file = open("E:\keyan\GATNE\data\ml\\test2.txt", "w")
    with open(file_name) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if i >= 200000:
                break
            if i % 4 == 0:
                file.write(line)
            i += 4
    f.close()
    file.close()


def split_to_test_and_valid():
    file = open("E:\keyan\GATNE\data\ml\\test_valid.txt", "w")
    with open("E:\keyan\GATNE\data\ml\\ratings.txt") as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if i >= 1000000:
                break
            if i % 12 == 0:
                file.write(line)
            i += 12
    f.close()
    file.close()

# split_to_test_and_valid()
# decrease("E:\keyan\GATNE\data\ml\\test.txt")
wite_neg_edge_to_File("E:\keyan\GATNE\data\ml\\5_neg_edge.txt", "E:\keyan\GATNE\data\ml\\5_edge.txt")
# G = load_graph("E:\keyan\GATNE\data\ml\\1_edge.txt")
# pos_edge_list = graph_to_edge_list(G)
# print_neg_edge()
# print(pos_edge_list)
# pos_edge_index = edge_list_to_tensor(pos_edge_list)

# print(neg_edge_list)
