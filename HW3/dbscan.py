import numpy as np
import pandas as pd
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics.cluster import adjusted_rand_score


RADIUS = 0.3
MinPts = 3


def DBScan(data):
    labels = [0] * len(data)
    cluster_id = 0

    for data_idx in range(len(data)):
        if labels[data_idx] != 0:
            continue
        NeighborPts = region_query(data, data_idx)
        # number of neighbors < MinPts, label as noise
        if len(NeighborPts) < MinPts:
            labels[data_idx] = -1
        else:
            cluster_id += 1
            merge_clusters(data, labels, NeighborPts, data_idx, cluster_id)
    return labels


def region_query(data, candidate_idx):
    neighbors = []
    for data_idx in range(len(data)):
        if np.linalg.norm(data[candidate_idx] - data[data_idx]) < RADIUS:
            neighbors.append(data_idx)
    return neighbors


def merge_clusters(data, labels, NeighborPts, candidate_idx, cluster_id):
    labels[candidate_idx] = cluster_id
    neighbor_idx = 0
    while neighbor_idx < len(NeighborPts):
        point = NeighborPts[neighbor_idx]
        if labels[point] == -1:
            labels[point] = cluster_id
        elif labels[point] == 0:
            labels[point] = cluster_id
            point_NeighborPts = region_query(data, point)

            if len(point_NeighborPts) >= MinPts:
                NeighborPts += point_NeighborPts
        neighbor_idx += 1


def visualization(data, clusters, data_type):
    color = [np.random.rand(3,) for _ in range(len(data))]
    fig = plt.figure()
    #plotting the clusters
    fig.suptitle('{} scatter Plot for dbscan clusters w/ radius {} MinPts {}'.format(data_type, RADIUS, MinPts))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for data_idx in range(len(clusters)):
        selected_data = clusters[data_idx]
        ax.scatter(data[data_idx, 0], data[data_idx, 1], color=color[selected_data])
    plt.savefig('./{}_dbscan/{}.png'.format(data_type, data_type))
    plt.clf()


if __name__ == "__main__":
    NOISE = 0.06
    # Data 2
    data_type = 'circles'
    data = make_circles(n_samples=100, factor=0.5, noise=NOISE)
    before = time.time()
    clusters = DBScan(data[0])
    print("cluster time {}".format(time.time() - before))
    visualization(data[0], clusters, data_type)
    ARI = adjusted_rand_score(clusters, data[1])
    print("{}'s ARI of dbscan is: {}".format(data_type, ARI))


    # Data 2
    data_type = 'moons'
    data = make_moons(n_samples=100, noise=NOISE)
    before = time.time()
    clusters = DBScan(data[0])
    print("cluster time {}".format(time.time() - before))
    visualization(data[0], clusters, data_type)
    ARI = adjusted_rand_score(clusters, data[1])
    print("{}'s ARI of dbscan is: {}".format(data_type, ARI))