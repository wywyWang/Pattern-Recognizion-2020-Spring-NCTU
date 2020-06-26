import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_circles, make_moons


def clustering(data, method):
    clusters = {}
    array = [_ for _ in range(data.shape[0])]
    clusters[0] = array.copy()
    distances = pairwise_distances(data, metric='euclidean')
    np.fill_diagonal(distances,sys.maxsize)
    min_row = -1
    min_col = -1
    for iteration in range(1, distances.shape[0]):
        min_dist = sys.maxsize
        min_row = -1
        min_col = -1
        for row in range(distances.shape[0]):
            for col in range(distances.shape[1]):
                if distances[row][col] <= min_dist:
                    min_dist = distances[row][col]
                    min_row = row
                    min_col = col
        
        if method == 'single':
            for update_col in range(distances.shape[0]):
                if update_col != min_col:
                    update_dist = min(distances[min_col][update_col], distances[min_row][update_col])
                    distances[min_col][update_col] = update_dist
                    distances[update_col][min_col] = update_dist
        elif method == 'complete':
            for update_col in range(distances.shape[0]):
                if update_col != min_col:
                    update_dist = max(distances[min_col][update_col], distances[min_row][update_col])
                    distances[min_col][update_col] = update_dist
                    distances[update_col][min_col] = update_dist
        elif method == 'average':
            for update_col in range(distances.shape[0]):
                if update_col != min_col:
                    update_dist = (distances[min_col][update_col] + distances[min_row][update_col]) / 2
                    distances[min_col][update_col] = update_dist
                    distances[update_col][min_col] = update_dist           

        for update_idx in range(distances.shape[0]):
            distances[min_row][update_idx] = sys.maxsize
            distances[update_idx][min_row] = sys.maxsize

        maximun = max(array[min_row], array[min_col])
        minimun = min(array[min_row], array[min_col])
        for idx in range(len(array)):
            if array[idx] == maximun:
                array[idx] = minimun
        clusters[iteration] = array.copy()
        # print()
        # print(min_row, '\t', min_col)
        # print("Iter: {}, clusters: {}".format(iteration, clusters[iteration]))
    return clusters


def visualization(data, clusters, link_type, data_type):
    # color = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
    # color = cm.rainbow(np.linspace(0, 1, len(data)))
    color = [np.random.rand(3,) for _ in range(len(data))]

    if data_type == 'circles' or data_type == 'moons':
        iterations = [_ for _ in range(0, len(clusters), 10)]
        iterations.append(len(clusters) - 2)
        iterations.append(len(clusters) - 1)
        fig = plt.figure()
        for selected_iteration in iterations:
            #plotting the clusters
            fig.suptitle('Scatter Plot for {} link clusters in iteration {}'.format(link_type, selected_iteration))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            for data_idx in range(len(clusters[selected_iteration])):
                selected_data = clusters[selected_iteration][data_idx]
                ax.scatter(data[data_idx, 0], data[data_idx, 1], color=color[selected_data])
            plt.savefig('./{}_{}_link/{}.png'.format(data_type, link_type, selected_iteration))
            plt.clf()
    else:
        for selected_iteration in range(len(clusters)):
            #plotting the clusters
            fig = plt.figure()
            fig.suptitle('Scatter Plot for {} link clusters in iteration {}'.format(link_type, selected_iteration))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            for data_idx in range(len(clusters[selected_iteration])):
                selected_data = clusters[selected_iteration][data_idx]
                ax.scatter(data[data_idx, 0], data[data_idx, 1], color=color[selected_data])
            plt.savefig('./{}_{}_link/{}.png'.format(data_type, link_type, selected_iteration))
            plt.clf()


if __name__ == "__main__":
    # # Data 1
    # data_type = 'simple'
    # data = np.array([0.40, 0.53, 0.22, 0.38, 0.35, 0.32, 0.26, 0.19, 0.08, 0.41, 0.45, 0.30]).reshape(6,2)
    # link_type = 'single'
    # clusters = clustering(data, link_type)
    # visualization(data, clusters, link_type, data_type)

    # link_type = 'complete'
    # clusters = clustering(data, link_type)
    # visualization(data, clusters, link_type, data_type)

    # link_type = 'average'
    # clusters = clustering(data, link_type)
    # visualization(data, clusters, link_type, data_type)


    # # Data 2
    # data_type = 'circles'
    # data = make_circles(n_samples=100, factor=0.5)
    # link_type = 'single'
    # clusters = clustering(data[0], link_type)
    # visualization(data[0], clusters, link_type, data_type)

    # link_type = 'complete'
    # clusters = clustering(data[0], link_type)
    # visualization(data[0], clusters, link_type, data_type)

    # link_type = 'average'
    # clusters = clustering(data[0], link_type)
    # visualization(data[0], clusters, link_type, data_type)

    # Data 3
    data_type = 'moons'
    data = make_moons(n_samples=100)
    link_type = 'single'
    clusters = clustering(data[0], link_type)
    visualization(data[0], clusters, link_type, data_type)

    link_type = 'complete'
    clusters = clustering(data[0], link_type)
    visualization(data[0], clusters, link_type, data_type)

    link_type = 'average'
    clusters = clustering(data[0], link_type)
    visualization(data[0], clusters, link_type, data_type)