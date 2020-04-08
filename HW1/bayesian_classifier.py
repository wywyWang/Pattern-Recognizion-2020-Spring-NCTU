import numpy as np
import pandas as pd

def train(train_x, train_y, class_num):
    """This function is training stage of bayesian classifier."""
    print("train x shape : {}".format(train_x.shape))
    print("train y shape : {}".format(train_y.shape))
    prior = np.zeros((class_num), dtype = float)
    train_square = np.zeros((class_num, train_x.shape[1]), dtype = float)
    train_mean = np.zeros((class_num, train_x.shape[1]), dtype = float)
    train_var = np.zeros((class_num, train_x.shape[1]), dtype = float)

    for data_idx in range(train_x.shape[0]):
        label = train_y[data_idx]
        prior[label] += 1
        for feature_idx in range(train_x.shape[1]):
            train_square[label][feature_idx] += (train_x[data_idx][feature_idx] ** 2)
            train_mean[label][feature_idx] += train_x[data_idx][feature_idx]
    #Calculate mean and standard deviation
    for label in range(class_num):
        for feature_idx in range(train_x.shape[1]):
            train_mean[label][feature_idx] = float(train_mean[label][feature_idx] / prior[label])
            train_var[label][feature_idx] = float(train_square[label][feature_idx] / prior[label]) - float(train_mean[label][feature_idx] ** 2)
            # psuedo count for variance
            if train_var[label][feature_idx] == 0:
                train_var[label][feature_idx] = 1e-6
    prior = prior / train_x.shape[0]
    prior = np.log(prior)
    return prior, train_mean, train_var