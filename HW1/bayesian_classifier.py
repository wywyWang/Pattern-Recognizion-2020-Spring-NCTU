import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import utils

def computeMultivariateGaussian(X, mean, cov):
    """This function is compute log of Gaussian distribution, because original multiply values will become smaller and smaller."""
    det = np.linalg.det(cov)
    norm_const = np.log(1.0 / (math.pow((2 * np.pi), float(len(X)) / 2) * math.pow(det, 1.0 / 2)))
    x_mu = np.matrix(X - mean)
    inv = np.linalg.inv(cov)
    result = -0.5 * (x_mu * inv * x_mu.T)
    return norm_const * result


def checkResult(prediction, answer):
    # print("Prediction: ", prediction, ", Ans: ", answer)
    if prediction == answer:
        return 0
    else:
        return 1


def normalization(probability, class_num):
    """This function is used to normalize the probability."""
    temp = 0
    for j in range(class_num):
        temp += probability[j]
    for j in range(class_num):
        probability[j] /= temp
    return probability


def train(train_x, train_y, class_num):
    """This function is training stage of bayesian classifier."""
    # print("train x shape : {}".format(train_x.shape))
    # print("train y shape : {}".format(train_y.shape))
    prior = []
    mean = []
    cov = []
    for class_idx in range(class_num):
        match_idx = np.where(train_y == class_idx)[0]
        train_x_class = train_x[match_idx].copy()
        class_prior = np.log(len(train_x_class) / len(train_x))
        class_mean = np.mean(train_x_class, axis=0)
        class_cov = np.cov(train_x_class.T)
        prior.append(class_prior)
        mean.append(class_mean)
        cov.append(class_cov)
    prior = np.array(prior)
    mean = np.array(mean)
    cov = np.array(cov)
    # psuedo count for covariance
    for class_idx in range(class_num):
        if np.linalg.det(cov[class_idx]) == 0:
            np.fill_diagonal(cov[class_idx], 1)
    # print(prior)
    # print(mean)
    # print(cov)
    return prior, mean, cov


def test(test_x, test_y, prior, train_mean, train_cov, class_num, filename, model_name, testing=False):
    """This function is testing stage of bayesian classifier."""
    error = 0
    total_probability = [0 for _ in range(test_x.shape[0])]
    multi_class = []
    for data_idx in range(test_x.shape[0]):
        probability = np.zeros((class_num), dtype = float)
        for class_idx in range(class_num):
            probability[class_idx] += prior[class_idx]
            likelihood = computeMultivariateGaussian(test_x[data_idx], train_mean[class_idx], train_cov[class_idx])
            probability[class_idx] += likelihood
        probability = normalization(probability, class_num)
        total_probability[data_idx] = probability[1]            # Select class 1
        multi_class.append(probability)
        prediction = np.argmin(probability)
        error += checkResult(prediction, test_y[data_idx])
    accuracy = (test_x.shape[0] - error) / test_x.shape[0]
    print("Error : {}. Accuracy {}".format(error, accuracy))

    #Plot ROC curve
    FA_PD = []
    if class_num == 2:
        slices = 50
        low = min(total_probability)
        high = max(total_probability)
        step = (abs(low) + abs(high)) / slices
        thresholds = np.arange(low-step, high+step, step)
        for threshold in thresholds:
            FA_PD.append(utils.computeConfusionMatrix(total_probability, test_y, class_num, model_name, threshold))
        FA = [row[0] for row in FA_PD]
        PD = [row[1] for row in FA_PD]
        FA_x = np.linspace(0.0, 1.0, slices)
        PD_interp = np.interp(FA_x, FA, PD)
        # Plot ROC curve of testing data
        if testing is True:
            fig = plt.figure()
            plt.plot(FA_x, PD_interp)
            plt.xlabel('FA')
            plt.ylabel('PD')
            fig.savefig('plotting/' + filename + '_' + model_name + '_roc_testing.png')
        return accuracy, FA_x, PD_interp
    else:
        utils.computeConfusionMatrix(multi_class, test_y, class_num, model_name)
        return accuracy