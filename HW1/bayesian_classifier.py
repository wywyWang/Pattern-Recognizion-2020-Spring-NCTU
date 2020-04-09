import numpy as np
import pandas as pd

def computeGaussian(value, mean, cov):
    """This function is compute log of Gaussian distribution, because original multiply values will become smaller and smaller."""
    # norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
    # x_mu = matrix(x - mu)
    # inv = sigma.I
    # result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
    return np.log(1.0 / (np.sqrt(2.0 * np.pi * var))) - ((value - mean) ** 2.0 / (2.0 * var))


def checkResult(prediction, answer):
    # prediction = np.argmin(probability)
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
    print("train x shape : {}".format(train_x.shape))
    print("train y shape : {}".format(train_y.shape))

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
    print(prior)
    print(mean)
    print(cov)
    return prior, mean, cov


def test(test_x, test_y, prior, train_mean, train_var, class_num, filename, model_name, testing=False):
    """This function is testing stage of bayesian classifier."""
    error = 0
    predict_list = []
    total_probability = [0 for _ in range(test_x.shape[0])]
    multi_class = []
    for data_idx in range(test_x.shape[0]):
        probability = np.zeros((class_num), dtype = float)
        for label in range(class_num):
            probability[label] += prior[label]
            for feature_idx in range(test_x.shape[1]):
                predict_value = computeGaussian(test_x[data_idx][feature_idx], train_mean[label][feature_idx], train_var[label][feature_idx])
                probability[label] += predict_value
        probability = normalization(probability, class_num)
        total_probability[data_idx] = probability[1]
        multi_class.append(probability)
        prediction = np.argmin(probability)
        predict_list.append(prediction)
        error += checkResult(prediction, test_y[data_idx])
    accuracy = (test_x.shape[0] - error) / test_x.shape[0]
    print("Error : {}. Accuracy {}".format(error, accuracy))