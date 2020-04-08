import numpy as np
import matplotlib.pyplot as plt
import utils

def computeGaussian(value, mean, var):
    """This function is compute log of Gaussian distribution, because original multiply values will become smaller and smaller."""
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
    """This function is training stage of naive-bayes classifier."""
    # print("train x shape : {}".format(train_x.shape))
    # print("train y shape : {}".format(train_y.shape))
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


def test(test_x, test_y, prior, train_mean, train_var, class_num, filename, testing=False):
    """This function is testing stage of naive-bayes classifier."""
    # print("test_x shape : {}".format(test_x.shape))
    # print("test_y shape : {}".format(test_y.shape))
    error = 0
    predict_list = []
    total_probability = [0 for _ in range(test_x.shape[0])]
    for data_idx in range(test_x.shape[0]):
        probability = np.zeros((class_num), dtype = float)
        for label in range(class_num):
            probability[label] += prior[label]
            for feature_idx in range(test_x.shape[1]):
                predict_value = computeGaussian(test_x[data_idx][feature_idx], train_mean[label][feature_idx], train_var[label][feature_idx])
                probability[label] += predict_value
        probability = normalization(probability, class_num)
        total_probability[data_idx] = probability[1]
        prediction = np.argmin(probability)
        predict_list.append(prediction)
        error += checkResult(prediction, test_y[data_idx])
    accuracy = (test_x.shape[0] - error) / test_x.shape[0]
    print("Error : {}. Accuracy {}".format(error, accuracy))

    #Plot ROC curve
    FA_PD = []
    if class_num == 2:
        slices = 20
        low = min(total_probability)
        high = max(total_probability)
        step = (abs(low) + abs(high)) / slices
        thresholds = np.arange(low-step, high+step, step)
        for threshold in thresholds:
            FA_PD.append(utils.computeConfusionMatrix(total_probability, test_y, class_num, threshold))
        FA = [row[0] for row in FA_PD]
        PD = [row[1] for row in FA_PD]
        FA_x = np.linspace(0.0, 1.0, slices)
        PD_interp = np.interp(FA_x, FA, PD)
        # print("FA : {}".format(FA))
        # print("PD : {}".format(PD))
        # print("FA : {}".format(FA[::-1]))
        # print("PD : {}".format(PD[::-1]))
        # print(PD_interp)

        if testing is True:
            # Plot ROC curve of testing data
            fig = plt.figure()
            plt.plot(FA_x, PD_interp)
            plt.xlabel('FA')
            plt.ylabel('PD')
            fig.savefig('plotting/' + filename + '_roc_test.png')
        return accuracy, FA_x, PD_interp
    else:
        utils.computeConfusionMatrix(total_probability, test_y, class_num)
        return accuracy