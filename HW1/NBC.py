import numpy as np

def computeGaussian(value, mean, var):
    return np.log(1.0 / (np.sqrt(2.0 * np.pi * var))) - ((value - mean) ** 2.0 / (2.0 * var))


def checkResult(probability, answer):
    prediction = np.argmin(probability)
    # print("Prediction: ", prediction, ", Ans: ", answer)
    if prediction == answer:
        return 0
    else:
        return 1


def normalization(probability, class_num):
    temp = 0
    for j in range(class_num):
        temp += probability[j]
    for j in range(class_num):
        probability[j] /= temp
    return probability


def train(train_x, train_y, class_num):
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
                print("HI")
                train_var[label][feature_idx] = 1e-4
    prior = prior / train_x.shape[0]
    prior = np.log(prior)
    return prior, train_mean, train_var


def test(test_x, test_y, prior, train_mean, train_var, class_num):
    print("test_x shape : ".format(test_x.shape))
    print("test_y shape : ".format(test_y.shape))
    error = 0
    for data_idx in range(test_x.shape[0]):
        probability = np.zeros((class_num), dtype = float)
        for label in range(class_num):
            probability[label] += prior[label]
            for feature_idx in range(test_x.shape[1]):
                predict_value = computeGaussian(test_x[data_idx][feature_idx], train_mean[label][feature_idx], train_var[label][feature_idx])
                probability[label] += predict_value
        probability = normalization(probability, class_num)
        error += checkResult(probability, test_y[data_idx])
        # print(probability)
    print("Error : {}. Accuracy {}".format(error, (test_x.shape[0] - error) / test_x.shape[0]))