import numpy as np
import matplotlib.pyplot as plt
import utils

def compare(train_x, train_y, weight):
    probability = np.dot(train_x, weight)
    y_pred = np.ones((probability.shape[0], 1))
    negative_idx = np.where(probability < 0)[0]
    y_pred[negative_idx] = -1
    wrong_idx = np.where(y_pred != train_y)[0]
    return wrong_idx


def update(train_x, train_y, weight):
    """This function is used to update weight based on wrong classified data."""
    num = len(compare(train_x, train_y, weight))
    new_weight = weight + train_y[compare(train_x, train_y, weight)][np.random.choice(num)] * train_x[compare(train_x, train_y, weight),:][np.random.choice(num)].reshape(-1, 1)
    return new_weight


def train(train_x, train_y, class_num):
    """This function is training stage of pocket algorithm classifier."""
    ITERATIONS = 5000
    train_y[train_y == 0] = -1          # Replace class 0 to -1 in order to handle perceptron 
    train_y = train_y.reshape(-1, 1)
    print("train x shape : {}".format(train_x.shape))
    print("train y shape : {}".format(train_y.shape))
    augment = np.ones((train_x.shape[0], 1))
    train_x = np.concatenate((train_x, augment), axis=1)
    weight = train_x[0].copy()
    weight[-1] = 0        # w0 initial 1
    weight = weight.reshape(-1, 1)
    best_weight = weight.copy()
    best_wrong = len(compare(train_x, train_y, weight))

    for iteration in range(ITERATIONS):
        wrong_idx = compare(train_x, train_y, weight)
        weight = update(train_x, train_y, weight)
        # print("# of wrong points : {}".format(len(wrong_idx)))
        if len(wrong_idx) < best_wrong:
            best_wrong = len(wrong_idx)
            best_weight = weight.copy()
    print("min error : {}".format(best_wrong))
    return weight


def test(test_x, test_y, weight, class_num, filename, model_name, testing=False):
    """This function is testing stage of pocket algorithm classifier."""
    test_y[test_y == 0] = -1
    test_y = test_y.reshape(-1, 1)
    augment = np.ones((test_x.shape[0], 1))
    test_x = np.concatenate((test_x, augment), axis=1)
    probability = np.dot(test_x, weight)
    y_pred = np.ones((probability.shape[0], 1))
    negative_idx = np.where(probability < 0)[0]
    y_pred[negative_idx] = -1
    error = len(np.where(y_pred != test_y)[0])
    accuracy = (test_x.shape[0] - error) / test_x.shape[0]
    print(probability)

    #Plot ROC curve
    FA_PD = []
    slices = 50
    low = min(probability)
    high = max(probability)
    step = (abs(low) + abs(high)) / slices
    thresholds = np.arange(low-step, high+step, step)
    for threshold in thresholds:
        FA_PD.append(utils.computeConfusionMatrix(probability, test_y, class_num, model_name, threshold))
    FA = [row[0] for row in FA_PD]
    PD = [row[1] for row in FA_PD]
    FA_x = np.linspace(0.0, 1.0, slices)
    PD_interp = np.interp(FA_x, FA[::-1], PD[::-1])
    # Plot ROC curve of testing data
    if testing is True:
        fig = plt.figure()
        plt.plot(FA_x, PD_interp)
        plt.xlabel('FA')
        plt.ylabel('PD')
        fig.savefig('plotting/' + filename + '_' + model_name + '_roc_testing.png')
    print("Error : {}. Accuracy {}".format(error, accuracy))
    return accuracy, FA_x, PD_interp