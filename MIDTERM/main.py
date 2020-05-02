import csv
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svmutil import *


PROPORTIONAL = 0.9
CLASS_NUM = 2
K = 5
KERNEL = 0


def readData():
    # Three classes, only select two classes
    iris = pd.read_csv('source/iris.data', header=None).reset_index(drop=True)
    iris = iris[iris[4] != 'Iris-virginica'].reset_index(drop=True)
    iris[4] = iris[4].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1
    })
    iris_train, iris_test = splitTrainTest(iris, 4)
    iris_train_x = iris_train.iloc[:, 0:2].copy()
    iris_train_y = iris_train.iloc[:, 4:].copy()
    iris_test_x = iris_test.iloc[:, 0:2].copy()
    iris_test_y = iris_test.iloc[:, 4:].copy()

    wine = pd.read_csv('source/wine.data', header=None).reset_index(drop=True)
    wine = wine[wine[0] != 3].reset_index(drop=True)
    wine[0] = wine[0].apply(lambda x: x-1)
    wine_train, wine_test = splitTrainTest(wine, 0)
    wine_train_x = wine_train.iloc[:, 1:3].copy()
    wine_train_y = wine_train.iloc[:, 0:1].copy()
    wine_test_x = wine_test.iloc[:, 1:3].copy()
    wine_test_y = wine_test.iloc[:, 0:1].copy()

    # Two classes
    breast = pd.read_csv('source/wdbc.data', header=None).reset_index(drop=True)
    breast[1] = breast[1].map({
        'B': 0,
        'M': 1
    })
    breast_train, breast_test = splitTrainTest(breast, 1)
    breast_train_x = breast_train.iloc[:, 2:4].copy()
    breast_train_y = breast_train.iloc[:, 1:2].copy()
    breast_test_x = breast_test.iloc[:, 2:4].copy()
    breast_test_y = breast_test.iloc[:, 1:2].copy()

    ionosphere = pd.read_csv('source/ionosphere.data', header=None).reset_index(drop=True)
    ionosphere[34] = ionosphere[34].map({
        'g': 1,
        'b': 0
    })
    ionosphere_train, ionosphere_test = splitTrainTest(ionosphere, 34)
    ionosphere_train_x = ionosphere_train.iloc[:, 0:2].copy()
    ionosphere_train_y = ionosphere_train.iloc[:, 34:].copy()
    ionosphere_test_x = ionosphere_test.iloc[:, 0:2].copy()
    ionosphere_test_y = ionosphere_test.iloc[:, 34:].copy()
    return iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
           ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
           breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
           wine_train_x, wine_train_y, wine_test_x, wine_test_y


def splitTrainTest(data, class_idx):
    """Split data into training set and testing set"""
    train = None
    test = None
    for selected_class in range(CLASS_NUM):
        class_data = data[data[class_idx] == selected_class]
        class_train = class_data.sample(frac=PROPORTIONAL)
        class_test = class_data[~class_data.isin(class_train)].dropna()
        train = pd.concat([train, class_train.reset_index(drop=True)], axis=0, ignore_index=True)
        test = pd.concat([test, class_test.reset_index(drop=True)], axis=0, ignore_index=True)
    print()
    print("# of training data : {}".format(len(train)))
    print("# of testing data : {}".format(len(test)))
    print("# of training each class : {}".format(train[class_idx].value_counts()))
    print("# of testing each class : {}".format(test[class_idx].value_counts()))
    return train, test


def GridSearch(x_train, y_train):
    C = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
    G = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]
    best_pair = (0, 0)
    best_acc = 0
    # K-fold cross validation
    for cost in C:
        for gamma in G:
            # Parameters explanation
            # https://gist.github.com/josephmisiti/7572696
            prob  = svm_problem(y_train, x_train)
            param = svm_parameter('-s 0 -t {} -v {} -c {} -g {} -q'.format(KERNEL, K, cost, gamma))          # 2 : rbf kernel
            acc = svm_train(prob, param)
            if acc > best_acc:
                best_acc = acc
                best_pair = (cost, gamma)
    return best_pair, best_acc


def plotROC(total_probability, test_y, filename):
    slices = 50
    FA_PD = []
    low = min(total_probability)
    high = max(total_probability)
    step = (abs(low) + abs(high)) / slices
    thresholds = np.arange(low-step, high+step, step)
    for threshold in thresholds:
        FA_PD.append(computeConfusionMatrix(total_probability, test_y, threshold))
    FA = [row[0] for row in FA_PD]
    PD = [row[1] for row in FA_PD]
    FA_x = np.linspace(0.0, 1.0, slices)
    PD_interp = np.interp(FA_x, FA, PD)
    # Plot ROC curve of testing data
    fig = plt.figure()
    plt.plot(FA_x, PD_interp)
    plt.xlabel('FA')
    plt.ylabel('PD')
    fig.savefig('plotting/' + filename + '_svm_roc_' + str(KERNEL) + '_testing.png')
    return FA_x, PD_interp


def computeConfusionMatrix(probability, answer, threshold=None):
    """Compute confusion matrix based on predicted class and actual class."""
    confusion_matrix = np.zeros((CLASS_NUM, CLASS_NUM))
    for data_idx in range(len(probability)):
        if threshold is None:
            if probability[data_idx] <= 0:
                predict = 1
            else:
                predict = 0
        else:
            if probability[data_idx] <= threshold:
                predict = 1
            else:
                predict = 0
        confusion_matrix[predict][int(answer[data_idx])] += 1
    PD = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]) if (confusion_matrix[0][1] + confusion_matrix[1][1]) != 0 else 0
    FA = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0]) if (confusion_matrix[1][0] + confusion_matrix[0][0]) != 0 else 0
    if threshold is None:
        print("Confusion matrix:")
        print(confusion_matrix)
    return [FA, PD]


def TestBest(x_train, y_train, x_test, y_test, best_pair):
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-s 0 -t {} -c {} -g {} '.format(KERNEL, best_pair[0], best_pair[1]))          # 2 : rbf kernel
    model = svm_train(prob, param)
    prediction = svm_predict(y_test, x_test, model)
    probability = []
    for i in range(len(prediction[2])):
        probability.append(prediction[2][i][0])
    probability = np.array(probability)
    # w = model.obj
    # b = -model.rho
    # if (model.Label(1) == -1):
    #     w = -w; b = -b
    # print("w = {}".format(w))
    # print("b = {}".format(b))
    return prediction[0], probability


if __name__ == '__main__':
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()

    print("====================== IRIS =================")
    before = time.time()
    best_pair, best_acc = GridSearch(iris_train_x.values, iris_train_y.values.ravel())
    print("Grid search time : {}".format(time.time() - before))
    print("Best pair  = {}".format(best_pair))
    print("Best acc = {}".format(best_acc))
    before = time.time()
    predict, probability = TestBest(iris_train_x.values, iris_train_y.values.ravel(), iris_test_x.values, iris_test_y.values.ravel(), best_pair)
    print("Testing time : {}".format(time.time() - before))
    computeConfusionMatrix(probability, iris_test_y.values.ravel())
    plotROC(probability, iris_test_y.values.ravel(), 'iris')

    print("=================== BREAST ==============")
    before = time.time()
    best_pair, best_acc = GridSearch(breast_train_x.values, breast_train_y.values.ravel())
    print("Grid search time : {}".format(time.time() - before))
    print("Best pair  = {}".format(best_pair))
    print("Best acc = {}".format(best_acc))
    before = time.time()
    predict, probability = TestBest(breast_train_x.values, breast_train_y.values.ravel(), breast_test_x.values, breast_test_y.values.ravel(), best_pair)
    print("Testing time : {}".format(time.time() - before))
    computeConfusionMatrix(probability, breast_test_y.values.ravel())
    plotROC(probability, breast_test_y.values.ravel(), 'breast')

    print("=================== IONOSPHERE ==============")
    before = time.time()
    best_pair, best_acc = GridSearch(ionosphere_train_x.values, ionosphere_train_y.values.ravel())
    print("Grid search time : {}".format(time.time() - before))
    print("Best pair  = {}".format(best_pair))
    print("Best acc = {}".format(best_acc))
    before = time.time()
    predict, probability = TestBest(ionosphere_train_x.values, ionosphere_train_y.values.ravel(), ionosphere_test_x.values, ionosphere_test_y.values.ravel(), best_pair)
    print("Testing time : {}".format(time.time() - before))
    computeConfusionMatrix(probability, ionosphere_test_y.values.ravel())
    plotROC(probability, ionosphere_test_y.values.ravel(), 'ionosphere')