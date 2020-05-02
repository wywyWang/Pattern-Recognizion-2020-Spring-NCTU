import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svmutil import *


PROPORTIONAL = 0.7
CLASS_NUM = 2


def readData():
    # Three classes, only select two classes
    iris = pd.read_csv('source/iris.data', header=None).reset_index(drop=True)
    iris = iris[iris[4] != 'Iris-virginica'].reset_index(drop=True)
    iris[4] = iris[4].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1
    })
    iris_train, iris_test = splitTrainTest(iris, 4)
    iris_train_x = iris_train.iloc[:, 0:4].copy()
    iris_train_y = iris_train.iloc[:, 4:].copy()
    iris_test_x = iris_test.iloc[:, 0:4].copy()
    iris_test_y = iris_test.iloc[:, 4:].copy()

    wine = pd.read_csv('source/wine.data', header=None).reset_index(drop=True)
    wine = wine[wine[0] != 3].reset_index(drop=True)
    wine[0] = wine[0].apply(lambda x: x-1)
    wine_train, wine_test = splitTrainTest(wine, 0)
    wine_train_x = wine_train.iloc[:, 1:].copy()
    wine_train_y = wine_train.iloc[:, 0:1].copy()
    wine_test_x = wine_test.iloc[:, 1:].copy()
    wine_test_y = wine_test.iloc[:, 0:1].copy()

    # Two classes
    breast = pd.read_csv('source/wdbc.data', header=None).reset_index(drop=True)
    breast[1] = breast[1].map({
        'B': 0,
        'M': 1
    })
    breast_train, breast_test = splitTrainTest(breast, 1)
    breast_train_x = breast_train.iloc[:, 2:11].copy()
    breast_train_y = breast_train.iloc[:, 1:2].copy()
    breast_test_x = breast_test.iloc[:, 2:11].copy()
    breast_test_y = breast_test.iloc[:, 1:2].copy()

    ionosphere = pd.read_csv('source/ionosphere.data', header=None).reset_index(drop=True)
    ionosphere[34] = ionosphere[34].map({
        'g': 1,
        'b': 0
    })
    ionosphere_train, ionosphere_test = splitTrainTest(ionosphere, 34)
    ionosphere_train_x = ionosphere_train.iloc[:, 0:34].copy()
    ionosphere_train_y = ionosphere_train.iloc[:, 34:].copy()
    ionosphere_test_x = ionosphere_test.iloc[:, 0:34].copy()
    ionosphere_test_y = ionosphere_test.iloc[:, 34:].copy()
    return iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
           ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
           breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
           wine_train_x, wine_train_y, wine_test_x, wine_test_y


def read_csv():
    with open('./data/X_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_train = list(csv_reader)
        x_train = [[float(y) for y in x] for x in x_train]

    with open('./data/Y_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        y_train_2d = list(csv_reader)
        y_train = [y for x in y_train_2d for y in x]
        y_train = [ int(x) for x in y_train ]

    with open('./data/X_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_test = list(csv_reader)
        x_test = [[float(y) for y in x] for x in x_test]

    with open('./data/Y_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        y_test_2d = list(csv_reader)
        y_test = [y for x in y_test_2d for y in x]
        y_test = [ int(x) for x in y_test ]
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


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


if __name__ == '__main__':
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()
    # # Another data
    # x_train, y_train, x_test, y_test = read_csv()