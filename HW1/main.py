import numpy as np
import pandas as pd
import NBC
import LC

def readData():
    # Three classes
    iris = pd.read_csv('source/iris.data', header=None)
    iris_train, iris_test = splitTrainTest(iris)
    iris_train_x = iris_train.iloc[:, 0:4].copy()
    iris_train_y = iris_train.iloc[:, 4:].copy()[4].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    iris_test_x = iris_test.iloc[:, 0:4].copy()
    iris_test_y = iris_test.iloc[:, 4:].copy()[4].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })

    wine = pd.read_csv('source/wine.data', header=None)
    wine_train, wine_test = splitTrainTest(wine)
    wine_train_x = wine_train.iloc[:, 1:].copy()
    wine_train_y = wine_train.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)
    wine_test_x = wine_test.iloc[:, 1:].copy()
    wine_test_y = wine_test.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)

    # Two classes
    breast = pd.read_csv('source/wdbc.data', header=None)
    breast_train, breast_test = splitTrainTest(breast)
    breast_train_x = breast_train.iloc[:, 2:12].copy()
    breast_train_y = breast_train.iloc[:, 1:2].copy()[1].map({
        'B': 0,
        'M': 1
    })
    breast_test_x = breast_test.iloc[:, 2:12].copy()
    breast_test_y = breast_test.iloc[:, 1:2].copy()[1].map({
        'B': 0,
        'M': 1
    })

    ionosphere = pd.read_csv('source/ionosphere.data', header=None)
    ionosphere_train, ionosphere_test = splitTrainTest(ionosphere)
    ionosphere_train_x = ionosphere_train.iloc[:, 0:34].copy()
    ionosphere_train_y = ionosphere_train.iloc[:, 34:].copy()[34].map({
        'g': 1,
        'b': 0
    })
    ionosphere_test_x = ionosphere_test.iloc[:, 0:34].copy()
    ionosphere_test_y = ionosphere_test.iloc[:, 34:].copy()[34].map({
        'g': 1,
        'b': 0
    })

    return iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
           ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
           breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
           wine_train_x, wine_train_y, wine_test_x, wine_test_y


def splitTrainTest(data):
    mask = np.random.rand(len(data)) < 0.7
    train = data[mask].reset_index(drop=True)
    test = data[~mask].reset_index(drop=True)
    print("# of training data : {}".format(len(train)))
    print("# of testing data : {}".format(len(test)))
    return train, test


if __name__ == "__main__":
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()
    # Run Naive-Bayes Classifier
    prior, train_mean, train_var = NBC.train(iris_train_x.values, iris_train_y.values, 3)
    NBC.test(iris_test_x.values, iris_test_y.values, prior, train_mean, train_var, 3)

    prior, train_mean, train_var = NBC.train(wine_train_x.values, wine_train_y.values, 3)
    NBC.test(wine_test_x.values, wine_test_y.values, prior, train_mean, train_var, 3)

    prior, train_mean, train_var = NBC.train(ionosphere_train_x.values, ionosphere_train_y.values, 2)
    NBC.test(ionosphere_test_x.values, ionosphere_test_y.values, prior, train_mean, train_var, 2)

    prior, train_mean, train_var = NBC.train(breast_train_x.values, breast_train_y.values, 2)
    NBC.test(breast_test_x.values, breast_test_y.values, prior, train_mean, train_var, 2)