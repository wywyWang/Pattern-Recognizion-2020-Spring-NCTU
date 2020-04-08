import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bayesian_classifier as BC
import naive_bayes_classifier as NBC
import pocket_classifier as PC

def readData():
    # Three classes
    iris = pd.read_csv('source/iris.data', header=None).sample(frac=1).reset_index(drop=True)
    # Plot feature realtions
    # iris_plot = iris.copy()
    # iris_plot.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    # sns_plot = sns.pairplot(iris_plot, hue='class', palette='husl', markers=['o', 's', 'D'])
    # sns_plot.savefig('plotting/iris_features.png')
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

    wine = pd.read_csv('source/wine.data', header=None).sample(frac=1).reset_index(drop=True)
    # Plot feature realtions
    wine_plot = wine.copy()
    # wine_plot.columns = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    # sns_plot = sns.pairplot(wine_plot, hue='class', palette='husl', markers=['o', 's', 'D'])
    # sns_plot.savefig('plotting/wine_features.png')
    wine_train, wine_test = splitTrainTest(wine)
    wine_train_x = wine_train.iloc[:, 1:].copy()
    wine_train_y = wine_train.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)
    wine_test_x = wine_test.iloc[:, 1:].copy()
    wine_test_y = wine_test.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)

    # Two classes
    breast = pd.read_csv('source/wdbc.data', header=None).sample(frac=1).reset_index(drop=True)
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

    ionosphere = pd.read_csv('source/ionosphere.data', header=None).sample(frac=1).reset_index(drop=True)
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
    """Split data into training set and testing set, proportional is 0.7."""
    mask = np.random.rand(len(data)) < 0.7
    train = data[mask].reset_index(drop=True)
    test = data[~mask].reset_index(drop=True)
    print("# of training data : {}".format(len(train)))
    print("# of testing data : {}".format(len(test)))
    return train, test


def crossValidation(train_x, train_y, class_num, filename, K=3):
    """Doing k-fold cross validation, default K is 3."""
    divided = int(len(train_x) / K)
    overall_acc = 0
    total_PD = []
    total_FA = []
    for fold in range(K):
        print()
        print("Now fold is {}".format(fold))
        # Compute start and end index
        start = divided * fold
        end = divided * (fold + 1)
        training_x = np.concatenate((train_x[:start], train_x[end:]))
        training_y = np.concatenate((train_y[:start], train_y[end:]))
        validation_x = train_x[start:end].values
        validation_y = train_y[start:end].values
        prior, train_mean, train_var = NBC.train(training_x, training_y, class_num)
        if class_num == 2:
            acc, FA, PD = NBC.test(validation_x, validation_y, prior, train_mean, train_var, class_num, filename, False)
            total_FA.append(np.array(FA))
            total_PD.append(np.array(PD))
        else:
            acc = NBC.test(validation_x, validation_y, prior, train_mean, train_var, class_num, filename, False)
        overall_acc += acc
    print("Overall accuracy: {}".format(overall_acc / K))
    # Plot ROC curve if it is binary classification.
    if class_num == 2:
        total_FA = np.array(total_FA)
        total_PD = np.array(total_PD)
        FA_mean = np.mean(total_FA, axis=0)
        PD_mean = np.mean(total_PD, axis=0)
        FA_var = np.var(total_FA, axis=0)
        PD_var = np.var(total_PD, axis=0)
        # Plot ROC curve of validation data
        fig = plt.figure()
        plt.errorbar(FA_mean, PD_mean, yerr=PD_var, uplims=True, lolims=True)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        plt.xlabel('FA')
        plt.ylabel('PD')
        fig.savefig('plotting/' + filename + '_roc_CV.png')


if __name__ == "__main__":
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()

    class_num = 3
    # Run Naive-Bayes Classifier
    # crossValidation(iris_train_x, iris_train_y, class_num, 'iris', 10)
    # prior, train_mean, train_var = NBC.train(iris_train_x.values, iris_train_y.values, class_num)
    # irirs_acc = NBC.test(iris_test_x.values, iris_test_y.values, prior, train_mean, train_var, class_num, 'iris', True)
    # prior, train_mean, train_var = BC.train(iris_train_x.values, iris_train_y.values, class_num)
    # irirs_acc = BC.test(iris_test_x.values, iris_test_y.values, prior, train_mean, train_var, class_num, 'iris', True)

    # crossValidation(wine_train_x, wine_train_y, class_num, 'wine', 10)
    # prior, train_mean, train_var = NBC.train(wine_train_x.values, wine_train_y.values, class_num)
    # NBC.test(wine_test_x.values, wine_test_y.values, prior, train_mean, train_var, class_num, 'wine', True)

    class_num = 2
    # crossValidation(ionosphere_train_x, ionosphere_train_y, class_num, 'ionosphere', 10)
    # prior, train_mean, train_var = NBC.train(ionosphere_train_x.values, ionosphere_train_y.values, class_num)
    # acc = NBC.test(ionosphere_test_x.values, ionosphere_test_y.values, prior, train_mean, train_var, class_num, 'ionosphere', True)

    # Run Naive-Bayes Classifier
    # crossValidation(breast_train_x, breast_train_y, class_num, 'breast', 10)
    # prior, train_mean, train_var = NBC.train(breast_train_x.values, breast_train_y.values, class_num)
    # acc = NBC.test(breast_test_x.values, breast_test_y.values, prior, train_mean, train_var, class_num, 'breast', True)
    train_weight = PC.train(breast_train_x.values, breast_train_y.values, class_num)
    acc = PC.test(breast_test_x.values, breast_test_y.values, train_weight, class_num, 'breast', True)