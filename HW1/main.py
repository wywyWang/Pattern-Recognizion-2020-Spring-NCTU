import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bayesian_classifier as BC
import naive_bayes_classifier as NBC
import pocket_classifier as PC

def readData():
    # Three classes
    iris = pd.read_csv('source/iris.data', header=None).sample(frac=1).reset_index(drop=True)
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
    wine_train, wine_test = splitTrainTest(wine)
    wine_train_x = wine_train.iloc[:, 1:].copy()
    wine_train_y = wine_train.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)
    wine_test_x = wine_test.iloc[:, 1:].copy()
    wine_test_y = wine_test.iloc[:, 0:1].copy()[0].apply(lambda x: x-1)

    # Two classes
    breast = pd.read_csv('source/wdbc.data', header=None).sample(frac=1).reset_index(drop=True)
    breast_train, breast_test = splitTrainTest(breast)
    breast_train_x = breast_train.iloc[:, 2:11].copy()
    breast_train_y = breast_train.iloc[:, 1:2].copy()[1].map({
        'B': 0,
        'M': 1
    })
    breast_test_x = breast_test.iloc[:, 2:11].copy()
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


def crossValidation(train_x, train_y, class_num, filename, model_name, K=3):
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
        #Choose model
        if model_name == 'NBC':
            prior, train_mean, train_var = NBC.train(training_x, training_y, class_num)
            if class_num == 2:
                acc, FA, PD = NBC.test(validation_x, validation_y, prior, train_mean, train_var, class_num, filename, model_name, False)
                total_FA.append(np.array(FA))
                total_PD.append(np.array(PD))
            else:
                acc = NBC.test(validation_x, validation_y, prior, train_mean, train_var, class_num, filename, model_name, False)
        elif model_name == 'PC':
            train_weight = PC.train(training_x, training_y, class_num)
            acc, FA, PD = PC.test(validation_x, validation_y, train_weight, class_num, filename, model_name, False)
            total_FA.append(np.array(FA))
            total_PD.append(np.array(PD))
        elif model_name == 'BC':
            prior, train_mean, train_cov = BC.train(training_x, training_y, class_num)
            if class_num == 2:
                acc, FA, PD = BC.test(validation_x, validation_y, prior, train_mean, train_cov, class_num, filename, model_name, False)
                total_FA.append(np.array(FA))
                total_PD.append(np.array(PD))
            else:
                acc = BC.test(validation_x, validation_y, prior, train_mean, train_cov, class_num, filename, model_name, False)
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
        fig.savefig('plotting/' + filename + '_' + model_name + '_roc_CV.png')


if __name__ == "__main__":
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()


    K = 5
    class_num = 3
    print("====================== IRIS =================")
    # Bayesian classifier
    crossValidation(iris_train_x, iris_train_y, class_num, 'iris', 'BC', K)
    prior, train_mean, train_cov = BC.train(iris_train_x.values, iris_train_y.values, class_num)
    acc = BC.test(iris_test_x.values, iris_test_y.values, prior, train_mean, train_cov, class_num, 'iris', 'BC', True)
    # Naive-Bayes classifier
    crossValidation(iris_train_x, iris_train_y, class_num, 'iris', 'NBC', K)
    prior, train_mean, train_var = NBC.train(iris_train_x.values, iris_train_y.values, class_num)
    acc = NBC.test(iris_test_x.values, iris_test_y.values, prior, train_mean, train_var, class_num, 'iris', 'NBC', True)
    
    print("====================== WINE =================")
    # Bayesian classifier
    crossValidation(wine_train_x, wine_train_y, class_num, 'wine', 'BC', K)
    prior, train_mean, train_cov = BC.train(wine_train_x.values, wine_train_y.values, class_num)
    acc = BC.test(wine_test_x.values, wine_test_y.values, prior, train_mean, train_cov, class_num, 'wine', 'BC', True)
    # Naive-Bayes classifier
    crossValidation(wine_train_x, wine_train_y, class_num, 'wine', 'NBC', K)
    prior, train_mean, train_var = NBC.train(wine_train_x.values, wine_train_y.values, class_num)
    acc = NBC.test(wine_test_x.values, wine_test_y.values, prior, train_mean, train_var, class_num, 'wine', 'NBC', True)


    class_num = 2
    print("=================== IONOSPHERE ==============")
    # Bayesian classifier
    crossValidation(ionosphere_train_x, ionosphere_train_y, class_num, 'ionosphere', 'BC', K)
    prior, train_mean, train_cov = BC.train(ionosphere_train_x.values, ionosphere_train_y.values, class_num)
    acc = BC.test(ionosphere_test_x.values, ionosphere_test_y.values, prior, train_mean, train_cov, class_num, 'ionosphere', 'BC', True)
    # Naive-Bayes classifier
    crossValidation(ionosphere_train_x, ionosphere_train_y, class_num, 'ionosphere', 'NBC', K)
    prior, train_mean, train_var = NBC.train(ionosphere_train_x.values, ionosphere_train_y.values, class_num)
    acc = NBC.test(ionosphere_test_x.values, ionosphere_test_y.values, prior, train_mean, train_var, class_num, 'ionosphere', 'NBC', True)
    # Pocket classifier
    crossValidation(ionosphere_train_x, ionosphere_train_y, class_num, 'ionosphere', 'PC', K)
    train_weight = PC.train(ionosphere_train_x.values, ionosphere_train_y.values, class_num)
    acc = PC.test(ionosphere_test_x.values, ionosphere_test_y.values, train_weight, class_num, 'ionosphere', 'PC', True)

    print("===================== BREAST ================")
    # Bayesian classifier
    crossValidation(breast_train_x, breast_train_y, class_num, 'breast', 'BC', K)
    prior, train_mean, train_cov = BC.train(breast_train_x.values, breast_train_y.values, class_num)
    acc = BC.test(breast_test_x.values, breast_test_y.values, prior, train_mean, train_cov, class_num, 'breast', 'BC', True)
    # Naive-Bayes classifier
    crossValidation(breast_train_x, breast_train_y, class_num, 'breast', 'NBC', K)
    prior, train_mean, train_var = NBC.train(breast_train_x.values, breast_train_y.values, class_num)
    acc = NBC.test(breast_test_x.values, breast_test_y.values, prior, train_mean, train_var, class_num, 'breast', 'NBC', True)
    # Pocket classifier
    crossValidation(breast_train_x, breast_train_y, class_num, 'breast', 'PC', K)
    train_weight = PC.train(breast_train_x.values, breast_train_y.values, class_num)
    acc = PC.test(breast_test_x.values, breast_test_y.values, train_weight, class_num, 'breast', 'PC', True)
