import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import LDA
import naive_bayes_classifier as NBC


PROPORTIONAL = 0.6
CLASS_NUM = 2
K = 5


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
    # print()
    # print("# of training data : {}".format(len(train)))
    # print("# of testing data : {}".format(len(test)))
    # print("# of training each class : {}".format(train[class_idx].value_counts()))
    # print("# of testing each class : {}".format(test[class_idx].value_counts()))
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
        validation_x = train_x[start:end]
        validation_y = train_y[start:end].values
        #Choose model
        if model_name == 'NBC':
            prior, train_mean, train_var = NBC.train(training_x, training_y, class_num)
            acc, FA, PD = NBC.test(validation_x, validation_y, prior, train_mean, train_var, class_num, filename, model_name, False)
            total_FA.append(np.array(FA))
            total_PD.append(np.array(PD))
        overall_acc += acc
    print("Overall accuracy: {}".format(overall_acc / K))
    # Plot ROC curve for cv
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
    fig.savefig('plotting/' + filename + '_' + model_name + '_lower_roc_CV.png')


def featureSelection(train_x, train_y, test_x):
    feature_pool = np.arange(train_x.shape[1])
    feature_selected_count = 0
    selected_feature = []
    while feature_selected_count < 2:
        best_acc = 0
        best_feature = None
        for feature in feature_pool:
            # Cannot select same features
            if feature in selected_feature:
                continue
            candidate_feature = selected_feature.copy()
            candidate_feature.append(feature)
            print()
            print("Candidate feature = {}".format(candidate_feature))
            divided = int(len(train_x) / K)
            overall_acc = 0
            total_PD = []
            total_FA = []
            for fold in range(K):
                # print()
                # print("Now fold is {}".format(fold))
                # Compute start and end index
                start = divided * fold
                end = divided * (fold + 1)
                # Filter data so candidate features
                training_x = np.concatenate((train_x[:start, candidate_feature], train_x[end:, candidate_feature]))
                training_y = np.concatenate((train_y[:start], train_y[end:]))
                validation_x = train_x[start:end, candidate_feature]
                validation_y = train_y[start:end]
                y_pred = LDA.knn(training_x, train_y, validation_x)
                acc = LDA.compute_accuracy(y_pred, validation_y)
                overall_acc += acc
            print("Overall accuracy: {}".format(overall_acc / K))
            if overall_acc >= best_acc:
                best_acc = overall_acc
                best_feature = candidate_feature
        selected_feature = candidate_feature
        feature_selected_count += 1
    print("Feature selected: {}".format(selected_feature))
    print(train_x[:, selected_feature])
    return train_x[:, selected_feature], test_x[:, selected_feature]


if __name__ == "__main__":
    # Preprocess selected data
    iris_train_x, iris_train_y, iris_test_x, iris_test_y, \
        ionosphere_train_x, ionosphere_train_y, ionosphere_test_x, ionosphere_test_y, \
        breast_train_x, breast_train_y, breast_test_x, breast_test_y, \
        wine_train_x, wine_train_y, wine_test_x, wine_test_y = readData()

    # print("====================== IRIS =================")
    # iris_train_x_selection, iris_test_x_selection = featureSelection(iris_train_x.values, iris_train_y.values.ravel(), iris_test_x.values)
    # iris_lower_dimension_train, iris_lower_dimension_test = LDA.LDA(iris_train_x_selection, iris_train_y.values.ravel(), iris_test_x_selection, iris_test_y.values.ravel(), 'iris')
    # prior, train_mean, train_cov = NBC.train(iris_train_x.values, iris_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(iris_test_x.values, iris_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'iris', 'NBC', True)
    # # Project to lower dimension
    # # crossValidation(iris_lower_dimension_train, iris_train_y, CLASS_NUM, 'iris', 'NBC', K)        # BUG
    # prior, train_mean, train_cov = NBC.train(iris_lower_dimension_train, iris_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(iris_lower_dimension_test, iris_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'iris_lower', 'NBC', True)


    # print("=================== BREAST ==============")
    # breast_train_x_selection, breast_test_x_selection = featureSelection(breast_train_x.values, breast_train_y.values.ravel(), breast_test_x.values)
    # breast_lower_dimension_train, breast_lower_dimension_test = LDA.LDA(breast_train_x_selection, breast_train_y.values.ravel(), breast_test_x_selection, breast_test_y.values.ravel(), 'breast')
    # prior, train_mean, train_cov = NBC.train(breast_train_x.values, breast_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(breast_test_x.values, breast_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'breast', 'NBC', True)
    # # # Project to lower dimension
    # # crossValidation(iris_lower_dimension_train, iris_train_y, CLASS_NUM, 'iris', 'NBC', K)        # BUG
    # prior, train_mean, train_cov = NBC.train(breast_lower_dimension_train, breast_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(breast_lower_dimension_test, breast_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'breast_lower', 'NBC', True)

    # print("=================== IONOSPHERE ==============")
    # ionosphere_train_x_selection, ionosphere_test_x_selection = featureSelection(ionosphere_train_x.values, ionosphere_train_y.values.ravel(), ionosphere_test_x.values)
    # ionosphere_lower_dimension_train, ionosphere_lower_dimension_test = LDA.LDA(ionosphere_train_x_selection, ionosphere_train_y.values.ravel(), ionosphere_test_x_selection, ionosphere_test_y.values.ravel(), 'ionosphere')
    # prior, train_mean, train_cov = NBC.train(ionosphere_train_x.values, ionosphere_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(ionosphere_test_x.values, ionosphere_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'ionosphere', 'NBC', True)
    # # # Project to lower dimension
    # # crossValidation(iris_lower_dimension_train, iris_train_y, CLASS_NUM, 'iris', 'NBC', K)        # BUG
    # prior, train_mean, train_cov = NBC.train(ionosphere_lower_dimension_train, ionosphere_train_y.values.ravel(), CLASS_NUM)
    # acc = NBC.test(ionosphere_lower_dimension_test, ionosphere_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'ionosphere_lower', 'NBC', True)

    print("=================== WINE ==============")
    wine_train_x_selection, wine_test_x_selection = featureSelection(wine_train_x.values, wine_train_y.values.ravel(), wine_test_x.values)
    wine_lower_dimension_train, wine_lower_dimension_test = LDA.LDA(wine_train_x_selection, wine_train_y.values.ravel(), wine_test_x_selection, wine_test_y.values.ravel(), 'wine')
    prior, train_mean, train_cov = NBC.train(wine_train_x.values, wine_train_y.values.ravel(), CLASS_NUM)
    acc = NBC.test(wine_test_x.values, wine_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'wine', 'NBC', True)
    # # Project to lower dimension
    # crossValidation(iris_lower_dimension_train, iris_train_y, CLASS_NUM, 'iris', 'NBC', K)        # BUG
    prior, train_mean, train_cov = NBC.train(wine_lower_dimension_train, wine_train_y.values.ravel(), CLASS_NUM)
    acc = NBC.test(wine_lower_dimension_test, wine_test_y.values.ravel(), prior, train_mean, train_cov, CLASS_NUM, 'wine_lower', 'NBC', True)