from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import re

LOWER_D = 3
PIC_COUNT = 10
SHAPE = (400, 400)
K = 5
PROPORTIONAL = 0.8


def read_input_gender(filename, storedir, gender):
    img = Image.open(filename)
    img = img.resize(SHAPE)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width, height))
    data = []
    target = []
    totalfile = []
    for i in range(PIC_COUNT):
        for j in range(PIC_COUNT):
            each_image = pixel[i * 40:(i+1) * 40, j * 40:(j+1) * 40].copy().reshape(40 * 40)
            data.append(each_image)
            target.append(gender)
            if gender == 0:
                totalfile.append(str(i) + str(j) + 'f.png')
            else:
                totalfile.append(str(i) + str(j) + 'm.png')
    data = np.array(data)
    target = np.array(target)
    totalfile = np.array(totalfile)
    # train_id = np.random.choice(data.shape[0], int(data.shape[0] * PROPORTIONAL), replace=False)
    train_id = np.arange(80)
    data_test = []
    data_train = []
    target_train = []
    target_test = []
    totalfile_train = []
    totalfile_test = []
    for i in range(data.shape[0]):
        if i not in train_id:
            data_test.append(data[i, :])
            target_test.append(target[i])
            totalfile_test.append(totalfile[i])
        else:
            data_train.append(data[i, :])
            target_train.append(target[i])
            totalfile_train.append(totalfile[i])
    data_test = np.array(data_test)
    target_test = np.array(target_test)
    totalfile_test = np.array(totalfile_test)
    data_train = np.array(data_train)
    target_train = np.array(target_train)
    totalfile_train = np.array(totalfile_train)
    print(data_test.shape)
    print(target_test.shape)
    print(totalfile_test.shape)
    print()
    print(data_train.shape)
    print(target_train.shape)
    print(totalfile_train.shape)
    return data_train, target_train, totalfile_train, data_test, target_test, totalfile_test


def compute_eigen(A):
    eigen_values, eigen_vectors = np.linalg.eigh(A)
    print("eigen_values = {}".format(eigen_values.shape))
    idx = eigen_values.argsort()[::-1]                          # sort largest
    return eigen_vectors[:,idx][:,:LOWER_D]


def visualization(storedir, totalfile, data):
    for each_id, each_data in enumerate(data):
        if each_id == 2:
            break
        img = Image.new('L', (40, 40), 'white')
        print(each_data)
        pixel = img.load()
        each_copy = each_data.reshape(40, 40).copy()
        for i in range(each_copy.shape[0]):
            for j in range(each_copy.shape[1]):
                print(each_copy[i, j])
                print(pixel[i, j])
                pixel[i, j] = each_copy[i, j].copy()
        img.save(storedir + totalfile[each_id])
        # each_image = each_data.reshape(40, 40).copy().T
        # plt.imshow(each_image, plt.cm.gray)
        # plt.savefig(storedir + totalfile[each_id])

    # img = Image.open(filename)
    # img = img.resize(SHAPE, Image.ANTIALIAS)
    # width, height = img.size
    # idx = 0
    # for file in totalfile:
    #     filename = dirname + file
    #     storename = storedir + file
    #     img = Image.open(filename)
    #     img = img.resize(SHAPE, Image.ANTIALIAS)
    #     width, height = img.size
    #     pixel = img.load()
    #     pixel[i * 40:(i+1) * 40, j * 40:(j+1) * 40] = data[idx].reshape(width, height).copy()
    #     img.save(storename + '.png')
    #     idx += 1


def draweigenface(storedir, eigen_vectors):
    title = "PCA Eigen-Face" + '_'
    eigen_vectors = eigen_vectors.T
    for i in range(0, LOWER_D):
        plt.clf()
        plt.suptitle(title + str(i))
        plt.imshow(eigen_vectors[i].reshape((40, 40)), plt.cm.gray)
        plt.savefig(storedir + title + str(i) + '.png')


def KNN(traindata, testdata, target):
    trainsize = traindata.shape[0]
    testsize = testdata.shape[0]
    y_pred = []
    for testidx in range(testsize):
        all_dist = np.zeros(trainsize)
        for trainidx in range(trainsize):
            all_dist[trainidx] = np.sqrt(np.sum((testdata[testidx] - traindata[trainidx]) ** 2))
        sort_idx = all_dist.argsort()
        neighbor = list(target[sort_idx][:K])
        prediction = max(set(neighbor), key=neighbor.count)
        y_pred.append(prediction)
    y_pred = np.array(y_pred)
    return y_pred


def checkperformance(targettest, predict):
    correct = 0
    for i in range(len(targettest)):
        print("Predict: {}, Answer: {}".format(predict[i], targettest[i]))
        if targettest[i] == predict[i]:
            correct += 1
    print("Accuracy of PCA = {}  ({} / {})".format(correct / len(targettest), correct, len(targettest)))


def PCA(data):
    covariance = np.cov(data.T)
    eigen_vectors = compute_eigen(covariance)
    lower_dimension_data = np.matmul(data, eigen_vectors)
    # print(lower_dimension_data)
    # print(eigen_vectors)
    print()
    return lower_dimension_data, eigen_vectors


if __name__ == '__main__':
    # dirtrain = './task3_data/fP1.bmp'
    # storedir = './task3_parsed/'
    # data_train_f, target_train_f, totalfile_train_f, data_test_f, target_test_f, totalfile_test_f = read_input_gender(dirtrain, storedir, 0)
    # dirtrain = './task3_data/mP1.bmp'
    # data_train_m, target_train_m, totalfile_train_m, data_test_m, target_test_m, totalfile_test_m = read_input_gender(dirtrain, storedir, 1)
    # data_train = np.concatenate((data_train_f, data_train_m))
    # target_train = np.concatenate((target_train_f, target_train_m))
    # totalfile_train = np.concatenate((totalfile_train_f, totalfile_train_m))

    # data_test = np.concatenate((data_test_f, data_test_m))
    # target_test = np.concatenate((target_test_f, target_test_m))
    # totalfile_test = np.concatenate((totalfile_test_f, totalfile_test_m))

    # lower_dimension_data, eigen_vectors = PCA(data_train)
    # print("data shape = {}".format(data_train.shape))
    # print("eigen vector shape = {}".format(eigen_vectors.shape))
    # print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))

    # reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
    # print("reconstruct_data shape: {}".format(reconstruct_data.shape))
    # storedir = './task3_eigenface/'
    # draweigenface(storedir, eigen_vectors)

    # # storedir = './task3_reconstruct/'
    # # visualization(storedir, totalfile_train, reconstruct_data)
    # lower_dimension_data_test, eigen_vectors_test = PCA(data_test)
    # print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    # print("lower_dimension_data_test shape: {}".format(lower_dimension_data_test.shape))
    # predict = KNN(lower_dimension_data, lower_dimension_data_test, target_train)
    # checkperformance(target_test, predict)