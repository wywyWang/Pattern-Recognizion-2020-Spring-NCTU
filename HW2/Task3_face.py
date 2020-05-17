from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import re

LOWER_D = 500
K = 5


def read_input_face(filename, storedir):
    img = Image.open(filename)
    width, height = img.size
    print(width, height)
    pixel = np.asarray(img)
    data = []
    target = []
    totalfile = []

    data_test = []
    data_train = []
    target_train = []
    target_test = []
    totalfile_train = []
    totalfile_test = []
    for i in range(16):
        for j in range(5):
            each_image = pixel[j * 40 : (j+1) * 40, i * 40 : (i+1) * 40].copy().reshape(40 * 40)
            if j == 0:
                data_test.append(each_image)
                target_test.append(i)
                totalfile_test.append(str(i) + str(j) + '.png')
            if j < 5:
                data_train.append(each_image)
                target_train.append(i)
                totalfile_train.append(str(i) + str(j) + '.png')
                # Data augmentation
                each_copy = each_image.copy().reshape(40, 40)
                # Vertical flip
                each_copy_vertical = np.flip(each_copy, axis=1).reshape(40 * 40)
                data_train.append(each_copy_vertical)
                target_train.append(i)
                totalfile_train.append(str(i) + str(j) + '.png')
                # if i == 0 and j != 0:
                #     new_img = Image.new(img.mode, (40, 40))
                #     new_img.putdata(each_copy_vertical)
                #     new_img.save(storedir + str(i) + str(j) + '_vertical.bmp')
            # data.append(each_image)
            # target.append(i)
            # totalfile.append(str(i) + str(j) + '.png')
            # # Test pixels are correct for one image
            # new_img = Image.new(img.mode, (40, 40))
            # new_img.putdata(each_image)
            # new_img.save(storedir + str(i) + str(j) + '.bmp')
            # plt.imshow(each_image.reshape(40, 40), plt.cm.gray)
            # plt.savefig(storedir + str(i) + str(j) + '.png')
    
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
        img = Image.new('L', (40, 40))
        pixel = img.load()
        each_copy = each_data.reshape(40, 40).copy()
        for i in range(each_copy.shape[0]):
            for j in range(each_copy.shape[1]):
                pixel[i, j] = each_copy[i, j].copy()
        img.save(storedir + totalfile[each_id])
        # each_image = each_data.reshape(40, 40).copy().T
        # plt.imshow(each_image, plt.cm.gray)
        # plt.savefig(storedir + totalfile[each_id])


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
        # print(neighbor)
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
    covariance = np.cov(np.matmul(data.T, data))
    eigen_vectors = compute_eigen(covariance)
    lower_dimension_data = np.matmul(data, eigen_vectors)
    # print(lower_dimension_data)
    # print(eigen_vectors)
    print()
    return lower_dimension_data, eigen_vectors


if __name__ == '__main__':
    dirtrain = './task3_data/facesP1.bmp'
    storedir = './task3_face_parsed/'
    data_train, target_train, totalfile_train,\
        data_test, target_test, totalfile_test = read_input_face(dirtrain, storedir)
    
    lower_dimension_data, eigen_vectors = PCA(data_train)
    print("data shape = {}".format(data_train.shape))
    print("eigen vector shape = {}".format(eigen_vectors.shape))
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))    

    # reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
    # print("reconstruct_data shape: {}".format(reconstruct_data.shape))
    # storedir = './task3_face_eigenface/'
    # draweigenface(storedir, eigen_vectors)

    # storedir = './task3_face_reconstruct/'
    # visualization(storedir, totalfile_train, reconstruct_data)
    # lower_dimension_data_test, eigen_vectors_test = PCA(data_test)
    lower_dimension_data_test = np.matmul(data_test, eigen_vectors)
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    print("lower_dimension_data_test shape: {}".format(lower_dimension_data_test.shape))
    predict = KNN(lower_dimension_data, lower_dimension_data_test, target_train)
    checkperformance(target_test, predict)
    