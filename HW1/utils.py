import numpy as np

def computeComfusionMatrix(predict, answer, class_num):
    confusion_matrix = np.zeros((class_num, class_num))
    for data_idx in range(len(predict)):
        confusion_matrix[predict[data_idx]][answer[data_idx]] += 1
    print("Confusion matrix:")
    print(confusion_matrix)