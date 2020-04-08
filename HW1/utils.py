import numpy as np

def computeConfusionMatrix(predict, answer, class_num, model_name, threshold=None):
    """Compute confusion matrix based on predicted class and actual class."""
    confusion_matrix = np.zeros((class_num, class_num))
    if model_name == 'PC':
        answer[answer == -1] = 0
    for data_idx in range(len(predict)):
        if model_name == 'NBC':
            if class_num == 2:
                if predict[data_idx] <= threshold:
                    prediction = 1
                else:
                    prediction = 0
                confusion_matrix[prediction][answer[data_idx]] += 1           
            else:
                prediction = np.argmin(predict[data_idx])
                confusion_matrix[prediction][answer[data_idx]] += 1
        elif model_name == 'PC':
            if predict[data_idx][0] <= threshold:
                prediction = 0
            else:
                prediction = 1
            confusion_matrix[prediction][answer[data_idx][0]] += 1
    if class_num == 2:
        PD = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1]) if (confusion_matrix[0][1] + confusion_matrix[1][1]) != 0 else 0
        FA = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[0][0]) if (confusion_matrix[1][0] + confusion_matrix[0][0]) != 0 else 0
        print("Confusion matrix:")
        print(confusion_matrix)
        return [FA, PD]
    print("Confusion matrix:")
    print(confusion_matrix)