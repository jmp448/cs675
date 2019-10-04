import numpy as np
import sys

def get_true_values(file_name):
    head_count = 0
    f = open(file_name)
    true_labels = []
    for line in f:
        line = line.strip()
        data = line.split()
        if head_count == 0:
            head_count = head_count + 1
            continue
        true_labels.append(float(data[-1]))
    return np.asarray(true_labels)


def get_predicted_values(predictions_file):
    f = open(predictions_file)
    predictions = []
    for line in f:
        line = line.rstrip()
        predictions.append(float(line))
    return np.asarray(predictions)


data_file = sys.argv[1]
predictions_file = sys.argv[2]




true = get_true_values(data_file)
predicted = get_predicted_values(predictions_file)

if len(predicted) != len(true):
    print('Number of lines in two files do not match')
    sys.exit()


rmse = np.sqrt(np.mean(np.square(true-predicted)))


print('RMSE: ' + str(rmse))
