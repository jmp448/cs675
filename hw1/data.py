""" Loads a dataset.

You shouldn't need to change any of this virtual_environment! (hopefully)
"""

import numpy as np
from scipy.sparse import csr_matrix


def load_data(filename, mc=True):
    """ Load data.

    Args:
        filename: A string. The path to the data file.
        mc: Whether or not the data is multiclass. MC labels are not 0-base idxd

    Returns:
        A tuple, (X, y, class_count). 
        X is a compressed sparse row matrix of floats with shape [num_examples, num_features]. 
        y is a dense array of ints with shape [num_examples].
        class_count is a count of the number of classes in the dataset.
    """

    X_nonzero_rows, X_nonzero_cols, X_nonzero_values = [], [], []
    y = []
    max_class = 0 # keep track of how many classes we have
    with open(filename) as reader:
        for example_index, line in enumerate(reader):
            if len(line.strip()) == 0:
                continue

            # Divide the line into features and labels.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
                if mc:
                    # multiclass labels are not 0-based
                    int_label -= 1
                if int_label > max_class:
                    max_class = int_label
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")
            y.append(int_label)

            for item in split_line[1:]:
                try:
                    # Features are not 0-based
                    feature_index = int(item.split(":")[0]) - 1
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                if feature_index < 0:
                    raise Exception('Expected feature indices to be 1 indexed, but found index of 0.')
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")

                if value != 0.0:
                    X_nonzero_rows.append(example_index)
                    X_nonzero_cols.append(feature_index)
                    X_nonzero_values.append(value)

    X = csr_matrix((X_nonzero_values, (X_nonzero_rows, X_nonzero_cols)), dtype=np.float)
    y = np.array(y, dtype=np.int)

    return X, y, max_class + 1
