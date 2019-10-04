""" Loads a dataset.

You shouldn't need to change any of this code! (hopefully)
"""

import numpy as np



def load_data(filename):
    """ Load data.

    Args:
        filename: A string. The path to the data file.

    Returns:
        A tuple, (X, y).
        X is a matrix of floats with shape [num_examples, num_features].
        y is an array of floats with shape [num_examples].
    """
    full_data = np.loadtxt(filename, delimiter='\t', skiprows=1)
    y = full_data[:, -1]
    X = full_data[:, 1:-1]
    return X, y
