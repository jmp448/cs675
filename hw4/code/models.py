"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


def initialize_variational_parameters(num_rows_of_image, num_cols_of_image, K):
    """ Helper function to initialize variational distributions before each E-step.
    Args:
                num_rows_of_image: Integer representing the number of rows in the image
                num_cols_of_image: Integer representing the number of columns in the image
                K: The number of latent states in the MRF
    Returns:
                q: 3-dimensional numpy matrix with shape [num_rows_of_image, num_cols_of_image, K]
     """
    q = np.random.random((num_rows_of_image, num_cols_of_image, K))
    for row_num in range(num_rows_of_image):
        for col_num in range(num_cols_of_image):
            q[row_num, col_num, :] = q[row_num, col_num, :]/sum(q[row_num, col_num, :])
    return q


def initialize_theta_parameters(K):
    """ Helper function to initialize theta before beginning of EM.
    Args:
                K: The number of latent states in the MRF
    Returns:
                mu: A numpy vector of dimension [K] representing the mean for each of the K classes
                sigma: A numpy vector of dimension [K] representing the standard deviation for each of the K classes
    """
    mu = np.zeros(K)
    sigma = np.zeros(K) + 10
    for k in range(K):
        mu[k] = np.random.randint(10, 240)
    return mu, sigma


def find_neighbors(s):
    """Helper function to determine each pixel's neighbors for the Potts model
    Args:
                s: the shape of the image, as a tuple
    Returns:
                neighbors: dictionary of position (i,j) to a list of all neighboring positions (i', j')
    """
    neighbors = {}
    for i in range(s[0]):
        for j in range(s[1]):
            if i == 0 and j == 0:
                neighbors[i, j] = [(i + 1, j), (i, j + 1)]
            elif i == 0 and j == s[1]-1:
                neighbors[i, j] = [(i + 1, j), (i, j - 1)]
            elif i == s[0]-1 and j == 0:
                neighbors[i, j] = [(i - 1, j), (i, j + 1)]
            elif i == s[0]-1 and j == s[1]-1:
                neighbors[i, j] = [(i - 1, j), (i, j - 1)]
            elif i == 0:  # j in middle
                neighbors[i, j] = [(i + 1, j), (i, j - 1), (i, j + 1)]
            elif i == s[0]-1:  # j in middle
                neighbors[i, j] = [(i - 1, j), (i, j - 1), (i, j + 1)]
            elif j == 0:  # i in middle
                neighbors[i, j] = [(i - 1, j), (i + 1, j), (i, j + 1)]
            elif j == s[1]-1:  # i in middle
                neighbors[i, j] = [(i - 1, j), (i + 1, j), (i, j - 1)]
            else:  # both in middle
                neighbors[i, j] = [(i, j - 1), (i - 1, j), (i + 1, j), (i, j + 1)]
    return neighbors


def norm(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)


class MRF(object):
    def __init__(self, J, K, n_em_iter, n_vi_iter):
        self.J = J
        self.K = K
        self.n_em_iter = n_em_iter
        self.n_vi_iter = n_vi_iter
        self.q = None

    def fit(self, *, X):
        """ Fit the model.
                Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]
        """

        # figure out the neighbors of each node in advance to avoid recalculating
        neighbors = find_neighbors(X.shape)
        mu, sigma = initialize_theta_parameters(self.K)

        # EM
        for _ in range(self.n_em_iter):
            # E STEP
            q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
            for _ in range(self.n_vi_iter):
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        denom = sum([norm(X[i, j], mu[ks], sigma[ks]) *
                                  np.exp(sum([self.J * q[n[0], n[1], ks] for n in neighbors[i, j]])) for ks in range(self.K)])
                        for k in range(self.K):
                            q[i, j, k] = norm(X[i, j], mu[k], sigma[k]) * \
                                      np.exp(sum([self.J * q[n[0], n[1], k] for n in neighbors[i, j]])) / denom
            # M STEP
            for k in range(self.K):
                mu[k] = sum(map(sum, np.multiply(q[:, :, k], X)))/sum(map(sum, q[:, :, k]))
                sigma[k] = np.sqrt(sum(map(sum, np.multiply(q[:, :, k], np.power(X-mu[k], 2))))/sum(map(sum, q[:, :, k])))

            # run E step one more time
            q = initialize_variational_parameters(X.shape[0], X.shape[1], self.K)
            for _ in range(self.n_vi_iter):
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        denom = sum([norm(X[i, j], mu[ks], sigma[ks]) *
                                     np.exp(sum([self.J * q[n[0], n[1], ks] for n in neighbors[i, j]])) for ks in
                                     range(self.K)])
                        for k in range(self.K):
                            q[i, j, k] = norm(X[i, j], mu[k], sigma[k]) * \
                                      np.exp(sum([self.J * q[n[0], n[1], k] for n in neighbors[i, j]])) / denom

            self.q = q


    def predict(self, X):
        """ Predict.
        Args:
                X: A matrix of floats with shape [num_rows_of_image, num_cols_of_image]

        Returns:
                A matrix of ints with shape [num_rows_of_image, num_cols_of_image].
                    - Each element of this matrix should be the most likely state according to the trained model for the pixel corresponding to that row and column
                    - States should be encoded as {0,..,K-1}
        """
        assignments = np.argmax(self.q, 2)
        return assignments
