"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.root = None

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """
        self.root = Node(max_depth=self.max_depth, depth=0)
        self.root.build_regression_tree(X, y)

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            curr = self.root
            while curr.is_leaf is False:
                if X[i, curr.split_feature] <= curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            y[i] = curr.label
        return y


class Node(RegressionTree):
    def __init__(self, max_depth, depth):
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = False
        self.left = None
        self.right = None
        self.split_feature = None
        self.threshold = None
        self.label = None

    def build_regression_tree(self, X, y):
        novar = True
        for k in range(X.shape[1]):
            if np.var(X[:, k]) != 0:
                novar = False

        if len(y) <= 1:  # Base case, single data point
            assert len(y) > 0, "Empty subtree"
            self.is_leaf = True
            self.label = y[0]
            return
        elif novar:
            self.is_leaf = True
            self.label = np.average(y)
            return
        elif self.depth >= self.max_depth:
            self.is_leaf = True
            self.label = np.average(y)
            return
        else:
            left_set = self.optimized_split(X, y)
            X_left = np.take(X, left_set, axis=0)
            y_left = np.take(y, left_set, axis=0)
            X_right = np.delete(X, left_set, axis=0)
            y_right = np.delete(y, left_set, axis=0)
            self.left = Node(self.max_depth, self.depth + 1)
            self.right = Node(self.max_depth, self.depth + 1)
            self.left.build_regression_tree(X_left, y_left)
            self.right.build_regression_tree(X_right, y_right)

    def optimized_split(self, X, y):

        # Perform sorting for speedup down the road
        Xsort = np.zeros(X.shape, dtype=int)
        for d in range(X.shape[1]):
            Xsort[:, d] = np.argsort(X[:, d])

        # Calculate SSEs
        min_sse = np.inf
        for d in range(X.shape[1]):
            visited = []
            for i in range(1, X.shape[0]-1):
                # Need to work backwards to ensure equal or less than values are included properly
                i_adj = X.shape[0]-i-1
                threshold = X[Xsort[i_adj, d], d]
                if visited.__contains__(threshold):
                    continue
                else:
                    visited.append(threshold)
                    left_set = Xsort[:i_adj, d]
                    if len(left_set) == 0 or len(left_set) == X.shape[0]:
                        continue
                    y_left = np.take(y, left_set, axis=0)
                    y_right = np.delete(y, left_set, axis=0)
                    sse = sum((y_left-np.full(len(y_left), np.average(y_left))) ** 2) +\
                        sum((y_right - np.full(len(y_right), np.average(y_right))) ** 2)
                    if sse < min_sse:
                        min_sse = sse
                        self.split_feature = d
                        self.threshold = threshold
                        threshold_index_max = i_adj

        left_set = Xsort[:threshold_index_max, self.split_feature]

        return left_set


class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")