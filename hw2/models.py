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
    def __init__(self, max_depth, depth, is_leaf=False, left=None, right=None, split_feature=None, threshold=None, label=None):
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = is_leaf
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.threshold = threshold
        self.label = label

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
            X_left, y_left, X_right, y_right = self.optimized_split(X, y)
            self.left = Node(self.max_depth, self.depth + 1)
            self.right = Node(self.max_depth, self.depth + 1)
            self.left.build_regression_tree(X_left, y_left)
            self.right.build_regression_tree(X_right, y_right)

    def optimized_split(self, X, y):

        # Perform sorting for speedup down the road
        # Xsort = np.zeros(X.shape)
        # for d in range(X.shape[1]):
        #     xsort = np.sort(X[:, d])
        #     for i in range(X.shape[0]):
        #         Xsort[i, d] = X[:, d].index(xsort[i])

        # Calculate SSEs
        min_sse = np.inf
        for d in range(X.shape[1]):
            for thresh in X[:, d]:
                left_set = []
                for i in range(X.shape[0]):
                    if X[i, d] <= thresh:
                        left_set.append(i)
                if len(left_set) == 0 or len(left_set) == X.shape[0]:
                    continue
                else:
                    y_left = np.take(y, left_set, 0)
                    y_right = np.delete(y, left_set, 0)
                    sse = sum((y_left-np.full(len(y_left), np.average(y_left))) ** 2) +\
                        sum((y_right - np.full(len(y_right), np.average(y_right))) ** 2)
                    if sse < min_sse:
                        min_sse = sse
                        self.split_feature = d
                        self.threshold = thresh
                        X_left_min = np.delete(X, right_set, 0)
                        X_right_min = np.delete(X, left_set, 0)
                        y_left_min = y_left
                        y_right_min = y_right

        return X_left_min, y_left_min, X_right_min, y_right_min


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