"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np


class RegressionTree(object):
    def __init__(self, nfeatures, max_depth, depth=0):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.depth = depth
        self.is_leaf = False
        self.left = None
        self.right = None
        self.split_feature = None
        self.threshold = None
        self.label = None

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
        """
        self.build_regression_tree(X, y)

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            curr = self
            while curr.is_leaf is False:
                if X[i, curr.split_feature] < curr.threshold:
                    curr = curr.left
                else:
                    curr = curr.right
            y[i] = curr.label
        return y

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
        elif np.var(y) == 0:
            self.is_leaf = True
            self.label = y[0]
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
            X_left = np.take(X, left_set, axis=0)[0]
            y_left = np.take(y, left_set, axis=0)[0]
            X_right = np.delete(X, left_set, axis=0)
            y_right = np.delete(y, left_set, axis=0)
            self.left = RegressionTree(nfeatures=X_left.shape[1], max_depth=self.max_depth, depth=self.depth + 1)
            self.right = RegressionTree(nfeatures=X_right.shape[1], max_depth=self.max_depth, depth=self.depth + 1)
            self.left.build_regression_tree(X_left, y_left)
            self.right.build_regression_tree(X_right, y_right)

    def optimized_split(self, X, y):

        # Calculate SSEs
        min_sse = np.inf
        for d in range(X.shape[1]):
            splits = np.unique(X[:, d])
            for i in range(1, len(splits)):  # split on maximum value will yield empty set
                threshold = splits[i]
                left = np.where(X[:, d] < np.full(X.shape[0], threshold))
                y_left = np.take(y, left, axis=0)[0]
                y_right = np.delete(y, left, axis=0)
                sse = len(y_left)*np.var(y_left) + len(y_right)*np.var(y_right)
                if sse < min_sse:
                    min_sse = sse
                    self.split_feature = d
                    self.threshold = threshold
                    left_set = left
        return left_set


class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
        self.trees = []
        self.f0 = 0

    def fit(self, *, X, y):
        self.f0 = np.average(y)
        f = np.full(len(y), np.average(y))
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(nfeatures=self.num_input_features, max_depth=self.max_depth))
            g = y - f
            self.trees[i].fit(X=X, y=g)
            h = self.trees[i].predict(X)
            f += self.regularization_parameter*h

        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        y = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            y += self.regularization_parameter*tree.predict(X)
        return y
