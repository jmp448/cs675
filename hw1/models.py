""" 
Keep model implementations in here.

This file is where you will write most of your virtual_environment!
"""

import numpy as np
from scipy.sparse import csr_matrix


class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures

    def fit(self, *, X, y, lr):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
            lr: A float, the learning rate of this fit step.
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class MCModel(Model):
    """ A multiclass model abstraction.
    It wants to know, up front:
        - How many features in the data
        - How many classes in the data
    """

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures)
        self.num_classes = nclasses


class MCPerceptron(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        W = self.W
        for obs in range(len(y)):  # once for each observation
            x = csr_matrix.toarray(X[obs])
            yhat = np.argmax(np.dot(x, np.transpose(W)))
            if yhat != y[obs]:
                W[yhat] = W[yhat] - lr * X[obs]
                W[y[obs]] = W[y[obs]] + lr * X[obs]
        self.W = W

    def predict(self, X):
        X = csr_matrix.toarray(self._fix_test_feats(X))
        W = np.transpose(self.W)
        yhat = np.matmul(X, W)
        predictions = np.zeros(len(yhat), dtype=int)
        for i in range(len(predictions)):
            predictions[i] = np.argmax(yhat[i])
        return predictions


class MCLogistic(MCModel):

    def __init__(self, *, nfeatures, nclasses):
        super().__init__(nfeatures=nfeatures, nclasses=nclasses)
        self.W = np.zeros((nclasses, nfeatures), dtype=np.float)

    def fit(self, *, X, y, lr):
        W = self.W
        correction = np.INF
        while correction > 0.01:
            X = csr_matrix.toarray(X)
            g = np.dot(X, np.transpose(W))
            for k in range(len(W)):
                p = self.softmax(g)
                correction = lr * p[0, k] * X
                if k == y[obs]:
                    W[k:k+1] += x - correction
                else:
                    W[k:k+1] -= correction
        self.W = W


    def predict(self, X):
        X = self._fix_test_feats(X)
        X = csr_matrix.toarray(self._fix_test_feats(X))
        W = np.transpose(self.W)
        yhat = np.matmul(X, W)
        predictions = np.zeros(len(yhat), dtype=int)
        for i in range(len(predictions)):
            predictions[i] = np.argmax(yhat[i])
        return predictions

    def softmax(self, logits):
        g = logits
        gmax = max(g)
        tot = 0
        for k in range(len(g)):
            tot += np.exp(g[k]-gmax)
        for k in range(len(g)):
            g[k] = np.exp(g[k]-gmax)/tot
        return g


class OneVsAll(Model):

    def __init__(self, *, nfeatures, nclasses, model_class):
        super().__init__(nfeatures)
        self.num_classes = nclasses
        self.model_class = model_class
        self.models = [model_class(nfeatures=nfeatures, nclasses=2) for _ in range(nclasses)]

    def fit(self, *, X, y, lr):
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        X = self._fix_test_feats(X)
        # TODO: Implement this!
        raise Exception("You must implement this method!")
