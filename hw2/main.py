""" Main file. This is the starting point for your code execution.

You shouldn't need to change much of this code, but it's fine to as long as we
can still run your code with the arguments specified!
"""

import os
import json
import pickle
import argparse as ap

import numpy as np
import models
from data import load_data


def get_args():
    p = ap.ArgumentParser()

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                    help="Operating mode: train or test.")
    p.add_argument("--train-data", type=str, help="Training data file")
    p.add_argument("--test-data", type=str, help="Test data file")
    p.add_argument("--model-file", type=str, required=True,
                    help="Where to store and load the model parameters")
    p.add_argument("--predictions-file", type=str,
                    help="Where to dump predictions")
    p.add_argument("--algorithm", type=str,
                   choices=['regression-tree', 'gradient-boosted-regression-tree'],
                    help="The type of model to use.")
    # Model Hyperparameters
    p.add_argument("--max-depth", type=int, default=3,
                   help="Model learning rate")
    p.add_argument("--n-estimators", type=int, default=100,
                  help="Number of boosting stages to perform")
    p.add_argument("--regularization-parameter", type=float, default=0.1,
                  help="Regularization parameter used in gradient-boosted-regression-tree")
    return p.parse_args()

def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X, y = load_data(args.train_data)

    # Initialize appropriate algorithm
    if args.algorithm == 'regression-tree':
        model = models.RegressionTree(nfeatures=X.shape[1], max_depth=args.max_depth)
    elif args.algorithm == 'gradient-boosted-regression-tree':
        model = models.GradientBoostedRegressionTree(nfeatures=X.shape[1], max_depth=args.max_depth, n_estimators=args.n_estimators, regularization_parameter=args.regularization_parameter)
    else:
        raise Exception("Algorithm argument not recognized")

    model.fit(X=X, y=y)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


def test(args):
    """ Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, observed_y = load_data(args.test_data)

    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    preds = model.predict(X)

    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%s')


if __name__ == "__main__":
    ARGS = get_args()
    if ARGS.mode.lower() == 'train':
        train(ARGS)
    elif ARGS.mode.lower() == 'test':
        test(ARGS)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
