""" Main file. This is the starting point for your virtual_environment execution. 

You shouldn't need to change much of this virtual_environment, but it's fine to as long as we
can still run your virtual_environment with the arguments specified!
"""


import os
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
                   choices=['mc-perceptron', 'mc-logistic', 
                            '1vA-perceptron', '1vA-logistic'],
                    help="The type of model to use.")

    # Model Hyperparameters
    p.add_argument("--learning-rate", type=float, default=0.001,
                   help="Model learning rate")
    p.add_argument("--train-epochs", type=int, default=15,
                  help="Number of epochs to train the model")

    return p.parse_args()


def check_args(args):
    mandatory_args = {'mode', 'model_file', 'test_data', 'train_data',
                      'predictions_file', 'algorithm', 'learning_rate',
                      'train_epochs'}

    if not mandatory_args.issubset(set(dir(args))):
        raise Exception(("You're missing essential arguments!"
                         "We need these to run your virtual_environment."))

    if args.model_file is None:
        raise Exception("--model-file should be specified in either mode")

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified during training")
        if args.model_file is None:
            raise Exception("--model-file should be specified during training")
        if args.train_data is None:
            raise Exception("--train-data should be specified during training")
        elif not os.path.exists(args.train_data):
            raise Exception("data file specified by --train-data does not exist.")
    elif args.mode.lower() == "test":
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified during testing")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")
        if args.test_data is None:
            raise Exception("--test-data should be specified during testing")
        elif not os.path.exists(args.test_data):
            raise Exception("data file specified by --test-data does not exist.")
    else:
        raise Exception("Invalid mode")


def test(args):
    """ Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _, _ = load_data(args.test_data)
    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    preds = model.predict(X)
    # re-align our preds with the 1-based labels in the data
    preds = [pred + 1 for pred in preds]
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """

    X, y, nclasses = load_data(args.train_data)

    # build the appropriate model
    if args.algorithm == "mc-logistic":
        model = models.MCLogistic(nfeatures=X.shape[1], nclasses=nclasses)
    elif args.algorithm == "mc-perceptron":
        model = models.MCPerceptron(nfeatures=X.shape[1], nclasses=nclasses)
    elif args.algorithm == "1vA-logistic":
        model = models.OneVsAll(nfeatures=X.shape[1], nclasses=nclasses, model_class=models.MCLogistic)
    elif args.algorithm == "1vA-perceptron":
        model = models.OneVsAll(nfeatures=X.shape[1], nclasses=nclasses, model_class=models.MCPerceptron)
    else:
        raise Exception("Algorithm argument not recognized")

    # Run the training loop
    for epoch in range(args.train_epochs):
        model.fit(X=X, y=y, lr=args.learning_rate)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


if __name__ == "__main__":
    ARGS = get_args()
    check_args(ARGS)

    if ARGS.mode.lower() == 'train':
        train(ARGS)
    elif ARGS.mode.lower() == 'test':
        test(ARGS)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
