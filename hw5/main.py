import os
import argparse
import pickle
import numpy as np
import time

import models
from data import load

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your models.")

    # generic
    parser.add_argument("--data", type=str, default="data",
                        help="The directory where your MNIST data files are stored.")
    parser.add_argument("--predictions-file", type=str, 
                        help="The predictions file to create. (Only used for testing.)")
    parser.add_argument("--test-split", type=str, default="dev",
                        help="The split to make predictions over.")

    # hyperparameters
    parser.add_argument("--dr-algorithm", type=str, default=None,
                        help="The name of the dimensionality reduction alg to use.")
    parser.add_argument("--target-dim", type=int, default=300,
                        help="The number of dimensions to retain.")
    parser.add_argument("--knn-k", type=int, default=5,
                        help="k hyperparameter for KNN model")
    parser.add_argument("--lle-k", type=int, default=10,
                        help="k hyperparameter for LLE model")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    X, y = load(args.data, 'train')
    test_X, _ = load(args.data, args.test_split)

    if args.dr_algorithm is not None:
        num_train = X.shape[0]
        concat_data = np.concatenate((X, test_X))

        start = time.time()
        if args.dr_algorithm == 'pca':
            reduced_X = models.PCA(concat_data, args.target_dim).fit(concat_data)
        elif args.dr_algorithm == 'lle':
            reduced_X = models.LLE(concat_data, args.target_dim, args.lle_k).fit(concat_data)
        else:
            raise Exception('Invalid dimensionality reduction algorithm')
        end = time.time()
        print(f"dimensionality reduction took {end - start} seconds!")
        X = reduced_X[:num_train]
        test_X = reduced_X[num_train:]

    start = time.time()
    model = models.KNN(args.knn_k)
    model.fit(X, y)
    y_hat = model.predict(test_X)
    end = time.time()
    print(f"KNN took {end-start} seconds!")
    np.savetxt(args.predictions_file, y_hat, fmt='%d')

if __name__ == "__main__":
    main()
