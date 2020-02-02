""" Main function, which trains our model and makes predictions with it. """
import csv
import argparse as ap

import torch
import torch.nn.functional as F
import numpy as np

from data import load
from models import FeedForward, SimpleConvNN, BestNN


def get_args():
    """ Define our command line arguments. """
    p = ap.ArgumentParser()

    # Mode to run the model in.
    p.add_argument("mode", choices=["train", "predict"], type=str)

    # File locations 
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--log-file", type=str, default="ff-logs.csv")
    p.add_argument("--model-save", type=str, default="ff-model.torch")
    p.add_argument("--predictions-file", type=str, default="ff-preds.txt")

    # hyperparameters
    p.add_argument("--model", type=str, default="simple-ff")
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--train-my-way", type=bool, default=True)
    p.add_argument("--min-iter", type=int, default=500)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--stepwise-cushion", type=float, default=-0.02)
    p.add_argument("--timesaver-cushion", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=0.001)

    # simple-ff hparams
    p.add_argument("--ff-hunits", type=int, default=100)

    # simple-cnn hparams
    p.add_argument('--cnn-n1-channels', type=int, default=40)
    p.add_argument('--cnn-n1-kernel', type=int, default=10)
    p.add_argument('--cnn-n2-kernel', type=int, default=5)

    # best model hparams
    p.add_argument('--best-n1-channels', type=int, default=40)
    p.add_argument('--best-n2-channels', type=int, default=10)
    p.add_argument('--best-n3-channels', type=int, default=5)
    return p.parse_args()


def train(args):
    np.random.seed(42)
    torch.manual_seed(42)

    # setup metric logging. It's important to log your loss!!
    log_f = open(args.log_file, 'w')
    fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
    logger = csv.DictWriter(log_f, fieldnames)
    logger.writeheader()

    # load data
    train_data, train_labels = load(args.data_dir, split="train")
    dev_data, dev_labels = load(args.data_dir, split="dev")

    # Build model
    if args.model.lower() == "simple-ff":
        model = FeedForward(args.ff_hunits)
    elif args.model.lower() == "simple-cnn":
        model = SimpleConvNN(args.cnn_n1_channels,
                            args.cnn_n1_kernel,
                            args.cnn_n2_kernel)
    elif args.model.lower() == "best":
        model = BestNN(args.best_n1_channels,
                       args.best_n2_channels,
                       args.best_n3_channels)
    else:
        raise Exception("Unknown model type passed in!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if not args.train_my_way:
        for step in range(args.train_steps):
            # run the model and backprop for train steps
            i = np.random.choice(train_data.shape[0], size=args.batch_size, replace=False)
            x = torch.from_numpy(train_data[i].astype(np.float32))
            y = torch.from_numpy(train_labels[i].astype(np.int))

            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # every 100 steps, log metrics
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model,
                                                                  train_data,
                                                                  train_labels)
                dev_acc, dev_loss = dev_acc_and_loss(model, dev_data, dev_labels)

                step_metrics = {
                    'step': step,
                    'train_loss': loss.item(),
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                print(f'On step {step}: Train loss {train_loss} | Dev acc is {dev_acc}')
                logger.writerow(step_metrics)
        # close the log file
        log_f.close()
        # save model
        print(f'Done training. Saving model at {args.model_save}')
        torch.save(model, args.model_save)
    else:
        '''
        MY OPTIMIZATION SCHEME
        
        Three conditions decide whether to continue training
        
        1. Always train for at least 'min_steps', and no more than 'max_steps'
        2. If dev acc drops by 'stepwise_cushion' or more between measured points (every 100 steps), stop training
        3. If dev acc has improved by less than 'timesaver_cushion' in the past 1000 iteration
        '''

        # Set up improving
        last_acc = 0
        improving = True

        # Set up got_time
        last1000 = 0
        got_time = True

        step = 0

        while step <= args.max_iter and (step <= args.min_iter or (improving and got_time)):
            # run the model and backprop for train steps
            i = np.random.choice(train_data.shape[0], size=args.batch_size, replace=False)
            x = torch.from_numpy(train_data[i].astype(np.float32))
            y = torch.from_numpy(train_labels[i].astype(np.int))

            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # every 100 steps, log metrics
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model,
                                                                  train_data,
                                                                  train_labels)
                dev_acc, dev_loss = dev_acc_and_loss(model, dev_data, dev_labels)

                step_metrics = {
                    'step': step,
                    'train_loss': loss.item(),
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc
                }

                print(f'On step {step}: Train loss {train_loss} | Dev acc is {dev_acc}')
                logger.writerow(step_metrics)

                # Update conditions
                diff = dev_acc - last_acc
                improving = diff > args.stepwise_cushion
                last_acc = dev_acc

                if step % 1000 == 0:
                    got_time = dev_acc - last1000 > args.timesaver_cushion
                    last1000 = dev_acc

            step += 1

        # close the log file
        log_f.close()
        # save model
        print(f'Done training. Saving model at {args.model_save}')
        torch.save(model, args.model_save)


def approx_train_acc_and_loss(model, train_data, train_labels):
    idxs = np.random.choice(len(train_data), 4000, replace=False)
    x = torch.from_numpy(train_data[idxs].astype(np.float32))
    y = torch.from_numpy(train_labels[idxs].astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()


def dev_acc_and_loss(model, dev_data, dev_labels):
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred.numpy()), loss.item()


def accuracy(y, y_hat):
    return (y == y_hat).astype(np.float).mean()


def test(args):
    # You should not change this function at all
    model = torch.load(args.model_save)
    test_data, _ = load(args.data_dir, split="test", load_labels=False)

    preds = []
    for test_ex in test_data:
        x = torch.from_numpy(test_ex.astype(np.float32))
        # Make the x look like it's in a batch of size 1
        x = x.view(1, -1)
        logits = model(x)
        pred = torch.max(logits, 1)[1]
        preds.append(pred.item())
    print(f'Done making predictions! Storing them in {args.predictions_file}')
    preds = np.array(preds)
    np.savetxt(args.predictions_file, preds, fmt='%d')


if __name__ == '__main__':
    ARGS = get_args()
    if ARGS.mode == 'train':
        train(ARGS)
    elif ARGS.mode == 'predict':
        test(ARGS)
    else:
        print(f'Invalid mode: {ARGS.mode}! Must be either "train" or "predict".')
