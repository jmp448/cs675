import argparse as ap
import numpy as np

p = ap.ArgumentParser()
p.add_argument('--labeled', type=str, help='.npy labels file')
p.add_argument('--predicted', type=str, help='.txt predictions file')
args = p.parse_args()

count = 0
correct = 0
golds = np.load(args.labeled)
preds = open(args.predicted, 'r')

for prediction in preds.readlines():
    prediction = int(prediction.strip())
    gold = golds[count]

    if int(gold) == int(prediction):
        correct += 1
    count += 1

acc = float(correct) * 100 / count
print(f"Accuracy is {acc}%")
