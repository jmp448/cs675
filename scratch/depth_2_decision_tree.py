import numpy as np


class Tree:
    def __init__(self, f, left=None, right=None):
        if left is None:
            self.f = f
        else:
            self.left = left
            self.right = right
            self.f = f+1


def BuildDecisionTree(data, labels):
    if np.var(labels) == 0:
        return Tree(labels[0])
    else:
        f = find_best_feature(data, labels)
        right_samples = []
        left_samples = []
        for s in range(len(labels)):
            if data[s, f] == 0:
                left_samples.append(s)
            else:
                right_samples.append(s)
        right_data = np.delete(data, left_samples, 0)
        right_labels = np.delete(labels, left_samples, 0)
        left_data = np.delete(data, right_samples, 0)
        left_labels = np.delete(labels, right_samples, 0)

        left = BuildDecisionTree(left_data, left_labels)
        right = BuildDecisionTree(right_data, right_labels)

        return Tree(f, left, right)


def find_best_feature(data, labels):
    nfeats = data.shape[1]
    nsamples = data.shape[0]

    hyx = np.zeros(nfeats)
    for k in range(nfeats):
        probs = np.zeros([2, 2])
        for xi in [0, 1]:
            for yi in [0, 1]:
                for s in range(nsamples):
                    if data[s, k] == xi and labels[s] == yi:
                        probs[xi, yi] += 1 / nsamples
        for xi in [0, 1]:
            if sum(probs[xi, :]) == 0:
                pass
            else:
                for yi in [0, 1]:
                    if probs[xi, yi] == 0:
                        pass
                    else:
                        hyx[k] -= probs[xi][yi] * np.log2(probs[xi, yi] / sum(probs[xi, :]))
    return np.argmin(hyx)

def main():

    data = np.array([[1, 0, 0, 0, 1],
                     [1, 0, 1, 0, 1],
                     [0, 1, 0, 0, 1],
                     [1, 1, 1, 0, 1],
                     [1, 0, 1, 1, 1],
                     [1, 0, 0, 1, 1],
                     [0, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1]])
    labels = np.array([1, 1, 0, 0, 0, 0, 1, 0])

    decision_tree = BuildDecisionTree(data, labels)

    print("First divide on x%d\n" % (decision_tree.f))
    print("If x%d is 0, divide next on x%d\n" % (decision_tree.f, decision_tree.left.f))
    print("Then, if x%d is 0, label is %d\n" % (decision_tree.left.f, decision_tree.left.left.f))
    print("Else, if x%d is 1, label is %d\n" % (decision_tree.left.f, decision_tree.left.right.f))

    print("Now, if x%d is 1, divide next on x%d\n" % (decision_tree.f, decision_tree.right.f))
    print("Then, if x%d is 0, label is %d\n" % (decision_tree.right.f, decision_tree.right.left.f))
    print("Else, if x%d is 1, label is %d\n" % (decision_tree.right.f, decision_tree.right.right.f))


if __name__ == "__main__":
    # main()
    a = [1, 0]
    print(np.var(a))
