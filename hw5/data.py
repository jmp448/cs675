""" Simple data loader """
import numpy as np

def load(dpath, split):
    if split == 'test' or split == 'dev':
        d = np.load(f'{dpath}/mnist-{split}.npy')
        l = None
    else:
        d = np.load(f'{dpath}/mnist-{split}.npy')
        l = np.load(f'{dpath}/labels-mnist-{split}.npy')
    return d, l
