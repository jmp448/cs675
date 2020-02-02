import numpy as np
import scipy.stats
import scipy.spatial.distance as dist
import scipy.sparse
from collections import defaultdict


def find_neighbors(k, neighborhood, new=None):
    if new is None:
        dists = dist.cdist(neighborhood, neighborhood)
        np.fill_diagonal(dists, np.inf)  # don't count self as neighbor
        nearest = np.argsort(dists, axis=1)[:, :k]
    else:
        dists = dist.cdist(new, neighborhood)
        nearest = np.argsort(dists, axis=1)[:, :k]
    return nearest


def normalize(X):
    X_avg = np.average(X, axis=0)
    X_sd = np.sqrt(np.var(X, axis=0))

    X_norm = np.zeros(X.shape)
    for m in range(X.shape[1]):
        if X_sd[m] == 0:
            X_norm[:, m] = X[:, m] - np.full(X.shape[0], X_avg[m])
        else:
            X_norm[:, m] = (X[:, m] - np.full(X.shape[0], X_avg[m])) / np.full(X.shape[0], X_sd[m])
    return X_norm


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
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

class PCA(Model):

    def __init__(self, X, target_dim):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]
        self.target_dim = target_dim
        self.W = None

    def fit(self, X):
        X_norm = normalize(X)
        X_cov = np.cov(np.transpose(X_norm))
        [eigval, eigvec] = scipy.linalg.eig(X_cov)
        self.W = eigvec[:, np.argsort(eigval)[::-1][:self.target_dim]]
        Y = np.matmul(X, self.W)
        return Y


class LLE(Model):

    def __init__(self, X, target_dim, lle_k):
        self.num_x = X.shape[0]
        self.x_dim = X.shape[1]

        self.target_dim = target_dim
        self.k = lle_k

    def get_weights(self, X, X_neighbors):
        W = np.zeros([self.num_x, self.num_x])
        for i in range(self.num_x):
            Z = np.zeros([self.x_dim, self.k])
            for k in range(self.k):
                Z[:, k] = X[i, :]-X[X_neighbors[i, k], :]
            C = np.matmul(np.transpose(Z), Z)
            eps = 0.001*np.trace(C)
            C = np.add(C, eps)
            w = np.linalg.solve(C, np.ones([C.shape[0], 1]))
            for k in range(self.k):
                W[i, X_neighbors[i, k]] = w[k]/sum(w)
        return W

    def fit(self, X):
        X = normalize(X)
        X_neighbors = find_neighbors(self.k, X)
        W = self.get_weights(X, X_neighbors)
        M = np.matmul(np.transpose(np.identity(X.shape[0])-W), np.identity(X.shape[0])-W)
        [eigval, eigvec] = scipy.sparse.linalg.eigsh(M, k=self.target_dim + 1, sigma=0.0)
        bottom_eigvecs = eigvec[:, eigval > np.finfo(float).eps]
        return bottom_eigvecs


class KNN(Model):

    def __init__(self, k):
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X, y):
        self.data = X
        self.labels = y

    def predict(self, X):
        neighbors = find_neighbors(self.k, self.data, X)
        neighborhood_labels = [self.labels[x] for x in neighbors]
        labels = scipy.stats.mode(neighborhood_labels, axis=1)[0]
        return labels


def notebook():
    import numpy as np
    from numpy.linalg import norm

    def load_vecs(path):
        """ Loads in word vectors from path.
        Will return a dictionary of word to index, and a matrix of vectors (each word is a row)
        """
        vecs = []
        w2i = {}

        with open(path, 'r') as inp:
            for line in inp.readlines():
                line = line.strip().split()
                word = str(line[0])
                w2i[word] = len(vecs)
                vecs.append(np.array(line[1:], dtype=float))
            vecs = np.array([v / norm(v) for v in vecs])
            print(f'Read in {vecs.shape[0]} words of size {vecs.shape[1]}')
        return w2i, vecs

    # This might take a little bit to run!
    indxr, wembs = load_vecs('data/glove.6B.100d.txt')

    def similarity(v1, v2):
        return np.dot(v1, v2)

    def bump(l, newb, pos, n):
        if len(l) < n:
            return np.concatenate((l[:pos], [newb], l[pos:]))
        else:
            return np.concatenate((l[:pos], [newb], l[pos:n - 1]))

    def find_neighbors(k, neighborhood, new):
        sims = [similarity(new, neighborhood[0])]
        nearest = np.zeros(1, dtype=int)
        for n in range(1, len(neighborhood)):
            for i in range(len(sims)):
                this_sim = similarity(neighborhood[n], new)
                if this_sim > sims[i]:
                    sims = bump(sims, this_sim, i, k)
                    nearest = bump(nearest, n, i, k)
                    break
        return nearest

    def find_words(neighbors):
        words = []
        for ind in neighbors:
            words.append(list(indxr.keys())[ind])
        return words

    def analogy(n, word1, word2, word3):
        emb1 = wembs[indxr[word1]]
        emb2 = wembs[indxr[word2]]
        emb3 = wembs[indxr[word3]]
        emb4 = emb2 - emb1 + emb3
        neighbors = find_neighbors(n + 3, wembs, emb4)
        top_words = find_words(neighbors)
        for w in [word1, word2, word3]:  # filter out any repeated words - boring analogies
            if top_words.__contains__(w):
                top_words.remove(w)
        return top_words[:n]

    print(analogy(10, "man", "woman", "king"))

    from numpy.linalg import svd
    gender_pairs = [('she', 'he'), ('her', 'his'), ('woman', 'man'), ('mary', 'john'), ('herself', 'himself'),
                    ('daughter', 'son'), ('mother', 'father'), ('gal', 'guy'), ('girl', 'boy'), ('female', 'male')]

    # Copy biased embeddings into a new object.
    debiased_wembs = np.copy(wembs)

    def build_gender_subspace(k):
        C = np.zeros([100, 100])
        for pair in gender_pairs:
            emb1 = wembs[indxr[pair[0]]]
            emb2 = wembs[indxr[pair[1]]]
            C = np.add(C, np.add(np.multiply(emb1 - emb2, np.transpose(emb1 - emb2)),
                                 np.multiply(emb2 - emb1, np.transpose(emb2 - emb1))))
        [u, _, _] = svd(C)
        return u[:, :k]

    B = build_gender_subspace(10)

    def debias_word(word):
        emb = wembs[indxr[word]]
        print(emb.shape)
        wb = np.zeros(emb.shape)
        for j in range(B.shape[1]):
            wb = np.add(wb, np.multiply(np.dot(emb, B[:, j]), B[:, j]))
        wub = np.subtract(emb, wb) / np.linalg.norm(np.subtract(emb, wb))
        debiased_wembs[indxr[word]] = wub
        return debiased_wembs

    word = "doctor"
    w1 = wembs[indxr[word]]
    debias_word("doctor")
    w2 = debiased_wembs[indxr[word]]
    w3 = wembs[indxr[word]]
    print(w1 == w2)


if __name__ == "__main__":
    notebook()
    # a1 = np.ndarray(3, buffer=np.array([3, 1, 2]), dtype=int)
    # a2 = np.ndarray(3, buffer=np.array([2, 1, 1]), dtype=int)
    # a3 = np.ndarray(3, buffer=np.array([4, 2, 5]), dtype=int)
    # a4 = a1 - a2 + a3
    # print(a4)
