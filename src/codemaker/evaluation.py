"""Functions to compute and preview nearest neighbors"""

import numpy as np
from random import Random
# TODO: move the matplotlib dependent code somewhere else
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scikits.learn.neighbors import Neighbors


def preview(filenames, rows=1):
    for i, f in enumerate(filenames):
        sp = plt.subplot(rows, len(filenames), i + 1)
        sp.imshow(np.flipud(mpimg.imread(f)))
    plt.show()


def preview_nearest(ref_idx, codes, filenames, n=5):
    if not isinstance(codes, list):
        codes = [codes]
    neighbors = Neighbors(k=n).fit(codes)
    for i, code_set in enumerate(codes):
        distances, nearest = neighbors.kneighbors(code_set[ref_idx])
        for j, f in enumerate([filenames[k] for k, _ in nearest]):
            columns = len(nearest) + 1
            rows = len(codes)
            sp = plt.subplot(rows, columns, (i * columns) + j + 1)
            sp.imshow(np.flipud(mpimg.imread(f)))


def local_match(data, code, query_size=10, ratio=1.0, seed=0):
    """Estimate the average accuracy of knn queries."""
    n_samples = data.shape[0]
    n_subset = int(ratio * n_samples)
    accuracies = np.zeros(n_subset)
    data_neighbors = Neighbors(k=query_size + 1).fit(data)
    code_neighbors = Neighbors(k=query_size + 1).fit(code)
    rng = Random(seed)

    for i, j in enumerate(rng.sample(xrange(n_samples), n_subset)):
        _, knn_data = data_neighbors.kneighbors(data[j])
        knn_data = knn_data[1:]
        _, knn_code = code_neighbors.kneighbors(code[j])
        knn_code = knn_code[1:]
        accuracies[i] = float(len(set(knn_data) & set(knn_code))) / query_size

    return accuracies.mean()


def pairwise_distances(data, code, seed=0, ax=None, title=None):
    """Utility to scatter plot random pairwise distances in data vs code

    Return random pairwise distance in data space, matching distances in code
    space and the Pearson correlation between both.
    """
    rng = Random(seed)
    indices = range(len(data))
    rng.shuffle(indices)
    data = data[indices]
    code = code[indices]

    d_data = np.sqrt(np.sum((data[1:] - data[:-1]) ** 2, axis=1))
    d_code = np.sqrt(np.sum((code[1:] - code[:-1]) ** 2, axis=1))

    if ax is not None:
        ax.scatter(d_data, d_code)
        ax.set_xlabel("Pairwise distances in data space")
        ax.set_ylabel("Pairwise distances in code space")
        if title is not None:
            ax.set_title(title)

    return d_data, d_code, np.corrcoef(d_data, d_code)[0][1]


