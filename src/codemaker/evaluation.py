"""Functions to naively compute nearest neighbors"""

import numpy as np
# TODO: move the matplotlib dependent code somewhere else
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def find_nearest(ref, vectors, n=5, max_dist=None, norm='l2', mean=True):
    """Find the n vector indices whose components match ref"""
    if norm == 'l2':
        distances = np.sqrt((vectors - ref) ** 2)
    elif norm == 'l1':
        distances = np.abs(vectors - ref)
    else:
        raise ValueError("unknown norm: " + norm)
    if mean:
        distances = distances.mean(axis=1)
    else:
        distances = distances.sum(axis=1)

    return [(i, distances[i]) for i in distances.argsort()[:n]
            if max_dist is None or distances[i] < max_dist]


def preview(filenames, rows=1):
    for i, f in enumerate(filenames):
        sp = plt.subplot(rows, len(filenames), i + 1)
        sp.imshow(np.flipud(mpimg.imread(f)))
    plt.show()


def preview_nearest(ref_idx, codes, filenames, n=5):
    if not isinstance(codes, list):
        codes = [codes]
    for i, code_set in enumerate(codes):
        nearest = find_nearest(code_set[ref_idx], code_set, n=n)
        for j, f in enumerate([filenames[k] for k, _ in nearest]):
            columns = len(nearest) + 1
            rows = len(codes)
            sp = plt.subplot(rows, columns, (i * columns) + j + 1)
            sp.imshow(np.flipud(mpimg.imread(f)))


