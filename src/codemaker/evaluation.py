"""Functions to compute and preview nearest neighbors"""

import numpy as np
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

