"""Buid a data set with several 2D swissroll manifolds randomly embedded"""

import numpy as np

from codemaker.datasets.swissroll import load as load_one
from codemaker.datasets.swissroll import random_rotate


def load(n_samples=10000, n_features=10, n_manifolds=10, seed=0):
    """Generate a multi spiral bands dataset randomly in many dims

    The resulting dataset hence holds a set of instrically 2 dim manifolds
    (spiralling band) embedded in an arbitrarly higher dimensional space.

    Parameters
    ----------

    n_samples : number of sample to generate

    n_features : total number of dimension

    n_manifolds : number of manifolds to generate

    seed : reproducible pseudo random dataset generation


    Returns
    -------

    data : an array of shape (n_samples, n_features) embedding the generated
           hyber-swissroll

    manifolds : an array of size (n_manifolds, n_samples / n_manifolds, 2)
                that contains the unrolled manifolds

    t : the array of parameters value which is the intrinsic dimension
        common to all manifolds

    """

    assert n_features >= 3
    rng = np.random.RandomState(seed)

    data = []
    manifolds = []
    for i in xrange(n_manifolds):
        n_samples_m = n_samples / n_manifolds
        if i < n_samples % n_manifolds:
            n_samples_m += 1

        data_m, manifold_m = load_one(
            n_samples=n_samples_m,
            n_features=n_features,
            n_turns=rng.uniform(0.2, 2),
            radius=rng.uniform(0.5, 2),
            hole=rng.uniform()> 0.5,
            rotate=False,
            seed=seed,
        )
        data_m[:, 0] += rng.uniform(-2, 2)
        data_m[:, 1] += rng.uniform(-2, 2)
        data_m[:, 2] += rng.uniform(-2, 2)
        data_m = random_rotate(data_m, rng)

        data.append(data_m[:, rng.permutation(n_features)])
        manifolds.append(manifold_m)

    data = random_rotate(np.vstack(data), rng)

    t_without_holes = np.hstack([m[:, 0] for m in manifolds])

    return data, manifolds, t_without_holes


if __name__ == "__main__":
    import pylab as pl
    pl.clf()

    data, manifolds, t = load(n_samples=4000, n_features=10, n_manifolds=8)

    # plot some 2D projections
    pl.subplot(221).scatter(data[:, 0], data[:, 1], c=t)
    pl.subplot(222).scatter(data[:, 4], data[:, 6], c=t)

    # plot some of the unrolled manifolds embedded in the data
    manifold = manifolds[0]
    colors = manifold[:, 0]
    pl.subplot(223).scatter(manifold[:, 0], manifold[:, 1], c=colors)

    manifold = manifolds[4]
    colors = manifold[:, 0]
    pl.subplot(224).scatter(manifold[:, 0], manifold[:, 1], c=colors)

    pl.show()


