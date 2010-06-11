"""Generalization of the swissroll maniforld learning example to N dim"""

from math import cos
from math import sin

import numpy as np


def load(n_samples=1000, n_features=3, rotate=True, n_turns=1.5, seed=0,
         radius=1.0, hole=False):
    """Generate spiral curve dataset on the first 2 dims

    The third dim is fill uniformly. The remaining dims are left to zeros
    (before random rotation).

    The resulting dataset hence holds a instrically 2 dim manifold (a spiralling
    band) embedded in an arbitrarly higher dimensional space.

    Parameters
    ----------

    n_samples : number of sample to generate

    n_features : total number of dimension including the first two that include
                 the actual spiral data (when not rotated)

    n_turns : number of rotations (times 2 pi) for the spiral manifold

    rotate : boolean flag to rotate randomly the spiral iteratively on all
             dimensions

    hole : boolean flag to dig a rectangular hole in the middle of the roll
           band

    Returns
    -------

    data : an array of shape (n_samples, n_features) embedding the generated
           hyber-swissroll

    manifold : an array of size (n_samples, 2) that contains the unrolled
               manifold

    """

    assert n_features >= 3
    rng = np.random.RandomState(seed)

    data = np.zeros((n_samples, n_features))
    t = rng.uniform(low=0, high=1, size=n_samples)

    # generate the 2D spiral data driven by a 1d parameter t
    max_rot = n_turns * 2 * np.pi
    data[:, 0:2] = np.asarray([[radius * t_i * cos(t_i * max_rot),
                                radius * t_i * sin(t_i * max_rot)]
                               for t_i in t])

    # fill the third dim with the uniform band of width [-1, 1]
    data[:, 2] = rng.uniform(-1, 1.0, n_samples)

    # copy the manifold data before performing the rotation
    manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()

    if hole:
        z = data[:, 2]
        indices = np.where(((0.3 > t) | (0.7 < t)) | ((-0.3 > z) | (0.3 < z)))
        data = data[indices]
        manifold = manifold[indices]

    if rotate:
        # rotate the randomly along all axes to avoid trivial orthogonal projections
        # achived by feature selection to recover the original spiral

        # WARNING: this is not a real random rotation matrix, the curse of
        # dimensionality is making the last features very small w.r.t. the first
        # dims: how to mitigate this? should we normalize / rescale?
        axis = range(1, n_features - 1)
        for i in axis:
            rotation = np.identity(n_features)
            angle = rng.normal(np.pi / 4, np.pi / 32)
            c, s = cos(angle), sin(angle)
            rotation[i][i], rotation[i + 1][i + 1] = c, c
            rotation[i][i + 1], rotation[i + 1][i] = s, -s
            data = np.dot(data, rotation)

    return data, manifold


if __name__ == "__main__":
    import pylab as pl
    pl.clf()

    data, manifold = load(n_features=4, n_turns=1.5, rotate=True, hole=True)

    # plot the 2d projection of the first two axes
    colors = manifold[:, 0]
    pl.subplot(121).scatter(data[:, 0], data[:, 1], c=colors)
    # TODO: find a way to make scatter 3D work with the same colors instead of
    # using an arbitrary 2D projection

    # plot the unrolled manifold embedded in the data
    pl.subplot(122).scatter(manifold[:, 0], manifold[:, 1], c=colors)
    pl.show()


