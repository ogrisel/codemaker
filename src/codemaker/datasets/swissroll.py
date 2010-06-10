"""Generalization of the swissroll maniforld learning example to N dim"""

from math import cos
from math import sin

import numpy as np


def load(n_samples=1000, n_features=3, rotate=True, n_turns=2, seed=0):
    """Generate spiral curve dataset on the first 2 dims

    The third dim is fill uniformly. The remaining dims are left to zeros
    (before random rotation).

    The resulting dataset hence holds a instrically 2 dim manifold (a spiralling
    band) embedded in an arbitrarly higher dimensional space.

    Parameters
    ----------

    n_samples : number of sample to generate

    n_features : total number of dimension including the first two that include
                 the actual spiral data

    n_turns : number of rotations (times 2 pi) for the spiral manifold

    Returns
    -------

    data : an array of shape (n_samples, n_features) embedding the generated
           hyber-swissroll

    t_span : the "time steps" values used for the generation of the individual
             samples, a linear scale from 0.0 to 1.0

    """

    assert n_features >= 3

    data = np.zeros((n_samples, n_features))
    t_span = np.linspace(0, 1, num=n_samples)

    # generate the 2D spiral data driven by a 1d parameter t
    max_rot = n_turns * 2 * np.pi
    data[:, 0:2] = np.asarray([[t * cos(t * max_rot), t * sin(t * max_rot)]
                               for t in t_span])

    # fill the third dim with the uniform band of width [-1, 1]
    rng = np.random.RandomState(seed)
    data[:, 2] = rng.uniform(-1, 1.0, n_samples)

    if rotate:
        # rotate the randomly along all axes to avoid trivial orthogonal projections
        # achived by feature selection to recover the original spiral
        for i in range(1, n_features - 1):
            rotation = np.identity(n_features)
            angle = rng.uniform(0, np.pi / 2)
            c, s = cos(angle), sin(angle)
            rotation[i][i], rotation[i + 1][i + 1] = c, c
            rotation[i][i + 1], rotation[i + 1][i] = s, -s
            data = np.dot(data, rotation)

    return data, t_span


if __name__ == "__main__":
    import pylab as pl

    X, y = load(rotate=False)

    # plot the 2d projection of the spiral
    pl.scatter(X[:, 0], X[:, 1], c=range(len(y)), cmap=pl.cm.spectral)
    pl.show()

    # plot the unrolled manifold embedded in the data
    pl.scatter(y, X[:, 2], c=range(len(y)), cmap=pl.cm.spectral)
    pl.show()


