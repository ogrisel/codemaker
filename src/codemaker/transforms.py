"""Random transforms"""

import numpy as np

def random_basis(data, rng=None):
    """Apply the same random basis change to each sample in data

    Dot product values in the new basis remain unchanged.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples, n_features = data.shape

    # generate a random orthogonal matrix in dimension n_features by taking the
    # QR decomposition of a random square matrix
    q, _ = np.linalg.qr(rng.uniform(size=(n_features, n_features)))
    return np.dot(data, q)


def random_rotate(data, rng=None):
    """Apply the same random rotation along all axes to each sample in data"""
    if rng is None:
        rng = np.random.RandomState()

    n_samples, n_features = data.shape

    # generate a random orthogonal matrix in dimension n_features by taking the
    # QR decomposition of a random square matrix
    q, _ = np.linalg.qr(rng.uniform(size=(n_features, n_features)))

    # make it a real rotation by ensuring det(q) = 1 (do we really care?)
    q[0] *= np.linalg.det(q)
    return np.dot(data, q)


def random_project(data, target_dim=2, rng=None):
    """Random project each sample of data to sub vector space of target_dim"""
    return random_basis(data, rng=rng)[:,0:target_dim]

