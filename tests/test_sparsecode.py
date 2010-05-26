from codemaker.sparse import sparse_encode
from codemaker.evaluation import find_nearest
import numpy as np
from nose.tools import *

def test_random_sparsecode():

    n_samples = 100
    n_features = 50
    n_basis = 30

    np.random.seed(0)

    # sample data from centered unit variance guaussian process
    x = np.random.normal(
        loc=0.0, scale=1.0, size=n_samples * n_features).reshape(
            n_samples, n_features)

    # lets take the first samples as dictionary
    D = x[:n_basis].T

    # encode all samples according to D
    encoded = sparse_encode(D, x)
    assert_equal(encoded.shape, (n_samples, n_basis))

    # check that the first sample are encoded using a single component (trivial
    # sparse code)
    avg_density = (encoded[:n_basis] != 0).sum(axis=1).mean()
    assert_almost_equal(avg_density, 1.0, 0.01)

    for i in xrange(n_basis):
        for j in xrange(n_basis):
            if i == j:
                assert_almost_equal(encoded[i][j], 1.0, 0.05)
            else:
                assert_almost_equal(encoded[i][j], 0.0, 0.01)

    # check that the remaining samples could also be encoding with a sparse code
    avg_nb_nonzeros = (encoded[n_basis:] != 0).sum(axis=1).mean()
    assert_almost_equal(avg_nb_nonzeros, 4, 0)


