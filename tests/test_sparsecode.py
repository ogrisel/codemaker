from codemaker.sparse import sparse_encode
from codemaker.sparse import SparseEncoder
import numpy as np
from nose.tools import *
from numpy.testing import assert_almost_equal

def test_random_sparsecode():

    n_samples = 500
    n_features = 50
    n_basis = 30

    np.random.seed(0)

    # sample data from centered unit variance guaussian process
    x = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    # lets take the first samples as dictionary
    D = x[:n_basis].T

    # encode all samples according to D
    encoded = sparse_encode(D, x)
    assert_equal(encoded.shape, (n_samples, n_basis))

    # check that the first sample are encoded using a single component (trivial
    # sparse code)
    for i in xrange(n_basis):
        expected_code = np.zeros(n_basis)
        expected_code[i] = 1.0
        assert_almost_equal(encoded[i], expected_code, decimal=2)

    # check that the remaining samples could also be encoding with a sparse code
    avg_nb_nonzeros = (encoded[n_basis:] != 0).sum(axis=1).mean()
    assert avg_nb_nonzeros <= 10


def test_parallel_sparse_code():
    n_samples = 50
    n_features = 20
    n_basis = 10

    np.random.seed(0)

    # sample data from centered unit variance guaussian process
    x = np.random.normal(size=(n_samples, n_features))

    # lets take the first samples as dictionary
    D = x[:n_basis].T

    encoded_1 = sparse_encode(D, x, max_features=3)
    encoded_2 = SparseEncoder(D, max_features=3, n_cores=2)(x)

    assert_almost_equal(encoded_1, encoded_2)



