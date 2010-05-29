from codemaker.embedding import compute_embedding
from codemaker.evaluation import Neighbors, local_match
import numpy as np
from nose.tools import *
from scikits.learn import datasets
import random

def test_compute_embedding():
    np.random.seed(0)
    random.seed(0)

    # sample data from the digits 8x8 pixels dataset
    digits = datasets.load_digits()
    data = digits.data
    n_samples, n_features = data.shape

    # right now this does not work well for very low dim (need to stack
    # autoencoders along with the local structure preserving penalty to
    # objective function)
    low_dim = 20

    # compute an embedding of the data
    code, encoder = compute_embedding(data, low_dim, epochs=15,
                                      learning_rate=0.0005)
    assert_equal(code.shape, (n_samples, low_dim))

    # compare nearest neighbors
    k = 50
    score = local_match(data, code, query_size=k, ratio=0.1, seed=0)
    assert_almost_equal(score, 0.51, 2)

