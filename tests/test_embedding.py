from codemaker.embedding import SDAEmbedder
from codemaker.transforms import random_project
from codemaker.evaluation import Neighbors, local_match
import numpy as np
from nose.tools import *
from scikits.learn import datasets
import random


def test_compute_embedding(check_asserts=True):
    np.random.seed(0)
    random.seed(0)

    # sample data from the digits 8x8 pixels dataset
    digits = datasets.load_digits()
    data = digits.data
    n_samples, n_features = data.shape
    low_dim = 2

    # baseline score using a random 2D projection
    projection = random_project(data, target_dim=2, rng=np.random)
    score = local_match(data, projection, query_size=50, ratio=0.1, seed=0)
    assert_almost_equal(score, 0.12, 2)

    # compute an embedding of the data
    embedder = SDAEmbedder((n_features, 20, low_dim), noise=0.1,
                           sparsity_penalty=0.0, learning_rate=0.1, seed=0)
    embedder.pre_train(data, epochs=500, batch_size=50)

    code = embedder.encode(data)
    assert_equal(code.shape, (n_samples, low_dim))

    # compare nearest neighbors
    score = local_match(data, code, query_size=50, ratio=0.1, seed=0)

    # TODO: score is bad since we only do pretraining right now: need to include
    # SNE objective function too
    assert_almost_equal(score, 0.03, 2)



