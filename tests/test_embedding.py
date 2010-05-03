from codemaker.embedding import compute_embedding
from codemaker.evaluation import find_nearest
import numpy as np
from nose.tools import *
from scikits.learn import datasets

def test_compute_embedding():

    # sample data from the digits 8x8 pixels dataset
    digits = datasets.load_digits()
    data = digits.data
    n_samples, n_features = data.shape

    # right now this does not work for very low dim (need to stack autoencoders
    # along with the local structure preserving penalty to objective function)
    low_dim = 10

    # compute an embedding of the data
    np.random.seed(0)
    code, encoder = compute_embedding(data, low_dim, epochs=25,
                                      learning_rate=0.0001)
    assert_equal(code.shape, (n_samples, low_dim))

    # compare nearest neighbors
    ref_idx = 41

    knn_data = [idx for idx, dist in find_nearest(data[ref_idx], data, n=50)]
    knn_code = [idx for idx, dist in find_nearest(code[ref_idx], code, n=50)]

    print find_nearest(data[ref_idx], data)
    print [digits.target[idx] for idx in knn_data]

    print find_nearest(code[ref_idx], code)
    print [digits.target[idx] for idx in knn_code]

    assert_equals(len(set(knn_data) & set(knn_code)), 27)


