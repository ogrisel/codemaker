from codemaker.embedding import compute_embedding
from codemaker.evaluation import Neighbors
import numpy as np
from nose.tools import *
from scikits.learn import datasets

def test_compute_embedding():

    # sample data from the digits 8x8 pixels dataset
    digits = datasets.load_digits()
    data = digits.data
    n_samples, n_features = data.shape
    print data.shape

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
    data_neighbors = Neighbors(k=50).fit(data, digits.target)
    code_neighbors = Neighbors(k=50).fit(code, digits.target)

    _, knn_data = data_neighbors.kneighbors(data[ref_idx])
    _, knn_code = code_neighbors.kneighbors(code[ref_idx])

    print knn_data
    print [digits.target[idx] for idx in knn_data]

    print knn_code
    print [digits.target[idx] for idx in knn_code]

    assert_equals(len(set(knn_data) & set(knn_code)), 29)

