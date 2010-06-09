import numpy as np
from scikits.learn.glm import ElasticNetPath
from scikits.learn.glm import LassoPath
# TODO: implement a variant using LassoLARS when ready


def sparse_encode(D, data, callback=None, rho=1.0, n_alphas=5, eps=1e-2):
    """Given dictionary D, find sparse encoding of vectors in data

    Encoding is performed using coordinate descent of the elastic net
    regularized least squares.
    """
    D = np.asanyarray(D, dtype=np.double)
    data = np.asanyarray(data, dtype=np.double)
    data = np.atleast_2d(data)
    encoded = np.zeros((data.shape[0], D.shape[1]), dtype=np.double)

    for i, code in enumerate(data):
        # TODO: parallelize me with multiprocessing!
        if rho == 1.0:
            clf = LassoPath(n_alphas=n_alphas, eps=eps).fit(D, code)
        else:
            clf = ElasticNetPath(
                n_alphas=n_alphas, eps=eps, rho=rho).fit(D, code)
        encoded[i][:] = clf.coef_
        if callback is not None:
            callback(i)
    return encoded


