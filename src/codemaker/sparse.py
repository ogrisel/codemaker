import numpy as np
from scikits.learn.glm.coordinate_descent import ElasticNetPath

def sparse_encode(D, data, rho=0.99, callback=None):
    """Given dictionary D, finc sparse encoding of vectors in data

    Encoding is performed using coordinate descent of the elastic net
    regularized least squares.
    """
    D = np.asanyarray(D, dtype=np.double)
    data = np.asanyarray(data, dtype=np.double)
    data = np.atleast_2d(data)
    encoded = np.zeros((data.shape[0], D.shape[1]), dtype=np.double)

    for i, code in enumerate(data):
        # TODO: parallelize me with multiprocessing!
        clf = ElasticNetPath(model_params={"rho": rho}).fit(D, code)
        encoded[i][:] = clf.coef_
        if callback is not None:
            callback(i)
    return encoded


