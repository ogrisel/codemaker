import numpy as np
from scikits.learn.glm import LeastAngleRegression


def sparse_encode(D, data, callback=None, max_features=10):
    """Given dictionary D, find sparse encoding of vectors in data

    Encoding is performed using Least Angle Regression with a predifined maximum
    number of non-zero components set to 10 by default.
    """
    D = np.asanyarray(D, dtype=np.double)
    data = np.asanyarray(data, dtype=np.double)
    data = np.atleast_2d(data)
    encoded = np.zeros((data.shape[0], D.shape[1]), dtype=np.double)

    for i, code in enumerate(data):
        # TODO: parallelize me!
        clf = LeastAngleRegression().fit(
            D, code, normalize=False, intercept=False,
            max_features=max_features)

        # threshold near zero values due to an implementation detail in the
        # current LAR implementation
        encoded[i][:] = np.where(abs(clf.coef_) < 1e-10,
                               np.zeros(clf.coef_.shape),
                               clf.coef_)

        if callback is not None:
            callback(i)
    return encoded


