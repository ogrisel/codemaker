from multiprocessing import Pool
from multiprocessing import cpu_count
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

    # TODO: use a smart sparse representation instead
    encoded = np.zeros((data.shape[0], D.shape[1]), dtype=np.double)

    for i, code in enumerate(data):
        clf = LeastAngleRegression().fit(
            D, code, normalize=False, intercept=False,
            max_features=max_features)

        # threshold near zero values due to an implementation detail in the
        # current LARS implementation
        encoded[i][:] = np.where(abs(clf.coef_) < 1e-10,
                               np.zeros(clf.coef_.shape),
                               clf.coef_)

        if callback is not None:
            callback(i)
    return encoded


# ugly wrapper for multiprocessing
def _job_fun(args):
    D, x, mf = args
    return sparse_encode(D, x, max_features=mf)


class SparseEncoder(object):
    """Compute sparse codes representing data w.r.t. to a fixed dictionary

    Parameters
    ----------

    dictionary : a 2D numpy array of dim (n_basis, n_features) where n_features
                 is the number of features of the input space. Dictionary can be
                 randomly generated, randomly sample from the dataset, extracted
                 by a PCA of the dataset or learned using dictionary learning
                 algorithms (not implemented yet).

    max_features : maximum number of non zero components in the sparse code

    n_cores : number of cores to use to parallelize the sparse coding
              processing. Use None for autodetection (default), 1 to disable
              parallelism.

    """

    def __init__(self, dictionary, max_features=10, n_cores=None):
        self.dictionary = np.atleast_2d(dictionary)
        self.max_features = max_features
        self.n_cores = cpu_count() if n_cores is None else n_cores
        if self.n_cores > 1:
            self._pool = Pool(processes=n_cores)

    def __call__(self, data, max_features=None):
        """Encode data against the reference dictionary"""
        if max_features is None:
            max_features = self.max_features

        if self.n_cores == 1:
            # single process call of the sparse_encode function
            return sparse_encode(self.dictionary, data, max_features)
        else:
            # split data in almost equals parts for parallelization
            job_args = [(self.dictionary, chunk, max_features)
                        for chunk in np.array_split(data, self.n_cores * 2)]

            return np.vstack(self._pool.map(_job_fun, job_args))


