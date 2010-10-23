from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from scikits.learn.glm import LassoCV
from codemaker.utils import WorkerInterrupt

def sparse_encode(D, data, callback=None, n_alphas=3):
    """Given dictionary D, find sparse encoding of vectors in data

    """
    D = np.asanyarray(D, dtype=np.double)
    data = np.asanyarray(data, dtype=np.double)
    data = np.atleast_2d(data)

    # TODO: use a smart sparse representation instead
    encoded = np.zeros((data.shape[0], D.shape[1]), dtype=np.double)

    for i, code in enumerate(data):
        clf = LassoCV(n_alphas=n_alphas).fit(D, code, fit_intercept=False)
        encoded[i][:] = clf.coef_

        if callback is not None:
            callback(i)
    return encoded


# ugly wrapper for multiprocessing robustness
def _job_fun(args):
    try:
        D, x = args
        return sparse_encode(D, x)
    except KeyboardInterrupt:
        raise WorkerInterrupt()


class SparseEncoder(object):
    """Compute sparse codes representing data w.r.t. to a fixed dictionary

    Parameters
    ----------

    dictionary : a 2D numpy array of dim (n_basis, n_features) where n_features
                 is the number of features of the input space. Dictionary can be
                 randomly generated, randomly sample from the dataset, extracted
                 by a PCA of the dataset or learned using dictionary learning
                 algorithms (not implemented yet).

    n_cores : number of cores to use to parallelize the sparse coding
              processing. Use None for autodetection (default), 1 to disable
              parallelism.

    """

    def __init__(self, dictionary, n_cores=None):
        self.dictionary = np.atleast_2d(dictionary)
        self.n_cores = cpu_count() if n_cores is None else n_cores
        if self.n_cores > 1:
            self._pool = Pool(processes=n_cores)

    def __call__(self, data):
        """Encode data against the reference dictionary"""
        data = np.atleast_2d(data)

        if self.n_cores == 1:
            # single process call of the sparse_encode function
            return sparse_encode(self.dictionary, data)
        else:
            # split data in almost equals parts for parallelization
            job_args = [(self.dictionary, chunk)
                        for chunk in np.array_split(data, self.n_cores * 2)]

            return np.vstack(self._pool.map(_job_fun, job_args))


