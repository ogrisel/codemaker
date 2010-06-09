from codemaker.embedding import compute_embedding
from codemaker.evaluation import Neighbors, local_match
import numpy as np
from nose.tools import *
from scikits.learn import datasets
import random


def test_compute_embedding(check_asserts=True, **kw):
    np.random.seed(0)
    random.seed(0)

    # sample data from the digits 8x8 pixels dataset
    digits = datasets.load_digits()
    data = digits.data
    n_samples, n_features = data.shape

    # right now this does not work well for very low dim (need to stack
    # autoencoders along with the local structure preserving penalty to
    # objective function)
    low_dim = 2

    # compute an embedding of the data
    code, encoder = compute_embedding(data, low_dim, **kw)
    assert_equal(code.shape, (n_samples, low_dim))

    # compare nearest neighbors
    score = local_match(data, code, query_size=50, ratio=0.1, seed=0)
    if check_asserts:
        assert_almost_equal(score, 0.25, 2)

    return score, digits, code, encoder


if __name__ == "__main__":
    import matplotlib
    import pylab as pl

    score, digits, code, encoder = test_compute_embedding(
        check_asserts=False, epochs=5000, learning_rate=0.0001, batch_size=5)
    print "match score: ", score

    knn_data = Neighbors(k=5).fit(digits.data, digits.target)
    accuracy_data = (knn_data.predict(digits.data) == digits.target).mean()
    print "accuracy of knn on data:", accuracy_data

    knn_code = Neighbors(k=5).fit(code, digits.target)
    accuracy_code = (knn_code.predict(code) == digits.target).mean()
    print "accuracy of knn on code:", accuracy_code
    print "ratio knn code / knn data:", accuracy_code / accuracy_data

    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = code[:, 0].min() - .1, code[:, 0].max() + .1
    y_min, y_max = code[:, 1].min() - .1, code[:, 1].max() + .1
    h = .02 # step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn_code.predict(np.c_[xx.ravel(), yy.ravel()])

    # plot the prediction results a color plot
    Z = Z.reshape(xx.shape)
    pl.set_cmap(pl.cm.Paired)
    pl.pcolormesh(xx, yy, Z)

    # plot the training point
    pl.scatter(code[:,0], code[:,1], c=digits.target)
#    codes_by_digits = [[] for _ in range(10)]
#    for c_i, i in zip(code, digits.target):
#        codes_by_digits[i].append(c_i)
#
#    colors = pl.cm.Paired(np.arange(10))
#    handles = []
#    for i, c_i in enumerate(codes_by_digits):
#        c_i = np.asarray(c_i)
#        color = colors[i] * np.ones((c_i.shape[0], 4))
#        handles.append(pl.scatter(c_i[:,0], c_i[:,1], c=color))
#    pl.legend(handles, [str(i) for i in range(10)])

    pl.axis('tight')
    pl.title('KNN classification in the 2D code space')
    pl.show()

