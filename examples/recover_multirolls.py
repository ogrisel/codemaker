"""Train an encoder to extract a 2D swissroll manifolds from higher dim data"""
import numpy as np
import pylab as pl
import time

from codemaker.datasets import multirolls
from codemaker.embedding import SDAEmbedder
from codemaker.evaluation import Neighbors, local_match

pl.clf()
np.random.seed(0)

n_features = 100
n_samples = 5000
n_manifolds = 100

print "Generating %d embedded swissrolls with n_features=%d, n_samples=%d" % (
    n_manifolds, n_features, n_samples)

data, manifolds, colors = multirolls.load(
    n_features=n_features,
    n_samples=n_samples,
    n_manifolds=n_manifolds,
)

# compute a baseline evaluation of the manifolds (ground truth)
stacked_manifolds = np.vstack([m + [10 * i, 0]
                               for i, m in enumerate(manifolds)])
score = local_match(data, stacked_manifolds, query_size=50, ratio=1, seed=0)
print "kNN score match manifolds/data (ground truth):", score

# compute the score of a projection
score = local_match(data, data[:,(0,1)], query_size=50, ratio=1, seed=0)
print "kNN score match projection/data (baseline):", score

# reshuffle the data since stochastic gradient descent assumes I.I.D. samples
perm = np.random.permutation(data.shape[0])
data, colors = data[perm], colors[perm]

# build model to extract the manifolds and learn a mapping / encoder to be able
# to reproduce this on test data
embedder = SDAEmbedder((n_features, 30, 10, 2), noise=0.1,
                       sparsity_penalty=0.0, learning_rate=0.1, seed=0)

random_code = embedder.encode(data)
score = local_match(data, random_code, query_size=50, ratio=1, seed=0)
print "kNN score match random code/data:", score

print "Training encoder to unroll the embedded data..."
start = time.time()
embedder.pre_train(data, epochs=500, batch_size=10)
print "done in %ds" % (time.time() - start)
code = embedder.encode(data)

score = local_match(data, code, query_size=50, ratio=1, seed=0)
print "kNN score match code/data:", score

# plot some 2D projections
sp = pl.subplot(221)
sp.scatter(data[:, 0], data[:, 1], c=colors)
sp.set_title("Projection to axis (x_0, x_1)")

sp = pl.subplot(222)
sp.scatter(data[:, 3], data[:, 7], c=colors)
sp.set_title("Projection to axis (x_3, x_7)")

# plot the learned 2D embedding recovered by untrained SDAE model
sp = pl.subplot(223)
sp.scatter(random_code[:, 0], random_code[:, 1], c=colors)
sp.set_title("2D manifold recovered by random projections")

# plot the learned 2D embedding recovered by the SDAE model
sp = pl.subplot(224)
sp.scatter(code[:, 0], code[:, 1], c=colors)
sp.set_title("2D manifold recovery by codemaker")

pl.show()

