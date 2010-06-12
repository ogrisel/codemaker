"""Train an encoder to extract a 2D swissroll manifolds from higher dim data"""
import numpy as np
import pylab as pl
import time

from codemaker.datasets import multirolls
from codemaker.embedding import compute_embedding
from codemaker.evaluation import Neighbors, local_match

try:
    import mdp
except ImportError:
    mdp = None

pl.clf()
np.random.seed(0)

n_features = 10
n_samples = 3000
n_manifolds = 5

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
print "Average kNN score match manifolds/data:", score

# reshuffle the data since stochastic gradient descent assumes I.I.D. samples
perm = np.random.permutation(data.shape[0])
data, colors = data[perm], colors[perm]

# build model to extract the manifolds and learn a mapping / encoder to be able
# to reproduce this on test data
print "Training encoder to unroll the embedded data..."
start = time.time()
code, encoder = compute_embedding(data, 2, epochs=100, batch_size=100,
                                  learning_rate=0.01, seed=0)
print "done in %ds" % (time.time() - start)

# evaluation of the quality of the embedding by comparing kNN queries from the
# original (high dim) data and the low dim code on the one hand, and from the
# ground truth low dim manifold and the low dim code on the other hand

score_code_data = local_match(data, code, query_size=50, ratio=1, seed=0)
print "kNN score match code/data:", score_code_data

if mdp is not None:
    # unroll the same data with HLLE
    print "Comptuting projection using Hessian LLE model..."
    start = time.time()
    hlle_code = mdp.nodes.HLLENode(15, output_dim=2)(data)
    print "done in %ds" % (time.time() - start)

    score_hlle_code_data = local_match(
        data, hlle_code, query_size=50, ratio=1, seed=0)
    print "kNN score match hlle_code/data:", score_hlle_code_data


# plot some 2D projections
sp = pl.subplot(221)
sp.scatter(data[:, 0], data[:, 1], c=colors)
sp.set_title("Projection to axis (x_0, x_1)")

sp = pl.subplot(222)
sp.scatter(data[:, 3], data[:, 7], c=colors)
sp.set_title("Projection to axis (x_3, x_7)")

# plot the learned 2D embedding recovered by codemaker
sp = pl.subplot(223)
sp.scatter(code[:, 0], code[:, 1], c=colors)
sp.set_title("2D manifold recovery by codemaker")

if mdp is not None:
    # plot the 2D embedding computed by HLLE
    sp = pl.subplot(224)
    sp.scatter(hlle_code[:, 0], hlle_code[:, 1], c=colors)
    sp.set_title("2D manifold recovery by HLLE")

pl.show()

