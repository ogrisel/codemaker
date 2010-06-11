"""Train an encoder to extract a 2D swissroll manifold from higher dim data"""
import numpy as np
import pylab as pl
import time

from codemaker.datasets import swissroll
from codemaker.embedding import compute_embedding
from codemaker.evaluation import Neighbors, local_match

n_features = 3
n_samples = 1000

print "Generating embedded swissroll with n_features=%d and n_samples=%d" % (
    n_features, n_samples)

data, manifold = swissroll.load(
    n_features=n_features,
    n_samples=n_samples,
    n_turns=1.5,
    radius=1.,
)

# build model to extract the manifold and learn a mapping / encoder to be able
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

score_code_manifold = local_match(manifold, code, query_size=50, ratio=1, seed=0)
print "kNN score match code/manifold:", score_code_manifold

score_manifold_data = local_match(data, manifold, query_size=50, ratio=1, seed=0)
print "kNN score match manifold/data:", score_manifold_data

# plot the 2d projection of the first two axes
colors = manifold[:, 0]
pl.subplot(131).scatter(data[:, 0], data[:, 1], c=colors)

# plot the unrolled manifold embedded in the data
pl.subplot(132).scatter(manifold[:, 0], manifold[:, 1], c=colors)

# plot the learned unrolled 2D embedding
pl.subplot(133).scatter(code[:, 0], code[:, 1], c=colors)
pl.show()

