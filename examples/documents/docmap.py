"""Train an encoder to extract a 2D swissroll manifold from higher dim data"""
import os
import numpy as np
import pylab as pl
import time
import random

from scikits.learn.datasets import load_mlcomp
from codemaker.embedding import SDAEmbedder
from codemaker.evaluation import Neighbors, local_match, pairwise_distances

pl.clf()
data_file = "20news_data.npy"
target_file = "20news_target.npy"

if not os.path.exists(data_file):
    print "Loading 20 newsgroups training set... "
    t0 = time.time()
    news_train = load_mlcomp('20news-18828', 'train')
    X, y = news_train.data, news_train.target
    n_samples, n_features = X.shape
    print "done in %fs" % (time.time() - t0)
    print "n_samples: %d, n_features: %d" % (n_samples, n_features)

    # reshuffle:
    print "Reshuffling the data"
    random.seed(0)
    permutation = range(n_samples)
    random.shuffle(permutation)
    X = X[permutation]
    y = y[permutation]

    # sample part of X to be used for plotting
    plot_size = 5000
    data = X[:plot_size]
    colors = y[:plot_size]

    print "Saving for later reuse..."
    np.save(file(data_file, 'wb'), data)
    np.save(file(target_file, 'wb'), colors)

else:
    print "Loading presaved data..."
    t0 = time.time()
    data = np.load(file(data_file))
    colors = np.load(file(target_file))
    print "done in %fs" % (time.time() - t0)

n_samples, n_features = data.shape


# build model to extract the manifold and learn a mapping / encoder to be able
# to reproduce this on test data
embedder = SDAEmbedder((n_features, 500, 30, 2),
                       noise=0.1,
                       reconstruction_penalty=1.0,
                       embedding_penalty=0.0,
                       sparsity_penalty=0.0,
                       learning_rate=0.1, seed=0)

print "Training encoder to extract a semantic preserving 2D mapping"
start = time.time()
embedder.pre_train(data, slice_=slice(None, None), epochs=1000, batch_size=100)
print "done in %ds" % (time.time() - start)

# evaluation of the quality of the embedding by comparing kNN queries from the
# original (high dim) data and the low dim code on the one hand, and from the
# ground truth low dim manifold and the low dim code on the other hand

fig = pl.figure(1)
code = embedder.encode(data)
score_code_data = local_match(data, code, query_size=50, ratio=1, seed=0)
print "kNN score match after pre-training code/data:", score_code_data
_, _, corr = pairwise_distances(data, code, ax=fig.add_subplot(1, 1, 1),
                                title="pre-training")
print "Pairwise distances correlation:", corr

## fine tuning
#print "Fine tuning encoder to unroll the embedded data..."
#start = time.time()
#embedder.fine_tune(data, epochs=100, batch_size=5)
#print "done in %ds" % (time.time() - start)

#code = embedder.encode(data)
#score_code_data = local_match(data, code, query_size=50, ratio=1, seed=0)
#print "kNN score match after fine-tuning code/data:", score_code_data
#_, _, corr = pairwise_distances(data, code, ax=fig.add_subplot(3, 1, 3),
#                                title="fine tuning")
#print "Pairwise distances correlation:", corr

pl.figure(2)
# plot the 2d projection of the first two axes
sp = pl.subplot(121)
sp.scatter(data[:, 0], data[:, 1], c=colors)
sp.set_title("2D Projection of the high dim data")

# plot the learned unrolled 2D embedding (not working yet)
sp = pl.subplot(122)
sp.scatter(code[:, 0], code[:, 1], c=colors)
sp.set_title("2D semantic mapping of the documents")
pl.show()

