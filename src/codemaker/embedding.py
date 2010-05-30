"""Utility to find local structure preserving low dimensional embeddings"""

from pynnet.layers.autoencoder import Autoencoder
from pynnet.trainers import get_updates
import numpy as np
import theano
import theano.tensor as T

def compute_embedding(data, target_dim, epochs=100, batch_size=100,
                      learning_rate=0.01):
    """Learn an embedding of data into a vector space of target_dim

    Current implementation: only minize the reconstruction error of a simple
    autoencoder

    TODO: stack several wide-band autoencoders with a sparsity penalty + input
    corruption during pre-training + add a local structure preservation penalty
    to the overall cost expression such as the one described in the Parametric
    t-SNE paper by Maarten 2009.

    """
    data = np.atleast_2d(data)
    data = np.asanyarray(data, dtype=theano.config.floatX)
    n_samples, n_features = data.shape

    # build a traditional autoencoder with sigmoid non linearities
    ae = Autoencoder(data.shape[1], target_dim, tied=True, noise=0.0)
    ae.build(T.fmatrix('x'))

    # symbolic expression of an estimator of the divergence between
    # similarities in input and output spaces
    dx = T.sum((ae.input[:-1] - ae.input[1:]) ** 2, axis=1)
    dy = T.sum((ae.output[:-1] - ae.output[1:]) ** 2, axis=1)
    avg_dx, avg_dy = dx.mean(), dy.mean()
    embedding_cost = T.sum(abs(dx/avg_dx - dy/avg_dy))

    # compound cost mix the regular autoencoder cost along with the embedding
    # cost
    cost = ae.cost + 2. * embedding_cost

    train = theano.function(
        [ae.input], cost,
        updates=get_updates(ae.pre_params, cost, learning_rate))
    encode = theano.function([ae.input], ae.output)

    n_batches = n_samples / batch_size
    for e in xrange(epochs):
        print "reshuffling data"
        shuffled = data.copy()
        np.random.shuffle(shuffled)

        print "training..."
        err = np.zeros(n_batches)
        for b in xrange(n_batches):
            batch_input = shuffled[b * batch_size:(b + 1) * batch_size]
            err[b] = train(batch_input).mean()
        print "epoch %d: err: %0.3f" % (e, err.mean())

    return encode(data), encode

