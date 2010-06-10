"""Utility to find local structure preserving low dimensional embeddings"""

from pynnet import errors
from pynnet.layers.autoencoder import Autoencoder
from pynnet.net import NNet
from pynnet.trainers import get_updates
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import theano.tensor as T

def compute_embedding(data, target_dim, epochs=100, batch_size=100,
                      learning_rate=0.01, seed=None):
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

    if seed is not None:
        np.random.seed(seed)
    theano_rng = RandomStreams(seed)

    # build autoencoders with sigmoid non linearities
    # TODO: patch pynnet to allow for seeding the autoencoder noise
    ae_in = Autoencoder(n_features, n_features / 2, tied=True, noise=0.2,
                        noise_rng=theano_rng)
    ae_in.build(T.fmatrix('ae_in'))

    ae_out = Autoencoder(n_features / 2, target_dim, tied=True, noise=0.1,
                         noise_rng=theano_rng)
    ae_out.build(ae_in.output)

    # build the forward encoder using the forward layers of the encoders
    encoder = NNet([ae_in, ae_out], errors.mse)
    encoder.build(T.fmatrix('enc_in'), T.fvector('enc_target'))

    # symbolic expression of an estimator of the divergence between
    # similarities in input and output spaces
    dx = T.sum((ae_in.input[:-1] - ae_in.input[1:]) ** 2, axis=1)
    dy = T.sum((ae_out.output[:-1] - ae_out.output[1:]) ** 2, axis=1)
    avg_dx, avg_dy = dx.mean(), dy.mean()
    embedding_cost = T.sum((dx/avg_dx - dy/avg_dy) ** 2
                           * T.exp(-(dx / (0.5 * avg_dx)) ** 2))

    # compound cost mix the regular autoencoder cost along with the embedding
    # cost
    cost = 1. * ae_in.cost + .1 * ae_out.cost + 5. * embedding_cost
    #cost = 5. * embedding_cost
    params = ae_in.pre_params + ae_out.pre_params
    train = theano.function(
        [ae_in.input], cost, updates=get_updates(params, cost, learning_rate))

    encode = theano.function([encoder.input], encoder.output)

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

