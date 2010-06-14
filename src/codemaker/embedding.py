"""Utility to find local structure preserving low dimensional embeddings"""

from pynnet import errors
from pynnet.layers.autoencoder import Autoencoder
from pynnet.net import NNet
from pynnet.trainers import get_updates
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import theano
import theano.tensor as T

class SDAEmbedder(object):
    """Build a stack of denoising autoencoders to perform low dim embedding"""

    def __init__(self, dimensions, noise=0.1, sparsity_penalty=1.0,
                 embedding_penalty=1.0, learning_rate=0.01, seed=None):
        """Initialize a stack of autoencoders with sparsity penalty

        dimensions is a python sequence of the input, hidden and output
        activation unit of the stacked architecture.

        TODO: implement fine-tuning by applying SGD on the encoder using a
        divergence measure on the pairwise similarities in input and output
        space as object function to minimize. E.g.: (t-)SNE or Elastic
        Embedding.

        """
        assert len(dimensions) >= 2
        self.rng = np.random.RandomState(seed)
        self.noise_rng = RandomStreams(seed)

        # build a stack of autoencoders for the requested dimensions
        self.autoencoders = []
        previous_output = T.matrix('ae_in')

        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            ae = Autoencoder(in_dim, out_dim, tied=True, noise=noise,
                             rng=self.rng, noise_rng=self.noise_rng)
            ae.build(previous_output)
            previous_output = ae.output
            self.autoencoders.append(ae)

        # chain the encoding parts as a feed forward network
        self.encoder = NNet(self.autoencoders, errors.mse)
        self.encoder.build(T.matrix('enc_in'), T.vector('enc_target'))

        # compile the training functions
        self.sparsity_penalty = sparsity_penalty
        self.embedding_penalty = embedding_penalty
        self.pre_trainers = []
        for ae in self.autoencoders:
            cost = self.get_ae_cost(ae)
            pre_train = theano.function(
                [self.autoencoders[0].input],
                cost,
                updates=get_updates(ae.pre_params, cost, learning_rate)
            )
            self.pre_trainers.append(pre_train)

        # compile the enconding function
        self.encode = theano.function([self.encoder.input], self.encoder.output)

    def pre_train(self, data, slice_=slice(None, None), batch_size=50,
                  epochs=100, checkpoint=10, patience=20, tolerance=1e-4):
        """Iteratively apply SGD to each autoencoder

        If slice_ is provided, only the matching layers are trained (by default
        all layers are trained).
        """
        data = np.atleast_2d(data)
        data = np.asanyarray(data, dtype=theano.config.floatX)
        n_samples, n_features = data.shape

        best_error = None
        best_epoch = 0
        n_batches = n_samples / batch_size

        # select the trainers to use
        trainers = self.pre_trainers[slice_]

        for i, trainer in enumerate(trainers):
            for e in xrange(epochs):
                # reshuffling data to enforce I.I.D. assumption
                shuffled = data.copy()
                self.rng.shuffle(shuffled)

                err = np.zeros(n_batches)
                for b in xrange(n_batches):
                    batch_input = shuffled[b * batch_size:(b + 1) * batch_size]
                    err[b] = trainer(batch_input).mean()

                error = err.mean()
                if e % checkpoint == 0:
                    print "layer [%d/%d], epoch [%03d/%03d]: err: %0.5f" % (
                        i + 1, len(trainers), e + 1, epochs,
                        error)
                if best_error is None or error <  best_error - tolerance:
                    best_error = error
                    best_epoch = e
                else:
                    if e - best_epoch > patience:
                        print "layer [%d/%d]: early stopping at epoch %d" % (
                            i + 1, len(trainers), e + 1)
                        break

    def get_ae_cost(self, ae):
        cost = ae.cost
        if self.sparsity_penalty > 0:
            # assuming the activation of each unit lies in [-1, 1], take the
            # L1 norm of the activation
            cost += self.sparsity_penalty * T.mean(abs(ae.output + 1))
        if self.embedding_penalty > 0:
            cost += self.embedding_penalty * self.get_embedding_cost(ae)
        return cost

    def get_embedding_cost(self, ae):
        """Local divergence from pairwise similarities in input and output"""
        ae_in = self.autoencoders[0]
        dx = T.sum((ae_in.input[:-1] - ae_in.input[1:]) ** 2, axis=1)
        dy = T.sum((ae.output[:-1] - ae.output[1:]) ** 2, axis=1)
        avg_dx, avg_dy = dx.mean(), dy.mean()
        # TODO: experiment with (t-)SNE and Elastic Embedding cost functions
        return T.mean((dx/avg_dx - dy/avg_dy) ** 2
                      * T.exp(-(dx / (0.5 * avg_dx)) ** 2))


