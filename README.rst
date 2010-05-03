=========
codemaker
=========

Python utilities based on theano_ and pynnet_ and scikits.learn_ to learn
vector encoders that map vector data to either:

  - dense codes in low dimensional space, useful for semantic mapping and
    visualization by trying to preserve the local structure

  - sparse codes in medium to high dimensional space, useful for semantic
    indexing, denoising and compression


.. _theano: http://deeplearning.net/software/theano/
.. _pynnet: http://code.google.com/p/pynnet/
.. _scikits.learn: http://scikit-learn.sf.net


Project status
==============

This is experimental code. Nothing is expected to work as advertised yet :)

Implemented:

  - deterministic (optimal) sparse encoding using an existing dictionary
    and coordinate descent (see ``codemaker.sparse``)

Work in progress:

  - stochastic neighbor embedding in low dim space using autoencoders

Planned:

  - stochastic dictionary learning and approximate sparse coding
    using sparsity inducing autoencoders (see Ranzanto 2007)


Licensing
=========

MIT: http://www.opensource.org/licenses/mit-license.php


Hacking
=======

Download the source distrib of the afore mentionned dependencies, untar them in
the parent folder of ``codemaker``, build scikits.learn_ in local mode with
``python setup build_ext -i`` and setup the dev environment with::

  $ . ./activate.sh
  
You should now be able to fire you favorite python shell and import
the `codemaker` package::

  >>> import codemaker
  >>> help(codemaker)

Run the tests with the nosetests_ command.

.. _nosetests: http://somethingaboutorange.com/mrl/projects/nose


Examples
========

TODO


