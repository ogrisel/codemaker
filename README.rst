=========
codemaker
=========

Python utilities based on theano_ and pynnet_ and scikitis.learn_ to learn
vector encoders that map vector data to either:

  - dense codes in low dimensional space, useful for semantic mapping and
    visualization

  - sparse codes in medium to high dimensional space, useful for semantic
    indexing, denoising and compression

In both cases, care is taken to preserve nearest neighboors relationships.

.. _theano: http://deeplearning.net/software/theano/
.. _pynnet: http://code.google.com/p/pynnet/
.. _`scikitis.learn`: http://scikit-learn.sf.net


Project status
==============

This is experimental code. Nothinh is expected to work as advertised yet :)


Licensing
=========

MIT: http://www.opensource.org/licenses/mit-license.php


Hacking
=======

Download the source distrib of the afore mentionned dependencies, untar them in
the parent folder of ``codemaker``, build `scikits.learn`_ in local mode with
`python setup build_ext -i` and setup the dev environment with::

  $ . ./activate.sh
  
You should now be able to fire you favorite python shell and import
`codemaker`::

  >>> import codemaker
  >>> help(codemaker)

Run the tests with the nosetests_ command.

.. _nosetests: http://somethingaboutorange.com/mrl/projects/nose


Examples
========

TODO


