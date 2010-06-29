Experiment at using sparse coding for faces recognition and indexing using a
sparse represention.

To run the experiment:

1- Install the version 2.1 of the OpenCV_ (be sure to build the python bindings).
   For ubuntu karmic and lucid it is recommended to use the `prebuilt packages`_

.. _OpenCV: http://opencv.willowgarage.com/wiki/
.. _`prebuild packages`: http://opencv.willowgarage.com/wiki/Ubuntu_Packages

2- Download the funneled version of the Labeled Faces in the Wild
   dataset (a.k.a. LFW_) and untar it in the current folder (or use a symbolic
   link named ``lfw-funneled``):

   http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

3- Run the ``preprocess.py`` script that will create a new folder
   ``lfw-funneled-gray`` containing resized, gray level pictures that will be
   used as training set for the dictionnary learner.

4- TODO

