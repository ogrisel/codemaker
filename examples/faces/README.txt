Experiment at using sparse coding for faces recognition and indexing using a
sparse represention.

To run the experiment:

1- Install the version 2.1 of the OpenCV_ (be sure to build the python bindings).
   For ubuntu karmic and lucid it is recommended to use the `prebuilt
   packages`_. For maverick, the distrib packages are up to date.

.. _OpenCV: http://opencv.willowgarage.com/wiki/
.. _`prebuild packages`: http://opencv.willowgarage.com/wiki/Ubuntu_Packages

Also install restkit_, pyleargist_ and lxml_::

   % sudo apt-get install python-lxml libfftw3-dev
   % sudo pip install restkit
   % sudo pip install pyleargist

.. _restkit: http://github.com/benoitc/restkit/
.. _pyleargist: http://github.com/benoitc/restkit/
.. _lxml: http://codespeak.net/lxml/

2- Download the funneled version of the Labeled Faces in the Wild
   dataset (a.k.a. LFW_) and untar it in the current folder (or use a symbolic
   link named ``lfw_funneled``)::

   % wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz # (233MB)
   % tar zxvf lfw-funneled.tgz

.. _LFW: http://vis-www.cs.umass.edu/lfw/

3- Dowload the OpenCV_ Haarmodel for faces detection and put it in the local
   directory too::

   % wget https://code.ros.org/svn/opencv/trunk/opencv/data/haarcascades/haarcascade_frontalface_alt.xml

4- Run the ``preprocess.py`` script that will create a new folder
   ``lfw_funneled_gray`` containing resized, gray level pictures that will be
   used as training set for the dictionnary learner.

4- TODO

