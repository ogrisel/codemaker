"""Script to encode faces into sparse representation"""


import numpy as np
import pylab as pl
import os
import sys
from gzip import GzipFile
from optparse import OptionParser
from codemaker.sparse import SparseEncoder
from codemaker.evaluation import Neighbors
from codemaker.evaluation import local_match
from codemaker.evaluation import pairwise_distances


def load_file(filename):
    if filename.endswith(".gz"):
        f = GzipFile(filename)
    else:
        f = file(filename)
    return np.load(f)


if __name__ == '__main__':
    pl.clf()
    parser = OptionParser(usage = "usage: %prog [option]")

    parser.add_option("-p", '--faces-file', action="store",
                      dest="faces_file", default="faces.npy.gz",
                      help="Packed numpy array of preprocessed img data")

    parser.add_option("-f", '--filename-index', action="store",
                      dest="index_file", default="face_filenames.txt",
                      help="Text index of the packed faces")

    parser.add_option("-d", '--dictionary-file', action="store",
                      dest="dictionary_file", default="faces.npy.gz",
                      help="Serialized numpy array holding")

    parser.add_option("-s", '--samples', action="store",
                      type=int,
                      dest="samples", default=None,
                      help="Portion of the samples to encode")

    parser.add_option("-n", '--dimension', action="store",
                      type=int,
                      dest="dimension", default=None,
                      help="Portion of the dictionary to use for encoding")

    parser.add_option("-z", '--non-zeros', action="store",
                      type=int,
                      dest="non_zeros", default=10,
                      help="Maximum number of non zero code components")

    (options, args) = parser.parse_args()

    print "loading data: " + options.faces_file
    faces = load_file(options.faces_file)
    filenames = [line.strip() for line in file(options.index_file)]

    print "loading dictionary: " + options.dictionary_file
    if options.dictionary_file == options.faces_file:
        dictionary = faces
    else:
        dictionary = load_file(options.dictionary_file)

    # restrict the number of samples to encode
    if options.samples:
        faces = faces[:options.samples]

    # restrict to a low dim portion of the dictionary if requested
    if options.dimension:
        dictionary = dictionary[:options.dimension]

    # first test: use the top 100 faces as dictionary
    print "encoding faces"
    encoder = SparseEncoder(faces.T, max_features=options.non_zeros)
    code = encoder(faces)

    f1 = pl.figure(1)
    _, _, corr = pairwise_distances(faces, code, ax=f1.add_subplot(1, 1, 1))
    print "correlation:", corr
    pl.show()


