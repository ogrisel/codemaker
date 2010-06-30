"""Script to encode faces into sparse representation"""


import numpy as np
import os
import sys
from gzip import GzipFile
from optparse import OptionParser
from codemaker.sparse import SparseEncoder

if __name__ == '__main__':
    parser = OptionParser(usage = "usage: %prog [option]")

    parser.add_option("-p", '--pack-file', action="store",
                      dest="pack_file", default="faces.npy.gz",
                      help="Packed numpy array of preprocessed img data")

    parser.add_option("-f", '--filename-index', action="store",
                      dest="index_file", default="face_filenames.txt",
                      help="Text index of the packed faces")

    (options, args) = parser.parse_args()

    print "loading " + options.pack_file
    faces = np.load(GzipFile(options.pack_file))
    filenames = [line.strip() for line in file(options.index_file)]

    # first test: use the top 100 faces as dictionary
    print "encoding 1000 faces using the first 100 prototypes"
    se_data_100 = SparseEncoder(faces[:100].T, max_features=10)
    code = se_data_100(faces[:1000])


