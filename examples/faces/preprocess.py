"""Preprocessing for the face recognition experiment

- scan all pictures from the LFW dataset
- convert to gray and rescale to a smaller size suitable
- use OpenCV to select a bounding rectangle and drop undetected faces
- rescale to even smaller size for neural nets
- build a numpy ndarray with the resulting gray data
"""

import os
import sys
import cv
import shutil
from multiprocessing import Pool
from optparse import OptionParser

RAW_DATA_FOLDER = "lfw_funneled"
GRAY_DATA_FOLDER = "lfw_funneled_gray"
CASCADE_FILE = "haarcascade_frontalface_alt.xml"

# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (20, 20)
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

def detect_and_extract(input_img_path, output_img_path, cascade, w=64, h=64,
                       print_progress=False):
    """Run the OpenCV facedetector and extract downscaled gray bounding box"""
    gray = cv.LoadImage(input_img_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cv.EqualizeHist(gray, gray)

    t = cv.GetTickCount()
    faces = cv.HaarDetectObjects(gray, cascade, cv.CreateMemStorage(0),
                                 haar_scale, min_neighbors, haar_flags, min_size)
    t = cv.GetTickCount() - t
    if faces:
        # TODO: handle the case where several faces appear in the same page
        rect, _ = faces[0]
        # select the rectangle to extract it
        cv.SetImageROI(gray, rect)
        small = cv.CreateImage((w, h), 8, 1)
        cv.Resize(gray, small, cv.CV_INTER_CUBIC)
        cv.SaveImage(output_img_path, small)
        cv.ResetImageROI(gray)

        if print_progress:
            sys.stdout.write(".")
            sys.stdout.flush()


class WorkerInterrupt(Exception): pass

def _job_fun(args):
    try:
        cascade_file, pairs = args
        cascade = cv.Load(cascade_file)
        for input_file, output_file in pairs:
            detect_and_extract(input_file, output_file, cascade,
                               print_progress=True)
    except KeyboardInterrupt:
        raise WorkerInterrupt()


if __name__ == '__main__':
    parser = OptionParser(usage = "usage: %prog [option]")

    parser.add_option("-i", "--input-folder", action="store",
                      dest="input_folder", default=RAW_DATA_FOLDER,
                      help="Input folder with LFW data")

    parser.add_option("-o", "--output-folder", action="store",
                      dest="output_folder", default=GRAY_DATA_FOLDER,
                      help="Folder to store preprocessed pictures")

    parser.add_option("-c", "--cascade", action="store", dest="cascade",
                      type="str", help="Haar cascade file",
                      default=CASCADE_FILE)

    parser.add_option("-x", "--clean", action="store_true",
                      dest="clean", default=False,
                      help="Clean any existing output folder")

    (options, args) = parser.parse_args()

    # check that the cascade file is loadable
    cascade = cv.Load(options.cascade)

    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)
    else:
        if options.clean:
            shutil.rmtree(options.output_folder)
            os.makedirs(options.output_folder)

    pool = Pool()
    job_args = []
    filepath_pairs = []
    try:
        for dirpath, dirnames, filenames in os.walk(options.input_folder):
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                if filename.endswith(".jpg"):
                    input_path = os.path.join(dirpath, filename)
                    output_path = os.path.join(options.output_folder, filename)
                    output_path = output_path[:-3] + "png"
                    if os.path.exists(output_path):
                        continue
                    filepath_pairs.append((input_path, output_path))
                    if len(filepath_pairs) >= 100:
                        job_args.append((options.cascade, filepath_pairs))
                        filepath_pairs = []

        pool.map(_job_fun, job_args)
        pool.close()

    except KeyboardInterrupt:
        pool.terminate()
    except WorkerInterrupt:
        pool.terminate()
    finally:
        print
        pool.join()

