#!/usr/bin/env python
"""Script to build image classification dataset from commons.wikipedia.org

Author: olivier.grisel@ensta.org
License: MIT - http://www.opensource.org/licenses/mit-license.php
"""

import os
import sys
import cgi
import random
from gzip import GzipFile
from PIL import Image
from cStringIO import StringIO
from restkit import Resource
from restkit import ConnectionPool
from lxml import etree
import numpy as np
import leargist

CONNECTION_POOL = ConnectionPool(max_connections=4)

BASE_SERVER_URL = "http://commons.wikimedia.org"
BASE_CATEGORY_URL = "/wiki/Category:"

# the magic single command to do gray level & crop / extent transforms
CONVERT_GRAY_CMD = ("convert {input} -colorspace Gray"
                    " -resize {w}x{h}^ -gravity center"
                    " -extent {w}x{h} {output}")

CONVERT_COLOR_CMD = ("convert {input}"
                    " -resize {w}x{h}^ -gravity center"
                    " -extent {w}x{h} {output}")

COLLECTIONS = (
    ("portraits", [
        "Portrait_photographs_of_men",
        "Portrait_photographs_of_women",
    ]),
    ("landscapes", [
        "Landscapes",
        "Coasts",
        "Landscape_of_France",
        "Snowy_landscapes",
    ]),
    ("diagrams", [
        "Diagrams",
    ]),
    ("paintings", [
        "Impressionist_paintings",
        "Post-Impressionist_paintings",
        "Realist_paintings",
    ]),
    ("drawings", [
        "Sepia_drawings",
        "Pencil_drawings",
        "Charcoal_drawings",
        "Pastel_drawings",
    ]),
    ("buildings", [
        "Columns",
        "Building_construction",
        "Building_interiors",
    ]),
)

def microthumbs(filenames, sizes=[2, 3, 4]):
    """Aggregate pixel values of thumbnails of various micro sizes

    This can be used as a poor man's replacement for GIST features for image
    clustering.
    """
    result = np.zeros((len(filenames), 3 * sum(s ** 2 for s in sizes)))
    for i, f in enumerate(filenames):
        vectors = []
        for s in sizes:
            img = Image.open(f).convert('RGB').resize((s, s))
            vectors.append(np.asarray(img, dtype=np.double).flatten() / 128)
        result[i][:] = np.concatenate(vectors)
    return result


#TODO: drop this and replace me by scipy.ndimage instead
def img_to_array(image, mode='L', w=32, h=32, dtype=np.float32):
    """Convert a PIL Image into a numpy array"""
    if image.mode != mode:
        image = image.convert(mode=mode)
    if image.size != (w, h):
        image = image.resize((w, h), Image.ANTIALIAS)

    data = np.array(image.getdata(), dtype)
    rescaled_data = data / 128. - 1
    return rescaled_data


def array_to_img(data, w=32, h=32):
    """Convert brain ouput to an Image with the same format as the input"""
    rescaled = ((data + 1.) * 128).clip(0, 255)
    image = Image.new('L', (w, h))
    image.putdata(list(rescaled.flat))
    return image


def urldecode(url):
    """Helper to convert query parameters as a dict"""
    if "?" in url:
        path, query = url.split("?", 1)
        return path, dict(cgi.parse_qsl(query))
    else:
        return url, {}


def find_image_urls(category, server_url=BASE_SERVER_URL,
                    category_prefix=BASE_CATEGORY_URL,
                    resource=None, pool=CONNECTION_POOL):
    """Scrap the mediawiki category pages to identify thumbs to download"""
    parser = etree.HTMLParser()
    if resource is None:
        resource = Resource(server_url, pool_instance=pool)
    collected_urls = []
    page = category_prefix + category
    parameters = {}
    while page is not None:
        response = resource.get(page, **parameters)
        tree = etree.parse(StringIO(response.body), parser)
        thumb_tags = tree.xpath(
            '//div[@class="thumb"]//img[contains(@src, "/thumb/")]')
        collected_urls.extend(tag.get('src') for tag in thumb_tags)
        links = tree.xpath('//a[contains(text(), "next 200")]')
        if links:
            page, parameters = urldecode(links[0].get('href'))
        else:
            page = None
    return collected_urls


def collect(collections, folder):
    """Collect categorized thumbnails from commons.mediawiki.org"""
    resource = Resource(BASE_SERVER_URL, pool_instance=CONNECTION_POOL)
    for category_name, subcategories in collections:
        category_folder = os.path.join(folder, category_name)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
        for subcategory in subcategories:
            for url in find_image_urls(subcategory, resource=resource):
                _, filename = url.rsplit("/", 1)
                if len(filename) > 200:
                    print "skipping file: ", filename
                    continue
                filepath = os.path.join(category_folder, filename)
                if os.path.exists(filepath):
                    print "skipping", url
                    continue
                print "downloading", url
                with file(filepath, 'wb') as f:
                    payload = Resource(
                        url, pool_instance=CONNECTION_POOL).get().body
                    f.write(payload)


def preprocess(input_folder, output_folder, w=32, h=32, gray_levels=True):
    """Convert images to fixed dimensional squared graylevel jpegs"""
    for category in os.listdir(input_folder):
        category_input_folder = os.path.join(input_folder, category)
        category_output_folder = os.path.join(output_folder, category)
        if not os.path.isdir(category_input_folder):
            continue
        if not os.path.exists(category_output_folder):
            os.makedirs(category_output_folder)

        for filename in os.listdir(category_input_folder):
            input_path = os.path.join(category_input_folder, filename)
            output_path = os.path.join(
                category_output_folder, filename + '-preprocessed.jpeg')
            if os.path.exists(output_path):
                print "skipping", filename
                continue
            if gray_levels:
                cmd = CONVERT_GRAY_CMD.format(
                    input=input_path, output=output_path, w=w, h=h)
            else:
                cmd = CONVERT_COLOR_CMD.format(
                    input=input_path, output=output_path, w=w, h=h)

            print "processing", filename
            os.system(cmd)


def pack(input_folder, index_file, data_output_file, label_output_file=None,
         seed=42, dtype=np.float32, transform=None):
    """Pack picture files as numpy arrays with category folder as labels"""
    all_filenames = [(category, os.path.join(input_folder, category, filename))
                     for category in os.listdir(input_folder)
                     for filename in os.listdir(
                         os.path.join(input_folder, category))]
    random.seed(seed)
    random.shuffle(all_filenames)

    # count the number of picture by category and drop excedent so that all
    # categories have equal number of samples

    counts = dict()
    for c, _ in all_filenames:
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = 1
    limit = min(counts.values())

    resampled_filenames = []
    counts = dict()
    for c, fn in all_filenames:
        if c in counts:
            counts[c] += 1
        else:
            counts[c] = 1
        if counts[c] <= limit:
            resampled_filenames.append(fn)

    with file(index_file, "wb") as f:
        f.write("\n".join(resampled_filenames))
        f.write('\n')

    reference = Image.open(resampled_filenames[0])

    w, h = reference.size

    dim = w * h
    if transform == 'gist':
        dim = 960 # hardcoded for now

    # TODO: add microthumb transform as baseline too

    data_array = np.zeros((len(resampled_filenames), dim), dtype=dtype)
    for i, filepath in enumerate(resampled_filenames):
        im = Image.open(filepath)
        if transform == 'gist':
            data_array[i,:] = leargist.color_gist(im)
        else:
            data_array[i,:] = img_to_array(im, w=w, h=h, dtype=dtype).flatten()

    if data_output_file.endswith('.gz'):
        np.save(GzipFile(data_output_file, 'wb'), data_array)
    else:
        np.save(file(data_output_file, 'wb'), data_array)

    # TODO: deal with label data at some point


if __name__ == "__main__":
    # TODO: use optparser to create a real CLI handler
    if sys.argv[1] == "collect":
        collect(COLLECTIONS, "collected-images")
    elif sys.argv[1] == "preprocess":
        preprocess("collected-images", "preprocessed-images")
    elif sys.argv[1] == "pack":
        pack("preprocessed-images", "image_filenames.txt", "images.npy.gz")
    elif sys.argv[1] == "crop-color":
        preprocess("collected-images", "cropped-images", h=200, w=200,
                   gray_levels=False)
    elif sys.argv[1] == "gist":
        pack("cropped-images", "gist_image_filenames.txt", "gist.npy.gz",
             transform='gist')

