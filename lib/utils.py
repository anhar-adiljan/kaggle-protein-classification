from __future__ import absolute_import

import os
import glob
import pandas as pd
import numpy as np
from skimage import io

def strToList(string_input):
	return [int(elem) for elem in string_input.split(' ')]

def blueFileNameToId(root_dir, fname):
    start = fname.index(root_dir) + len(root_dir)
    if fname[start] == '/':
        start += 1
    end = fname.index('_blue.png')
    return fname[start:end]

def getChannelAttr(pixel_data, channel):
    mean = np.mean(pixel_data)
    var = np.var(pixel_data)
    max_entry = np.max(pixel_data)
    min_entry = np.min(pixel_data)
    return mean, var, max_entry, min_entry

def getIds(root_dir):
    fnames = glob.glob(root_dir + '/*_blue.png')
    ids = [blueFileNameToId(root_dir, fname) for fname in fnames]
    return sorted(ids)


def getLabels(label_file):
    if label_file:
        labels_frame = pd.read_csv(label_file)
        labels_frame['Target'] = labels_frame['Target'].apply(strToList)
        return labels_frame
    else:
        return None

def getImageByColor(root_dir, img_id, color):
    fname = img_id + '_' + color + '.png'
    fname = os.path.join(root_dir, fname)
    image = io.imread(fname)
    return image

def getImagesById(root_dir, img_id):
    colors = ['blue', 'green', 'red', 'yellow']
    images = dict([(c, getImageByColor(root_dir, img_id, c)) for c in colors])
    return images
