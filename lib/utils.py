from __future__ import absolute_import

import numpy as np

def getChannelAttr(pixel_data, channel):
    mean = np.mean(pixel_data)
    var = np.var(pixel_data)
    max_entry = np.max(pixel_data)
    min_entry = np.min(pixel_data)
    return mean, var, max_entry, min_entry
