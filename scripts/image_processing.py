import os, json, math, glob, re, numpy, skimage
import numpy as np
import scipy.spatial
import ClearMap.IO as io
from ClearMap.Alignment.Resampling import resamplePoints
from ClearMap.Alignment.Elastix import transformPoints

import settings

def resampleSum_nd(data, resolutionSource, resolutionSink):
    assert(data.ndim == len(size) and all([d > s for d,s in zip(data.shape, size)]))
    binsize = tuple([ int(np.ceil(1.0 * d / s)) for d,s in zip(data.shape, size) ])
    reshapesize = [_ for s,b in zip(size, binsize) for _ in (s,b)]
    data = data.reshape(reshapesize)
    for i in range(len(size)):
        data = data.sum(axis = (i+1))
    return data

def resampleSum_3d(data, resolutionSource, resolutionSink, size = None):
    assert(data.ndim == 3)
    n = numpy.asarray
    if size is None:
        size = (n(data.shape) * n(resolutionSource) / n(resolutionSink)).astype(int)
    pct = numpy.asarray(data.shape, dtype=float) / numpy.asarray(size)
    assert(numpy.all(pct > 1))
    delta = tuple(pct.astype(int))
    result = numpy.zeros(size)
    for index in numpy.ndindex(*size):
        si = tuple(numpy.asarray(delta) * numpy.asarray(index))
        ei = tuple(numpy.asarray(si) + numpy.asarray(delta))
        result[index] = data[si[0]:ei[0], si[1]:ei[1], si[2]:ei[2]].sum()
    return result

