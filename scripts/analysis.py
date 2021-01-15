import os, json, pprint, numpy, scipy, pandas, urllib
from collections import OrderedDict
import numpy as np
from numpy.linalg import svd
from matplotlib import gridspec, pyplot as plt
import scipy.ndimage
import ClearMap.IO as io
from ClearMap.Analysis.Label import Label
import ClearMap.Analysis.Label as lbl
from ClearMap.Analysis.Label import countPointsInRegions

from settings import TempFolder

def addMissingIds(ids, counts, all_ids, sort='id'):
    missing_ids = numpy.array([id for id in all_ids if id not in ids])
    if missing_ids.size == 0:
        return ids, counts
    ids = numpy.concatenate((ids, missing_ids))
    counts = numpy.concatenate((counts, numpy.zeros(missing_ids.shape)))
    if sort == 'id':
        indices = numpy.argsort(ids)
    elif sort == 'count':
        indices = numpy.argsort(counts)
    else:
        indices = slice(None)
    ids = ids[indices]
    counts = counts[indices]
    return ids, counts

def propagateValuesUpHierarchy(ids, counts, label = lbl.Label, sortIds = True, verbose = False, allow_overcount = False):
  idToCount = { id: count for id, count in zip(ids, counts) }
  startIds = ids
  for id in startIds:
      count = idToCount[id]
      if id not in label.ids:
        print('WARNING [analysis.propagateValuesUpHierarchy]: id = %d not in master list of ids' % id)
        continue
      parent = label.parent(id)
      while parent > 0:
          if parent not in idToCount:
            idToCount[parent] = 0
          elif parent in ids:
            if allow_overcount:
                idToCount[parent] += count
            elif verbose:
                print('WARNING [analysis.propagateValuesUpHierarchy]: id = %d being overwritten' % id)
          else:
            idToCount[parent] += count
          parent = label.parent(parent)
  ids, counts = zip(*idToCount.items())
  ids, counts = numpy.array(ids), numpy.array(counts).squeeze()
  ids, counts = addMissingIds(ids, counts, label.ids, sort = ('id' if sortIds else None))
  return ids, counts

def computePairwiseMetric(group1, group2, metric):
    scores = np.zeros((len(group1), len(group2)), dtype = float)
    for i in range(len(group1)):
        for j in range(len(group2)):
            g1, g2 = group1[i].ravel(), group2[j].ravel()
            mask = (g1 >= 0) & (g2 >= 0)
            scores[i,j] = metric(g1[mask].ravel(), g2[mask].ravel())
    return scores

def compareToThreshold(data, threshold=None):
    return (data > threshold).astype(data.dtype) if threshold is not None else data

def getTransformSingularValues(affineParameterFmt):
    idToAffineSingularValues = {}
    for id in idToTime.keys():
        affineParameters = [line for line in open(affineParameterFmt % id) if line.startswith('(TransformParameters')][0]
        affineParameters = affineParameters.split(' ')[1:10]
        affineParameters = [float(el) for el in affineParameters]
        matrix = numpy.array(affineParameters).reshape((3,3))
        u, s, v = svd(matrix)
        idToAffineSingularValues[id] = s
    return idToAffineSingularValues


