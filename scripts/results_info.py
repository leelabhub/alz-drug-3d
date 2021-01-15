import numpy, scipy, os, pandas
import matplotlib.pyplot as plt

import ClearMap.IO as io
import ClearMap.Analysis.Label as lbl
from ClearMap.Analysis.Label import Label, LabelInfo, DefaultAnnotationFile
from ClearMap.Settings import ClearMapPath

from settings import AnnotationFile 


# load region stats and define function for normalization
AnnotationStatsPath = os.path.join(ClearMapPath, 'Data/ARA2_annotation_statistics.csv')
annotationStats = pandas.read_csv(AnnotationStatsPath)

def normalizeCountsByRegionVolume(annotationStats, ids, counts):
    assert(len(ids) == len(counts))
    counts = counts.copy().astype('float64')
    for index in xrange(len(ids)):
        id = ids[index]
        count = counts[index]
        search = annotationStats.index[annotationStats['RegionID'] == id]
        if len(search) > 0:
            searchindex = search[0]
            regionInfo = annotationStats.iloc[searchindex]
            volume = regionInfo[-1] #regionInfo[' Volume']
            # print counts[index], '/', volume, '=', counts[index] / volume # for debugging
            if volume > 0:
                counts[index] = counts[index] / volume
            else:
                counts[index] = float('nan')
        else:
            counts[index] = float('nan')
    return counts


# normalizes by total count after filtering out the ids that aren't in the annotation statistics spreadsheet (i.e. background, ventricles, etc.)
def normalizeCountsByTotalCount(annotationStats, ids, counts):
    indicesForIDsInStats = numpy.array([index for index in xrange(len(ids)) if ids[index] in annotationStats['RegionID'].values])
    indicesForIDsNotInStats = numpy.array([index for index in xrange(len(ids)) if ids[index] not in annotationStats['RegionID'].values])
    idsInStats = ids[indicesForIDsInStats]
    countsForIdsInStats = counts[indicesForIDsInStats]
    totalCount = numpy.sum(countsForIdsInStats)
    counts = counts.astype('float64') / totalCount
    counts[indicesForIDsNotInStats] = float('nan')
    return counts


# Use the annotation atlas to get the color to show for each cell points
def getRegionColorsForCellPoints(pointsSource, colorsSink, AnnotationFile = AnnotationFile):
    points = io.readPoints(pointsSource)
    labels = lbl.labelPoints(points, AnnotationFile)
    # colors = [lbl.Label.color(id) for id in labels] # should work, but throws error
    # colors = [lbl.Label.color(id) for id in labels if id in lbl.Label.ids] # too slow
    colors = [lbl.Label.color(id) if id < 100000 else [0,0,0] for id in labels]
    return io.writePoints(colorsSink, numpy.array(colors))

def rgb2hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def descendentsForId(startId, labelInfo = Label):
    descendents = []
    queue = [startId]
    while len(queue) is not 0:
        nextId = queue.pop()
        children = [id for id in labelInfo.ids if labelInfo.parent(id) == nextId]
        descendents.append(nextId)
        queue += children
    return descendents

root_id = 8 # Basic cell groups and regions

def pathToRootForId(id, stopId = root_id):
    path = []
    while id != root_id and id is not None:
        if id not in Label.ids:
            break
        id = Label.parent(id)
        path.insert(0,id)
    return path

def breadthFirstSearch(startId, labelInfo = Label, pathstopcond = lambda id,children : len(children) is 0):
    """
    Traverses the hierarchy (given a starting node id) in a breadth-first fashion, and will traverse
    all possible paths. It will stop along a given path when the `pathstopcond` function parameter
    returns a value of true
    """
    searchresults = []
    queue = [startId]
    while len(queue) is not 0:
        nextId = queue.pop(0)
        children = [id for id in labelInfo.ids if labelInfo.parent(id) == nextId]
        if pathstopcond(nextId, children):
            searchresults.append(nextId)
        else:
            queue += children
    return searchresults

def regionIdsAtLevel(level, labelInfo = Label):
    return [id for id in labelInfo.ids if labelInfo.level(id) == level]

# This can be passed into the `leavesForId` function as the leafcondition to ignore layer-specific regions
def isLeafOrGroupOfLayers(id, children):
    if len(children) is 0:
        return True
    else:
        return 'layer' in Label.name(children[0]).lower()

def leavesForId(startId, labelInfo = Label, leafcondition = lambda id,children : len(children) is 0):
    leaves = []
    queue = [startId]
    while len(queue) is not 0:
        nextId = queue.pop()
        children = [id for id in labelInfo.ids if labelInfo.parent(id) == nextId]
        if leafcondition(nextId,children):
            leaves.append(nextId)
        else:
            queue += children
    return leaves

def acronymToId(acronym, labelInfo = Label):
    return labelInfo.acronyms.keys()[labelInfo.acronyms.values().index(acronym)]

def idToAcronym(id, labelInfo = Label):
    return labelInfo.acronym(id)

def childrenForId(parentId, labelInfo = Label):
    return [id for id in labelInfo.ids if labelInfo.parent(id) == parentId]


