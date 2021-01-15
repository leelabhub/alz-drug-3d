import os, json, math, glob, re, numpy, skimage, urllib, uuid
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2gray
from skimage import measure
from skimage.draw import line_aa
#import skimage.util.montage
from PIL import Image, ImageDraw
import ClearMap.IO as io
from ClearMap.Alignment.Elastix import transformData
from ClearMap.Utils.ParameterTools import getParameter
import json_cleaner

from settings import TempFolder
import imagej as ij

def loadCleanedJSON(filename):
    return json.loads(json_cleaner.remove_comments(open(filename).read()))

def getGlobalArgument(name, globals):
    if name in globals:
        return globals[name]
    elif name in os.environ:
        return os.environ[name]
    else:
        return None

def getBooleanParameter(dict, parameter, default=None):
    value = getParameter(dict, parameter, default = default)
    return value is not None and str(value).lower()[0] == "t"

def swapKeys(dict, key1, key2):
    dict[key1], dict[key2] = dict[key2], dict[key1]
    return dict

def divideSafe(a, b):
    return numpy.divide(a, b, out = numpy.zeros_like(a), where = (b != 0))

def buildOmeTiffPath(folder, d = 6):
    # if this is already an ome tiff path, return it
    if "\d" in folder:
        return folder
    # if this folder has no subfolders then convert the first file into the path
    if all([not os.path.isdir(os.path.join(folder, p)) for p in os.listdir(folder)]):
        return os.path.join(folder, sorted(os.listdir(folder))[0].replace('0000','\d{4}'))
    # find the subdirectory with the most files
    walk_result = list(os.walk(folder))
    data_dir = max(walk_result, key = lambda res : len(res[2]))[0]
    data_dir_leaf = os.path.split(data_dir)[1]
    # construct full path
    path = os.path.join(folder, data_dir, data_dir_leaf + "_\d{%d}.tif" % (d,))
    return path

def getRegionMask(regionId, sink = True, resolution = 25):
    assert(resolution in [10, 25, 50, 100])
    urlfile = urllib.URLopener()
    url_fmt = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_%d/structure_%d.nrrd"
    uid = uuid.uuid4()
    save_fn = os.path.join(TempFolder, uid.hex + "structure_mask_%d.nrrd") % regionId
    urlfile.retrieve(url_fmt % (resolution, regionId), save_fn)
    mask = io.readData(save_fn)
    if sink is True:
        return save_fn
    elif (sink is None) or (sink is False):
        os.remove(save_fn)
        return mask
    else:
        os.remove(save_fn)
        return io.writeData(sink, mask)

def buildAtlas(regionIds, regionColors, resolution = 25):
    assert(resolution in [10, 25, 50, 100])
    urlfile = urllib.URLopener()
    url_fmt = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_%d/structure_%d.nrrd"
    save_fmt = os.path.join(TempFolder, "structure_mask_%d.nrrd")
    saved_fns = []
    for id in regionIds:
        fn = url_fmt % (resolution, id)
        urlfile.retrieve(fn, save_fmt % id)
        saved_fns.append(fn)
    colorAnnotation = None
    annotation = None
    for id, color in zip(regionIds, regionColors):
        mask = io.readData(save_fmt % id)
        if annotation is None:
            annotation = np.zeros(mask.shape, dtype = 'uint16')
            colorAnnotation = np.zeros(mask.shape + (3,), dtype = 'uint16')
        annotation[mask > 0] = id
        colorAnnotation[mask > 0,:] = color
    # REVIEW: delete the saved NRRD files?
    return annotation, colorAnnotation

# REVIEW: need to implement tile_scale parameter
def montage(img, grid_shape, sink = None, offset = 0, delta = None, projection_fn = None, write_fn = io.writeData): #, tile_scale = 1.0):
    tile_scale = 1.0
    img = io.readData(img)
    assert(img.ndim > 2)
    nrows, ncols = grid_shape
    if delta is None:
        delta = img.shape[2] // (nrows * ncols)
    tile_shape = tuple([(int)(d * tile_scale) for d in img.shape[:2]])
    result_shape = (tile_shape[0]*nrows, tile_shape[1]*ncols) + (1,) + img.shape[3:]
    result = numpy.zeros(result_shape, dtype = img.dtype)
    z = 0
    for i in range(nrows):
        for j in range(ncols):
            r = (int)(i * tile_shape[0])
            c = (int)(j * tile_shape[1])
            if projection_fn is None:
                tile = img[:,:,z,...]
            else:
                tile = img[:,:,z:(z+delta),...]
                tile = projection_fn(tile, axis = 2)
            # if tile.shape != tile_shape:
                # tile = skimage.resize(tile, tile_shape)
            result[r:(r+tile_shape[0]),c:(c+tile_shape[1]),0,...] = tile
            z += delta
    if sink is None:
        return result
    else:
        return write_fn(sink, result)

def replaceInFile(filepath, searchToReplace, sink = None):
    contents = open(filepath,'r').read()
    for search, replace in searchToReplace.iteritems():
        if search not in contents:
            print('WARNING: could not find %s in %s' % (search, filepath))
        contents = contents.replace(search, replace)
    if sink is not None:
        open(sink,'w').write(contents)
    else:
        return contents

def process_folder(inFolderName, outFolderName, processFunc, saveFunc, skip_existing = True):
    if isinstance(inFolderName, list):
        files = inFolderName
    else:
        files = os.listdir(inFolderName)
    for infile in files:
        full_infile = os.path.join(inFolderName, infile)
        full_outfile = os.path.join(outFolderName, infile)
        if os.path.exists(full_outfile) and skip_existing:
            continue
        result = processFunc(full_infile)
        saveFunc(full_outfile, result)

def recursivelySetField(dictionary, key, value):
    if key in dictionary:
        dictionary[key] = value
    for child in dictionary.values():
        if type(child) == dict:
            recursivelySetField(child, key, value)

def coronalToSagittal(img):
    img = numpy.flip(img, axis=1)
    img = numpy.flip(img, axis=0)
    img = img.transpose((1,2,0) + tuple(range(img.ndim))[3:])
    return img

def sagittalToCoronal(img):
    img = img.transpose((2,0,1) + tuple(range(img.ndim))[3:])
    img = numpy.flip(img, axis=0)
    img = numpy.flip(img, axis=1)
    return img

def prettifyHeatmap(vox, atlas, sink = None, mask_background = True, mask_thresh = 20):
    vox = io.readData(vox)
    atlas = io.readData(atlas)
    if mask_background:
        vox[atlas <= mask_thresh] = 0
    vox = mergeChannels((atlas, vox))
    vox = vox.transpose([2,0,1,3])
    vox = numpy.flip(vox, axis=0)
    vox = numpy.flip(vox, axis=1)
    if sink is not None:
        # return ij.writeData(sink, vox.astype('uint16'), LUTs = [ij.grays, ij.greens])
        return io.writeData(sink, vox)
    else:
        return vox

def generateContour(annotation, sink = None, dash_stride = 20, isovalues = None):
    annotation = io.readData(annotation)
    if len(annotation.shape) == 4: # if this is a color image, we need to convert it to grayscale first
        annotation = rgb2gray(annotation)
    if isovalues is None:
        isovalues = numpy.unique(annotation)
    overlay = numpy.zeros(annotation.shape, dtype = annotation.dtype)
    for level in tqdm(isovalues):
        for z in range(annotation.shape[2]):
            contours = measure.find_contours(annotation[...,z], level)
            for contour in contours:
                for start, end in zip(contour[:-1:dash_stride], contour[1::dash_stride]):
                    r0 = int(start[0])
                    c0 = int(start[1])
                    r1 = int(end[0])
                    c1 = int(end[1])
                    rr, cc, val = line_aa(r0, c0, r1, c1)
                    overlay[rr, cc, z] = val * 255
    if sink is not None:
        return io.writeData(sink, overlay)
    else:
        return overlay

def mergeChannels(channels):
    channels = [io.readData(ch) for ch in channels]
    # since some of the channels might already have multiple channels,
    # we need to make sure all the input is of the same size
    channel_dims = [ch.ndim for ch in channels]
    dims_equal = len(set(channel_dims)) == 1
    if dims_equal is False:
        maxdim = max(channel_dims)
        channels = [numpy.reshape(ch, ch.shape + (1,) * (maxdim - ch.ndim)) for ch in channels]
        return numpy.concatenate(channels, axis=-1)
    else:
        return numpy.stack(channels, axis=-1)


def applyToVolumeSections(volume, func = lambda section : section):
    output = numpy.zeros(volume.shape)
    for z in range(volume.shape[2]):
        output[:,:,z] = func(volume[:,:,z])
    return output


def reindexAnnotation(annotation, sink=None):
    annotation = io.readData(annotation)
    uniq, indices = numpy.unique(annotation, return_inverse = True)
    newvals = numpy.arange(uniq.shape[0])
    annotation2 = newvals[indices]
    annotation2 = numpy.reshape(annotation2, annotation.shape)
    # annotation2 = annotation2.astype('uint8')
    if sink is None:
        return annotation2
    else:
        return io.writeData(sink, annotation2)


def convertTiffVolumeToStack(filename, outputFolder):
    data = io.readData(filename)
    if os.path.exists(outputFolder) == False:
        os.makedirs(outputFolder)
    outputFileFormat = os.path.join(outputFolder,'img_Z\d{4}.tif')
    io.writeData(outputFileFormat, data)


def stackVolumes(volumes, sink = None):
    volumes = [io.readData(v) for v in volumes]
    fullvolume = numpy.concatenate(volumes, axis=2)
    if sink is not None:
        return io.writeData(sink, fullvolume)
    else:
        return fullvolume


def mergePoints(points_all, sink=None):
    points_all = [io.readPoints(p) for p in points_all]
    points = numpy.concatenate(points_all, axis=0)
    if sink is not None:
        return io.writePoints(sink, points)
    else:
        return points


def nonzeroPoints(thresholdedData):
    thresholdedData = io.readData(thresholdedData)
    points = numpy.array(numpy.nonzero(thresholdedData)).T
    return points

