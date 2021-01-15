# -*- coding: utf-8 -*-

import numpy, os, json, pandas, glob
import xarray as xr
from tqdm import tqdm
import numpy as np
from scipy.stats import linregress, pearsonr

import ClearMap.Analysis.Statistics as stat
import ClearMap.Analysis.Label as lbl
import ClearMap.IO.IO as io
import ClearMap.Alignment.Resampling as rsp
import ClearMap.Analysis.Tools.MultipleComparisonCorrection as mcc

from utility import prettifyHeatmap, sagittalToCoronal, coronalToSagittal, montage, divideSafe
from image_processing import resampleSum_3d
from analysis import propagateValuesUpHierarchy
import imagej as ij
from settings import AtlasFile, AtlasFileLowRes, AnnotationFile

Atlas           = io.readData(AtlasFile)
AtlasLowRes     = io.readData(AtlasFileLowRes)
Annotation      = io.readData(AnnotationFile)
AnnotationIds   = numpy.unique(Annotation)

OutputMontageGrid = (6,5)
OutputMontageGridOffset = 50
pcutoff_voxels = [0.10, 0.05, 0.01, 0.001, 0.0001]
HeatmapResolution = (25,25,25)
HeatmapLowResolution = (100,100,100)

rgb2hex = lambda r,g,b: '#%02x%02x%02x'.upper() % (r,g,b)
def labelToHexColor(label):
  return [rgb2hex(*lbl.Label.color(x)) for x in label];

def labelToLevel(label):
  return [lbl.Label.level(x) for x in label];

def labelToIsLeaf(label):
  return [x in lbl.Label.parents.values() for x in label];

def labelToOrder(label):
  return [lbl.Label.order(x) for x in label];

def labelToCollapse(label):
  return [lbl.Label.collapse[x] for x in label];

writePrettyHeatmapIJ = lambda sink, vox : ij.writeData(sink, vox.astype('uint16'), LUTs = [ij.grays, ij.greens])
writePrettyHeatmapFloatIJ = lambda sink, vox : ij.writeData(sink, vox.astype('float32'), LUTs = [ij.grays, ij.greens])

def writePrettyStatisticsIJ(sink, vox):
  result = ij.writeData(sink, vox.astype('float32'), LUTs = [ij.grays, ij.reds, ij.greens])
  cmd = ij.convertCommand(result, os.path.splitext(result)[0]+'.png', format = "PNG", channelToRange = {2 : (0, min(pcutoff_voxels)), 3 : (0, min(pcutoff_voxels))})
  ij.queueCommand(cmd)

  fn, fext = os.path.splitext(sink)
  result = ij.writeData(fn + '_alt1' + fext, vox.astype('float32'), LUTs = [ij.grays, ij.blues, ij.reds])
  cmd = ij.convertCommand(result, os.path.splitext(result)[0]+'.png', format = "PNG", channelToRange = {2 : (0, min(pcutoff_voxels)), 3 : (0, min(pcutoff_voxels))})
  ij.queueCommand(cmd)

  result = ij.writeData(fn + '_alt2' + fext, vox.astype('float32'), LUTs = [ij.grays, ij.magentas, ij.greens])
  cmd = ij.convertCommand(result, os.path.splitext(result)[0]+'.png', format = "PNG", channelToRange = {2 : (0, min(pcutoff_voxels)), 3 : (0, min(pcutoff_voxels))})
  ij.queueCommand(cmd)

  return result


def readVoxelDataGroup(group):
  g1 = stat.readDataGroup(group)
  if g1.ndim > 4:
    g1 = numpy.array([coronalToSagittal(g[...,1]) for g in g1])
  else:
    g1 = numpy.array([coronalToSagittal(g) for g in g1])
  return g1


def voxelBasedMeanForGroup(group, group_name, outputDirectory):
  
  print('Loading sample heatmaps...')

  if os.path.exists(outputDirectory) == False:
    os.makedirs(outputDirectory)

  g1 = stat.readDataGroup(group)

  if g1.ndim > 4:
    atlas = coronalToSagittal(g1[0][...,0])
    g1 = numpy.array([coronalToSagittal(g[...,1]) for g in g1])
  else:
    atlas = Atlas
    g1 = numpy.array([coronalToSagittal(g) for g in g1])

  # dtype = g1[0].dtype
  # print('Data type is', dtype)

  # Generated average and standard deviation maps
  print('Generating average, median, and standard deviation maps...')

  g1a = numpy.mean(g1, axis = 0)
  # g1m = numpy.median(g1, axis = 0)
  g1s = numpy.std(g1, axis = 0)

  result = prettifyHeatmap(g1a, atlas, sink = os.path.join(outputDirectory, group_name+'_mean.tif'))
  montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, sink = os.path.join(outputDirectory, group_name+'_mean_montage.tif'), write_fn = writePrettyHeatmapIJ)

  result = prettifyHeatmap(g1s, atlas, sink = os.path.join(outputDirectory, group_name+'_std.tif'))
  montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, sink = os.path.join(outputDirectory, group_name+'_std_montage.tif'), write_fn = writePrettyHeatmapIJ)


def voxelBasedGroupAnalysis(group1, group2, outputDirectory):
  
  print('Loading sample heatmaps...')

  if os.path.exists(outputDirectory) == False:
    os.makedirs(outputDirectory)

  g1 = stat.readDataGroup(group1);
  g2 = stat.readDataGroup(group2);

  if g1.ndim > 4:
    atlas = coronalToSagittal(g1[0][...,0])
    g1 = numpy.array([coronalToSagittal(g[...,1]) for g in g1])
    g2 = numpy.array([coronalToSagittal(g[...,1]) for g in g2])
  else:
    atlas = Atlas
    g1 = numpy.array([coronalToSagittal(g) for g in g1])
    g2 = numpy.array([coronalToSagittal(g) for g in g2])

  # dtype = g1[0].dtype
  # print('Data type is', dtype)

  # Generate the p-values map
  print('Generating p-values map...')

  pvals_raw, psign = stat.tTestVoxelization(g1.astype('float'), g2.astype('float'), signed = True, pcutoff = None)
  pvalsc_raw = stat.colorPValues(pvals_raw, psign, positive = [1,0], negative = [0,1])
  prettifyHeatmap(pvalsc_raw.astype('float32'), atlas, sink = os.path.join(outputDirectory, 'pvalues_raw.tif'))
  # io.writeData(os.path.join(outputDirectory, 'tvalues_raw.tif'), tvals_raw.astype('float32'))

  # perform FDR correction as well
  print('Performing FDR correction')
  mask = Annotation > 0
  pvalues_corr = np.ones(pvals_raw.shape)
  pvalues_corr[mask] = mcc.correctPValues(pvals_raw[mask], 'bh')
  pvaluesc_corr = stat.colorPValues(pvalues_corr, psign, positive = [1,0], negative = [0,1])
  prettifyHeatmap(pvaluesc_corr.astype('float32'), atlas, sink = os.path.join(outputDirectory, 'pvalues_FDR_raw.tif'))

  #pcutoff: only display pixels below this level of significance
  for pcutoff in pcutoff_voxels:
    fn_suffix = '%.4f' % pcutoff

    pvals = stat.cutoffPValues(pvals_raw, pcutoff = pcutoff)
    # color the p-values according to their sign (defined by the sign of the difference of the means between the 2 groups)
    # pvalsc = stat.colorPValues(pvals, psign, positive = [0,1], negative = [1,0]);
    pvalsc = stat.colorPValues(pvals, psign, positive = [1,0], negative = [0,1])
    result = prettifyHeatmap(pvalsc.astype('float32'), atlas, sink = os.path.join(outputDirectory, 'pvalues_p'+fn_suffix+'.tif'))
    montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, sink = os.path.join(outputDirectory, 'pvalues_p'+fn_suffix+'_montage.tif'), write_fn = writePrettyStatisticsIJ)
    montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, projection_fn = numpy.max, sink = os.path.join(outputDirectory, 'pvalues_p'+fn_suffix+'_montage_mip.tif'), write_fn = writePrettyStatisticsIJ)

    pvals = stat.cutoffPValues(pvalues_corr, pcutoff = pcutoff)
    pvalsc = stat.colorPValues(pvals, psign, positive = [1,0], negative = [0,1])
    result = prettifyHeatmap(pvalsc.astype('float32'), atlas, sink = os.path.join(outputDirectory, 'pvalues_FDR_p'+fn_suffix+'.tif'))
    montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, sink = os.path.join(outputDirectory, 'pvalues_FDR_p'+fn_suffix+'_montage.tif'), write_fn = writePrettyStatisticsIJ)
    montage(result, OutputMontageGrid, offset = OutputMontageGridOffset, projection_fn = numpy.max, sink = os.path.join(outputDirectory, 'pvalues_FDR_p'+fn_suffix+'_montage_mip.tif'), write_fn = writePrettyStatisticsIJ)


def regionBasedGroupAnalysis(group1, group2, hemisphere, outputDirectory, intensityRow = None, normalizeByTotalCount = False, collapseRegions = None):

  print('Performing region-based statistics on ' + hemisphere + ' hemisphere (row = %s, norm = %d)' % (str(intensityRow), normalizeByTotalCount))

  if os.path.exists(outputDirectory) == False:
    os.makedirs(outputDirectory)

  group1 = [os.path.join(os.path.split(fn)[0], "cells_transformed_to_Atlas_" + hemisphere + ".npy") for fn in group1]
  group2 = [os.path.join(os.path.split(fn)[0], "cells_transformed_to_Atlas_" + hemisphere + ".npy") for fn in group2]

  group1i = [fn.replace('cells_transformed_to_Atlas', 'intensities') for fn in group1];
  group2i = [fn.replace('cells_transformed_to_Atlas', 'intensities') for fn in group2];

  ids, pc1, pc1i = stat.countPointsGroupInRegions(group1, intensityGroup = group1i, intensityRow = (intensityRow if intensityRow is not None else 0), 
                                                  returnIds = True, labeledImage = AnnotationFile, returnCounts = True, collapse = collapseRegions);
  pc2, pc2i      = stat.countPointsGroupInRegions(group2, intensityGroup = group2i, intensityRow = (intensityRow if intensityRow is not None else 0), 
                                                  returnIds = False, labeledImage = AnnotationFile, returnCounts = True, collapse = collapseRegions);

  original_ids = ids
  ids, pc1 = propagateValuesUpHierarchy(original_ids, pc1, sortIds = True)
  _, pc2   = propagateValuesUpHierarchy(original_ids, pc2, sortIds = True)

  # print(ids.shape)
  # print(pc1.shape)
  # print(pc1i.shape)

  if intensityRow is not None:

    _, pc1i  = propagateValuesUpHierarchy(original_ids, pc1i, sortIds = True)
    _, pc2i  = propagateValuesUpHierarchy(original_ids, pc2i, sortIds = True)

    if normalizeByTotalCount:
      # pc1 = pc1i / pc1
      # pc2 = pc2i / pc2
      pc1 = divideSafe(pc1i, pc1)
      pc2 = divideSafe(pc2i, pc2)
    else:
      pc1 = pc1i
      pc2 = pc2i

  pvals, psign = stat.tTestPointsInRegions(pc1, pc2, pcutoff = None, signed = True);
  # pvalsi, psigni = stat.tTestPointsInRegions(pc1i, pc2i, pcutoff = None, signed = True, equal_var = True);

  # iid = pvals < 1;
  # iid = pvalsi < 1;
  iid = ids > 0
  
  ids0 = ids[iid];
  pc10 = pc1[iid];
  pc20 = pc2[iid];
  psign0 = psign[iid];
  pvals0 = pvals[iid];
  qvals0 = mcc.estimateQValues(pvals0);

  # write results to csv
  dtypes = [('id','int64'),('mean1','f8'),('std1','f8'),('mean2','f8'),('std2','f8'),('pvalue', 'f8'),('qvalue', 'f8'),('psign', 'int64')];
  for i in range(len(group1)):
      dtypes.append(('count1_%d' % i, 'f8'));
  for i in range(len(group2)):
      dtypes.append(('count2_%d' % i, 'f8'));   
  dtypes.append(('name', 'a256'));
  dtypes.append(('acronym', 'a256'));
  dtypes.append(('level', 'int64'))
  dtypes.append(('color','a256'))
  dtypes.append(('is_leaf','int64'))
  dtypes.append(('order','int64'))
  dtypes.append(('collapse','int64'))
  dtypes.append(('in_annot','int64'))

  table = numpy.zeros(ids0.shape, dtype = dtypes)
  table["id"] = ids0;
  table["mean1"] = pc10.mean(axis = 1);
  table["std1"] = pc10.std(axis = 1);
  table["mean2"] = pc20.mean(axis = 1);
  table["std2"] = pc20.std(axis = 1);
  table["pvalue"] = pvals0;
  table["qvalue"] = qvals0;

  table["psign"] = psign0;
  for i in range(len(group1)):
      table["count1_%d" % i] = pc10[:,i];
  for i in range(len(group2)):
      table["count2_%d" % i] = pc20[:,i];
  table["name"] = lbl.labelToName(ids0);
  table["acronym"] = lbl.labelToAcronym(ids0);
  table["level"] = labelToLevel(ids0);
  table["color"] = labelToHexColor(ids0);
  table["is_leaf"] = labelToIsLeaf(ids0);
  table["order"] = labelToOrder(ids0);
  table["collapse"] = labelToCollapse(ids0);
  table["in_annot"] = [x in AnnotationIds for x in ids0];

  #sort by qvalue
  ii = numpy.argsort(pvals0);
  tableSorted = table.copy();
  tableSorted = tableSorted[ii];

  df = pandas.DataFrame(tableSorted, columns = [str(item) for item in table.dtype.names])
  df.to_csv(os.path.join(outputDirectory, 'counts_table.csv'), header=True, sep=',', index=False)
