import os
import tempfile

ImageJPath                  = '/oak/stanford/groups/ljinhy/dadgarki/downloads/Fiji.app/ImageJ-linux64'

PathReg                     = '../atlasfiles/'
AtlasFile                   = os.path.join(PathReg, 'template_25.tif')
AtlasFileReg                = os.path.join(PathReg, 'template_25_Left_fullWD.tif')
AtlasFileLowRes             = os.path.join(PathReg, 'template_100.tif')
AnnotationFile              = os.path.join(PathReg, 'annotation_25_full_2017.nrrd')
AnnotationFileReindexed     = os.path.join(PathReg, 'annotation_25_full_reindexed.nrrd')

TempFolder = '/scratch/users/dadgarki/temp'
if TempFolder is None:
        TempFolder = '/tmp'
tempfile.tempdir = TempFolder

