import os, uuid, struct, numpy, tifffile
import skimage.io
import ClearMap.IO as io

from settings import ImageJPath, TempFolder

commandQueue = []

def viewData(data, extension = 'nrrd', saveFunc = lambda fn,data : io.writeData(fn,data)):
    if isinstance(data, basestring):
        filename = data;
    else:
        identifier = str(uuid.uuid4())
        filename = os.path.join(TempFolder, identifier + '.' + extension);
        saveFunc(filename, data);
    command = '%s %s' % (ImageJPath, filename);
    os.system(command);

def queueCommand(script_contents):
    commandQueue.append(script_contents)

def runCommandQueue():
    if len(commandQueue) == 0:
        return
    runCommand('\n'.join(commandQueue))
    commandQueue[:] = []

def convertCommand(in_fn, out_fn, format = "PNG", channelToRange = {}):
    script_contents = """
    open("%s");
    """ % in_fn;
    for channel, minAndMax in channelToRange.items():
        if len(channelToRange) > 1:
            script_contents += """
            Stack.setChannel(%d);
            """ % channel
        if minAndMax == 'auto':
            script_contents += """
            run("Enhance Contrast", "saturated=0.35");
            """
        else:
            script_contents += """
            setMinAndMax(%f, %f);
            """ % minAndMax;
    script_contents += """
    saveAs("%s", "%s");
    close();
    """ % (format, out_fn)
    return script_contents

def runCommand(script_contents):
    identifier = str(uuid.uuid4())
    script_path = os.path.join(TempFolder, identifier+'.ijm')
    open(script_path, 'a').write(script_contents)
    command = '%s --headless --run %s' % (ImageJPath, script_path)
    os.system(command)
    os.system('rm %s' % script_path)


grays = numpy.tile(numpy.arange(256, dtype='uint8'), (3, 1))
reds = numpy.zeros((3, 256), dtype='uint8')
reds[0] = numpy.arange(256, dtype='uint8')
greens = numpy.zeros((3, 256), dtype='uint8')
greens[1] = numpy.arange(256, dtype='uint8')
blues = numpy.zeros((3, 256), dtype='uint8')
blues[2] = numpy.arange(256, dtype='uint8')
magentas = numpy.zeros((3, 256), dtype='uint8')
magentas[0] = reds[0]
magentas[2] = blues[2]

def writeData(fname, data, cmaps = None, LUTs = None):
    # REVIEW: this function only works with 2D images with multiple color channels
    data = io.readData(data)
    data = numpy.squeeze(data)
    if data.ndim == 2:
        data = data[..., numpy.newaxis]
    assert(data.ndim == 3)

    if LUTs is None:
        assert(cmaps is not None)
        LUTs = [colormapToRGBArray(cmap) for cmap in cmaps]
    assert(len(LUTs) == data.shape[-1])

    params = {}
    params['byteorder'] = '>'
    params['imagej'] = True
    params['metadata'] = {'mode':'composite'}
    ijtags = buildMetadataTags({'LUTs' : LUTs}, '>')
    params['extratags'] = ijtags
    return io.writeData(fname, data, **params)

def writeDataToTransparentPNG(pngSink, tiffSink, img, mask, LUTs = None, channelToRange = {}):
    img = io.readData(img)
    result = writeData(tiffSink, img.astype('float32'), LUTs = LUTs)
    cmd = convertCommand(result, pngSink, channelToRange = channelToRange)
    runCommand(cmd)
    img = numpy.swapaxes(skimage.io.imread(pngSink), 0, 1)
    if img.shape[2] == 3:
        img = numpy.pad(img, ((0,0),(0,0),(0,1)), 'constant', constant_values = 255)
    img[:,:,3][mask] = 0
    skimage.io.imsave(pngSink, numpy.swapaxes(img, 0, 1))
    return pngSink

def colormapToRGBArray(cmap):
    return cmap(numpy.arange(256), bytes = True)[:,:3].T


# taken from:
#   https://stackoverflow.com/questions/50258287/how-to-specify-colormap-when-saving-tiff-stack/50263336#50263336
#   https://stackoverflow.com/questions/50948559/how-to-save-imagej-tiff-metadata-using-python
def buildMetadataTags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder+('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder+'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder+('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts)//4, bytecounts, True))
