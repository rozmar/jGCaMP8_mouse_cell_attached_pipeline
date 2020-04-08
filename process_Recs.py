# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:04:17 2020

based on main_cell_attached_processing jupyter notebook
@author: ilya kolb

TODO
    run this code in batch mode
    
    
"""


import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import glob, os, warnings
from suite2p import run_s2p
from pywavesurfer import ws
from scipy.signal import find_peaks
from polygondrawer import PolygonDrawer
from utils import *

# 1 -- input animal, cell, stim number, run suite2p
froot = r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell4' # root dir with tiffs and ws file
stimPrefix = "cell4_stim1_"
wsFile = 'cell4_stim1_0001.h5'

isDirFroot = os.path.isdir(froot)
if not isDirFroot:
    raise ValueError("froot is not a directory!")

files = glob.glob(os.path.join(froot, stimPrefix + "*.tif"))
for n,f in enumerate(files):
    files[n] = os.path.basename(f)

print("Number of files: " + str(len(files)))
if len(files) == 0:
    raise ValueError("No files found!")

if len(files) > 100: # there are typically 96 files. if more may be wrong filter
    warnings.warn('Num files: ' + str(len(files)) + ' more than expected (96)')
    
# ops = run_s2p.default_ops() # populates ops with the default options
db = {
      'h5py': [], # a single h5 file path
      'h5py_key': 'data',
      'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
      'data_path': [froot], # a list of folders with tiffs 
                                             # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
                                            
      'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
      'fast_disk': 'C:/BIN', # string which specifies where the binary file will be stored (should be an SSD)
      'tiff_list': files
    }


ops = np.load(r"F:\ufGCaMP2pData\suite2p_opts.npy", allow_pickle=True).item()
_ = run_s2p.run_s2p(ops=ops, db=db)

# 2 -- load registered file and generate max projection to feed to mask maker
chan0RegImgDir = os.path.join(froot, 'suite2p', 'plane0', 'reg_tif', 'file000_chan0.tif')
chan1RegImgDir = os.path.join(froot, 'suite2p', 'plane0', 'reg_tif_chan2', 'file000_chan1.tif')

# load channel 0 registered image
chan0regImg = tiff.imread(chan0RegImgDir)
chan0maxIntensity = np.max(chan0regImg, 0)
tiff.imsave(os.path.join(froot, 'suite2p', 'plane0', 'maxproj_chan0_' + stimPrefix + '.tif'),chan0maxIntensity)

# load channel 1 registered image
chan1regImg = tiff.imread(chan1RegImgDir)
chan1maxIntensity = np.max(chan1regImg, 0)
tiff.imsave(os.path.join(froot, 'suite2p', 'plane0', 'maxproj_chan1_' + stimPrefix + '.tif'),chan1maxIntensity)
del(chan1regImg)

# renames the saved registered streams to names that won't be overwritten
os.replace(chan0RegImgDir, os.path.join(froot, 'suite2p', 'plane0', 'reg_tif', 'registered_chan0_' + stimPrefix + '.tif'))
os.replace(chan1RegImgDir, os.path.join(froot, 'suite2p', 'plane0', 'reg_tif_chan2', 'registered_chan1_' + stimPrefix + '.tif'))
os.remove(os.path.join(froot, 'suite2p', 'ops1.npy')) # must remove ops.npy to make sure directory will run again in suite2p


# 3 -- outlining cell from max intensity projection
# ============================================================================


scaledImg = (chan0maxIntensity - np.min(chan0maxIntensity)) / (np.max(chan0maxIntensity) - np.min(chan0maxIntensity))

pd = PolygonDrawer("Polygon", scaledImg)
mask = pd.run()

mask = mask.astype(bool) # convert to boolean mask array


print(str(np.shape(mask)))


# save the mask
np.save(os.path.join(froot, 'suite2p', 'plane0', 'mask' + stimPrefix + '.npy'), mask)

# 4 -- mask off cell, get ftrace, load WS file

# load WS file
data_as_dict = ws.loadDataFile(filename=os.path.join(froot, wsFile), format_string='double' )


regImg_masked = chan0regImg[:,mask]
ftrace = np.mean(regImg_masked, axis=1)


# to get analog channels scaled in float64 :

sweep = data_as_dict['sweep_0001']
voltage = sweep['analogScans'][0,:]
frames = sweep['analogScans'][1,:]
sRate = data_as_dict['header']['AcquisitionSampleRate'][0][0]

tArray = np.arange(len(voltage))/sRate

plt.figure(figsize=[10,4])
plt.plot(tArray, norm0To1(voltage))

# extract frame occurences from frames array
frames_diff = np.diff(frames) > 1

# find peaks in diff array to get frame indices
peaks, _ = find_peaks(frames_diff, height=.5, distance=100)

imgFrateRate = sRate / np.mean(np.diff(peaks))

print('Num fire signals in WS = ' + str(len(peaks)))
print('Num frames in tiffs = ' + str(len(ftrace)))
print('Imaging framerate = ' + str(imgFrateRate) + ' Hz')

tArray_F = tArray[peaks]

# if there are more frames than images, truncate the time array
if len(tArray_F) > len(ftrace):
    warnings.warn('More frames detected in WS trace than true frames. Truncating')
    tArray_F = tArray_F[:len(ftrace)]
elif len(tArray_F) < len(ftrace):
    warnings.warn('number of frames is greater than number of frame triggers! Truncating')
    ftrace = ftrace[:len(tArray_F)]

plt.plot(tArray_F, norm0To1(ftrace))
plt.show()


# save voltage, ftrace, mask, maybe other variables?
cell_dict = {'tArray': tArray, 'voltage': voltage, 'tArray_F': tArray_F, 'ftrace': ftrace}
np.save(os.path.join(froot, 'suite2p', 'plane0', 'cell_dict_' + stimPrefix + '.npy'), cell_dict)