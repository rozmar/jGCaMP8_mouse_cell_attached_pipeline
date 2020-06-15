# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:10:48 2020

based on main_cell_attached_processing jupyter notebook
use this if you already have voltage and mask saved

loads mask, loads WS file, generates ftrace, voltage, and time arrays

@author: labadmin
"""


import numpy as np
import matplotlib.pyplot as plt
from pywavesurfer import ws
from scipy.signal import find_peaks
import os, warnings, tifffile as tiff
import utils

# batch mode -- create batch
# batch: [froot, stimPrefix, ws data file]
batch = [[r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim06_', 'cell5_stim06_0001.h5'],\
        [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim07_', 'cell5_stim07_0001.h5'],\
        [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim08_', 'cell5_stim08_0001.h5'],\
        [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim09_', 'cell5_stim09_0001.h5']]
    
#%%

# batch mode -- load masks, save traces (to correct for timing error)

for b in batch:
    
    print('PROCESSING NEW BATCH: ' + str(b))
    froot = b[0]
    stimPrefix = b[1]
    wsFile = b[2]
    
    # load mask and registered image to get ftrace
    mask = np.load(os.path.join(froot, 'suite2p', 'plane0', 'mask' + stimPrefix + '.npy'), allow_pickle=True)
    chan0regImg = tiff.imread(os.path.join(froot, 'suite2p', 'plane0', 'reg_tif', 'registered_chan0_' + stimPrefix + '.tif'))
    regImg_masked = chan0regImg[:,mask]
    ftrace = np.mean(regImg_masked, axis=1)

    
    data_as_dict = ws.loadDataFile(filename=os.path.join(froot, wsFile), format_string='double' )
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


    # save voltage, ftrace
    cell_dict = {'tArray': tArray, 'voltage': voltage, 'tArray_F': tArray_F, 'ftrace': ftrace}
    np.save(os.path.join(froot, 'suite2p', 'plane0', 'cell_dict_' + stimPrefix + '.npy'), cell_dict)