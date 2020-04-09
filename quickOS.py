# -*- coding: utf-8 -*-
"""
quickOS: quickly plots orientation selectivity (OS) bar plot of voltage trace

@usage: 
    from command prompt: python quickOS.py <WS h5 path> <vis stim mat path>
    e.g.:  `python quickOS.py test\testWS.h5 test\test.mat`

@inputs:
        WS_path: path to wavesurfer file
        vis_path: path to visual stim file

@todo:
    open first detected sweep as opposed to sweep_0001 just in case
    error bars in mean spike plot
    
@author: kolbi
"""
import sys, os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from pywavesurfer import ws
from utils import *


if __name__ == "__main__":
    # print(f"Arguments count:" + str(len(sys.argv)))
    WS_path = sys.argv[1]
    vis_path = sys.argv[2]
    
    
    # check path names!
    # if not os.path.isfile(WS_path):
    #     raise ValueError(WS_path + " is not a proper path!")
    if not os.path.isfile(vis_path):
        raise ValueError(vis_path + " is not a proper path!")
    
    vis = processVisMat(vis_path)
    
    
    ws_data = ws.loadDataFile(filename=WS_path, format_string='double' )
    sweep = ws_data['sweep_0001']
    sRate_v = ws_data['header']['AcquisitionSampleRate'][0][0]
    voltage = sweep['analogScans'][0,:]
    vis_frames = sweep['analogScans'][2,:]
    
    # get vis stim frame indices
    vis_idx = getEdges(vis_frames, minDist_samples=20, diffThresh=.02, edge = 'positive')
    
    # check to make sure num detected frames matches num frames in mat file
    nFrames_ws = len(vis_idx)
    nFrames_vis = vis['total_frames']
    if not (nFrames_ws == nFrames_vis):
        raise ValueError('Number of frames in WS file (' + str(nFrames) + ') does not match num frames in MAT file (' + str(nFrames_vis) + ')')
    
    tArray = np.arange(len(voltage))/sRate_v
    v_filt = hpFilter(voltage, 10, 1, sRate_v)
    
    # AP indices in voltage trace
    AP_idx_inV = findAPs(v_filt, sRate_v)
    
    
    tArray_frame = tArray[AP_idx_inV]
    plt.subplot(311)
    plt.plot(tArray, v_filt) # plot v trace
    plt.plot(tArray_frame, v_filt[AP_idx_inV], 'o') # show peaks
    
    
    angle = vis['angle']
    nAngles = len(angle)
    spikes = np.zeros(nAngles)
    
    vis_stim_fs = vis['fs']# visual stim sampling rate
    stim_dur_samples = vis['stim_dur'] * vis_stim_fs
    gray_dur_samples = vis['gray_dur'] * vis_stim_fs
    stim_init_samples = vis['stim_init']
    
    for i,start_stim_idx in enumerate(stim_init_samples):
        end_stim_idx = start_stim_idx + stim_dur_samples-1
        
        # print('start stim idx: ' + str(start_stim_idx))
        # print('end stim idx: ' + str(end_stim_idx))

        nSpikes = np.sum((AP_idx_inV >= vis_idx[start_stim_idx]) & (AP_idx_inV <= vis_idx[end_stim_idx]))
        # print('angle: ' + str(angle[i]) + '   nSpikes: ' + str(nSpikes) + '   start: ' + str(tArray[vis_idx[start_stim_idx]]) + '   end: ' + str(tArray[vis_idx[end_stim_idx]]))
    
        # calculate num spikes in interval

        spikes[i] = nSpikes
    
    # collect all spikes and plot
    uniqueAngles = np.unique(angle)
    totalSpikes = np.zeros_like(uniqueAngles)
    meanSpikes = np.zeros_like(uniqueAngles)
    for i,a in enumerate(uniqueAngles):
        
        totalSpikes[i] = sum(spikes[angle == a])
        meanSpikes[i] = np.mean(spikes[angle == a])
        
    
    ax = plt.subplot(312)
    plt.bar(uniqueAngles, totalSpikes, width=30)
    ax.set_xlabel("Angle")
    ax.set_ylabel("Total num spikes")
    ax.set_xticks(uniqueAngles)
    
    ax = plt.subplot(313)
    plt.bar(uniqueAngles, meanSpikes, width=30)
    ax.set_xlabel("Angle")
    ax.set_ylabel("mean num spikes")
    ax.set_xticks(uniqueAngles)
    
    plt.show()