# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:51:00 2020
 
Utility functions for processing cell attached + 2P recordings

@author: kolbi
"""



from scipy import signal
from scipy.io import loadmat
import numpy as np

def norm0To1(x):
    """
    normalizes x to 0 to 1 range
    """
    return (x-min(x)) / (max(x) - min(x))
 


def hpFilter(sig, HPfreq, order, sRate):
    """
    High pass filter
    """
    sos = signal.butter(order, HPfreq, 'hp', fs=sRate, output='sos')
    return signal.sosfilt(sos, sig)
    

def findAPs(v, sRate, cut_s=[0.1, 1]):
    """
    findAPs: identify APs in voltage trace
    @inputs:
            v: voltage trace (should already be filtered)
            sRate: sampling rate
            cut_s: cut seconds from [start end] of the recording due to artifacts, edge effects
    
    @outputs:
            peak indices
    
    @todo:
            determine min peak height based on SNR
    """
    peaks_, _ = signal.find_peaks(v, height=1, distance=100)
    
    # ignore peaks within the first 100 ms because they may be an artifact of filtering
    # ignore peaks in last 1 s to avoid edge effects
    return peaks_[(peaks_ > sRate * cut_s[0]) & (peaks_ < len(v) - cut_s[1]*sRate)]



def getEdges(sig, minDist_samples = 50, diffThresh = 1, edge = 'positive'):
    """
    getEdges: returns indices of rising/falling edges, for parsing frame fire signal
    
    @inputs:
        sig: [Nx1] numpy array from which to find edges
        minDist_samples: minimum distance between found edges (in samples)
        diffThresh: threshold for detecting edge
        edge: 'positive'
    @outputs:
        indices of edges
    """
    # take diff of signal and threshold it
    sig_diff = (np.diff(sig) > diffThresh) if edge is 'positive' else (np.diff(sig) > -1*diffThresh)
    
    # find peaks in diff array to get indices
    peaks, _ = signal.find_peaks(sig_diff, height=diffThresh/2, distance=minDist_samples)
    
    # sometimes first frame is rising edge, doesn't get detected by peak finder
    # if so, insert 0th frame into array
    if sig_diff[0] > diffThresh:
        peaks = np.insert(peaks, 0, 0)
    
    return peaks


def processVisMat(vis_mat_path):
    """
    processVisMat
        load visual stim file and get relevant parameters
    
    @input:
        vis_mat_path: path to vis stim mat file
    
    @outputs:
        vis
            ncondition: 8
            ntrial: 5
            fs: 122
            stim_dur: 2
            gray_dur: 2
            total_frames: 19520
            total_time: 160
            stim_init: [1×40 double]
            numStim: 40
            angle: [1×40 double]
            trials: {1×40 cell}
    """
    mat = loadmat(vis_mat_path)['vis'][0][0]
    
    vis = {}
    vis['ncondition'] = mat[0][0][0]
    vis['ntrial'] = mat[1][0][0]
    vis['fs'] = mat[2][0][0]
    vis['stim_dur'] = mat[3][0][0]
    vis['gray_dur'] = mat[4][0][0]
    vis['total_frames'] = mat[5][0][0]
    vis['total_time'] = mat[6][0][0]
    vis['stim_init'] = mat[7][0]
    vis['numStim'] = mat[8][0][0]
    vis['angle'] = mat[9][0]
    vis['trials'] = mat[10]
    
    return vis