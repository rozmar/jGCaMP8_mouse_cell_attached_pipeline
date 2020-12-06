# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:51:00 2020
 
Utility functions for processing cell attached + 2P recordings

@author: kolbi
"""



from scipy import signal
from scipy.io import loadmat
import scipy.ndimage as ndimage
import numpy as np



def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    
    y = np.concatenate([y[window::-1],y,y[:-1*window:-1]])
    ys = list()
    for idx in range(window):    
        ys.append(np.roll(y,idx-round(window/2)))
    if func =='mean':
        out = np.nanmean(ys,0)[window:-window]
    elif func == 'min':
        out = np.nanmin(ys,0)[window:-window]
    elif func == 'max':
        out = np.nanmax(ys,0)[window:-window]
    elif func == 'std':
        out = np.nanstd(ys,0)[window:-window]
    elif func == 'median':
        out = np.nanmedian(ys,0)[window:-window]
    else:
        print('undefinied funcion in rollinfun')
    return out



def norm0To1(x):
    """
    normalizes x to 0 to 1 range
    """
    return (x-min(x)) / (max(x) - min(x))
 


def hpFilter(sig, HPfreq, order, sRate, padding = True):
    """
    High pass filter
    -enable padding to avoid edge artefact
    """
    padlength=10000
    sos = signal.butter(order, HPfreq, 'hp', fs=sRate, output='sos')
    if padding: 
        sig_out = signal.sosfilt(sos, np.concatenate([sig[padlength-1::-1],sig,sig[:-padlength-1:-1]]))
        sig_out = sig_out[padlength:-padlength]
    else:
        sig_out = signal.sosfilt(sos, sig)
    return sig_out

def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f
    

def findAPs(v, sRate, SN_min = 10,refracter_period = 1,method = 'diff'):
    """
    findAPs: identify APs in voltage trace
    @inputs:
            v: voltage trace (should already be filtered)
            sRate: sampling rate
            SN_min: minimum SNR to be involved as a spike
    
    @outputs:
            peak indices
            peak SNR
    """
    #%%
    if method == 'diff':
        v_orig = v.copy()
        v=np.diff(v)*sRate
        window = int(sRate*.1)
        step=int(window/2)
        starts = np.arange(0,len(v)-window,step)
        stds = list()
        for start in starts:
            stds.append(np.std(v[start:start+window]))
        stds_roll = rollingfun(stds,100,'min')
        stds_roll = rollingfun(stds_roll,500,'mean')
        #%
        v_scaled = np.copy(v)
        noise_level = np.ones(len(v)+1)
        for start,std in zip(starts,stds_roll):
            v_scaled[start:start+window]=v[start:start+window]/std
            noise_level[start:start+window]=std
        v_scaled[start:]=v[start:]/std
        noise_level[start:]=std
        peaks_, properties = signal.find_peaks(v_scaled,height = SN_min, distance=int((refracter_period/1000)*sRate))
        snr = properties['peak_heights']
        # ignore suspiciuously low or high 'spikes'
        if len(peaks_)>10:
            snr_rolling = rollingfun(snr, window = 5, func = 'median')
            needed_spikes = (snr>snr_rolling*.1) & (snr<snr_rolling*10)
            peaks_=peaks_[needed_spikes]
            snr =snr[needed_spikes]
        peaks_v = list()
        for peak_dv in peaks_:
            peaks_v.append(peak_dv+np.argmax(v_orig[peak_dv:int(.001*sRate)+peak_dv]))
    else:
        window = int(sRate*10)
        step=int(window/2)
        starts = np.arange(0,len(v)-window,step)
        stds = list()
        for start in starts:
            stds.append(np.median(np.abs((v[start:start+window]))/.6745))
        #%
        v_scaled = np.copy(v)
        noise_level = np.ones(len(v))
        for start,std in zip(starts,stds):
            v_scaled[start:start+window]=v[start:start+window]/std
            noise_level[start:start+window]=std
        v_scaled[start:]=v[start:]/std
        noise_level[start:]=std
        peaks_v, properties = signal.find_peaks(v_scaled,height = SN_min, distance=int((refracter_period/1000)*sRate))
        snr = properties['peak_heights']
        
    #%%
    return peaks_v, snr, noise_level
    


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
    sig_diff = (np.diff(sig) > diffThresh) if edge == 'positive' else (np.diff(sig) < -1*diffThresh)
    
    # find peaks in diff array to get indices
    peaks, _ = signal.find_peaks(sig_diff, height=diffThresh/2, distance=minDist_samples)
    
    # sometimes first frame is rising edge, doesn't get detected by peak finder
    # if so, insert 0th frame into array
    if sig_diff[0] > diffThresh:
        peaks = np.insert(peaks, 0, 0)
    
    return peaks


def processVisMat(vis_mat_path):
    #%%
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
    vis['trials_list'] = dict()
    keys = vis['trials'][0][0].dtype.fields.keys()
    for key in keys:
        vis['trials_list'][key]=list()
        for trial in vis['trials'][0]:
            vis['trials_list'][key].append(trial[key][0][0][0][0])
            
        
        
    #%%
    return vis


