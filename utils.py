# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:51:00 2020
 
Utility functions for processing cell attached + 2P recordings

@author: kolbi
"""

"""
normalizes x to 0 to 1 range
"""

from scipy import signal

def norm0To1(x):
    return (x-min(x)) / (max(x) - min(x))
 

"""
High pass filter
"""
def hpFilter(sig, HPfreq, order, sRate):
    sos = signal.butter(order, HPfreq, 'hp', fs=sRate, output='sos')
    return signal.sosfilt(sos, sig)
    
"""
findAPs: identify APs in voltage trace
@inputs:
        v: voltage trace
        sRate: sampling rate
        cut_s: cut seconds from [start end] of the recording due to artifacts, edge effects

@outputs:
        peak indices

@todo:
        determine min peak height based on SNR
"""
def findAPs(v, sRate, cut_s=[0.1, 1]):
    peaks_, _ = signal.find_peaks(v, height=1, distance=100)
    
    # ignore peaks within the first 100 ms because they may be an artifact of filtering
    # ignore peaks in last 1 s to avoid edge effects
    return peaks_[(peaks_ > sRate * cut_s[0]) & (peaks_ < len(v) - cut_s[1]*sRate)]

