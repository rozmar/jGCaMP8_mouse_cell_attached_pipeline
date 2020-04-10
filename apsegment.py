# -*- coding: utf-8 -*-


"""
APsegment: isolated segment of F and V recording

@inputs:
        batchDict: [froot, stimPrefix, ws data file] dict
        Fsegment: fluorescence segment
        Vsegment: voltage segment
        sRates: ['sr_v': __ , 'sr_f': __] dict of sampling rates for v and f trace
        nAPs: number of APs detected in V trace
        
        
@author: kolbi
"""

import numpy as np

class APsegment(object):
    
    
    def __init__(self, batchDict, Fsegment, Vsegment, sRates, nAPs):
        
        self.batchDict = batchDict
        self.Fsegment = Fsegment
        self.Vsegment = Vsegment
        self.sr_v = sRates['sr_v']
        self.sr_f = sRates['sr_f']
        self.nAPs = nAPs
        
        # assumes response starts in the middle of the trace
        self.mdpt = int(len(self.Fsegment)/2)
        self.F0 = self.calcF0()
        self.dFF = self.calcdFF()
        
        (self.dFFpeak, self.dFFpeak_t_s) = self.dFFpeak()
    
    def calcF0(self):
        """
        calculates F0 of Fsegment
        """
        
        # F0: Fsegment[mdpt-offset-dur:mdpt-offset]
        offset = 5
        dur = 5 # num samples to take
        return np.mean(self.Fsegment[self.mdpt-offset-dur:self.mdpt-offset])
    
    def calcdFF(self):
        """
        return delta F / F0 trace
        """
        
        return (self.Fsegment - self.F0) / self.F0
    
    def dFFpeak(self):
        """
        calculate peak dF/F of Fsegment
        return dF/F_peak and t_dF/F_peak (time from mdpt to peak)
        """
        
        afterMid = self.Fsegment[self.mdpt:]
        max_idx_ = np.argmax(afterMid)
        max_idx = max_idx_ + self.mdpt
        
        return (self.Fsegment[max_idx], max_idx_*self.sr_f)
    
    def isGood(self):
        """
        flag to determine whether the APsegment is good
            e.g. noise too high?
        """