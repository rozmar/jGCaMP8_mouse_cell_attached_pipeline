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

class APsegment(object):
    def __init__(self, batchDict, Fsegment, Vsegment, sRates, nAPs):
        
        self.batchDict = batchDict
        self.Fsegment = Fsegment
        self.Vsegment = Vsegment
        self.sr_v = sRates['sr_v']
        self.sr_f = sRates['sr_f']
        self.nAPs = nAPs
        
        
