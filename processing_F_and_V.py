# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:29:35 2020

originally from processing_F_and_V jupyter notebook

takes F, v traces from process_Recs, processes them
    spike identification in voltage trace
    split up 1AP, 2AP, etc buckets

@todo:
    generate and save dicts for 1AP, 2AP, etc
    generate F_params dict for each AP where df/f, taus etc are calculated
    enable running in batch mode
    
@author: kolbi
"""


import os, numpy as np
import matplotlib.pyplot as plt
from utils import *
from apsegment import APsegment

# processing F and V (cell-attached) recordings from main_cell_attached_processing

batch = [[r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell4', 'cell4_stim1_', 'cell4_stim1_0001.h5'],\
         [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell4', 'cell4_stim8_', 'cell4_stim8_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell4', 'cell4_stim9_', 'cell4_stim9_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell4', 'cell4_stim10_', 'cell4_stim10_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell4', 'cell4_stim11_', 'cell4_stim11_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell4', 'cell4_stim12_', 'cell4_stim12_0001.h5'],\
             
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell5', 'cell5_stim03_', 'cell5_stim03_0001.h5'],\
         [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim06_', 'cell5_stim06_0001.h5'],\
         [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim07_', 'cell5_stim07_0001.h5'],\
         [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim08_', 'cell5_stim08_0001.h5'],\
         [r'F:\ufGCaMP2pData\martonData\20200322-anm472004\cell5', 'cell5_stim09_', 'cell5_stim09_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell5', 'cell5_stim10_', 'cell5_stim10_0001.h5'],\
         ['F:\\ufGCaMP2pData\\martonData\\20200322-anm472004\\cell5', 'cell5_stim11_', 'cell5_stim11_0001.h5']]

batch = batch[6:]
#%%
# conditioning F and V traces

# initialize bucket array
bucket_1AP = list()
bucket_2AP = list()
bucket_3AP = list()
bucket_4AP = list()
bucket_5AP = list()
bucket_6AP = list()
bucket_7AP = list()
bucket_8AP = list()
    
for b in batch:
    print('ANALYZING NEW BATCH: ' + str(b))
    
    froot = b[0]
    stimPrefix = b[1]
    wsFile = b[2] # not needed for this?
    
    cell_dict = np.load(os.path.join(froot, 'suite2p', 'plane0', 'cell_dict_' + stimPrefix + '.npy'), allow_pickle=True)
    voltage = cell_dict.item()['voltage']
    F = cell_dict.item()['ftrace']
    tArray = cell_dict.item()['tArray']
    tArray_F = cell_dict.item()['tArray_F']
    
    sRate_F = 1 / np.mean(np.diff(tArray_F))
    sRate_v = 1 / np.mean(np.diff(tArray))
    
    print('Num frames  = ' + str(len(F)))
    print('Imaging framerate = ' + str(sRate_F) + ' Hz')
    print('Ephys framerate = ' + str(sRate_v) + ' Hz')
    
    # cut voltage trace to length of F trace
    tArray = tArray[tArray <= np.max(tArray_F)]
    voltage = voltage[:len(tArray)]
    
    # filter V trace to get stable baseline
    v_filt = hpFilter(voltage, 10, 1, sRate_v)
    # plt.plot(tArray, voltage)
    
    plt.figure()
    plt.plot(tArray, v_filt)
    peaks = findAPs(v_filt, sRate_v)
    plt.plot(tArray[peaks], v_filt[peaks], 'o')
    
    print('Number of APs: ' + str(len(peaks)))
    
    
    # AP bucketing
    
    thresh_ms = 500 # ms away from other peaks
    thresh_double_ms = 10 # max distance to next spike
    
    thresh_s = thresh_ms / 1000
    thresh_double_s = thresh_double_ms / 1000
    thresh_samples_v = np.round(thresh_s * sRate_v)
    thresh_samples_f = np.round(thresh_s * sRate_F)
    thresh_double_samples = np.round(thresh_double_s * sRate_v)
    
    peaksT = tArray[peaks] # timestamp (s) of each AP
    
    # classify APs into buckets
    APlists = np.zeros([len(peaks), 8]) # [nPeaks x num buckets to classify]
    
    for idx,_ in enumerate(peaksT):
        
        # identify single APs
        if idx > 0 and idx < len(peaksT) - 1:
            APlists[idx][0] = np.all(np.abs([peaksT[idx-1], peaksT[idx+1]] - peaksT[idx]) >thresh_s) # identify single APs: [t_AP_before, t_AP_after] - t_AP_current > thresh_ms ==> single AP
        
        # identify doubles
        if idx > 0 and idx < len(peaksT) - 3:
            APlists[idx][1] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+2] - peaksT[idx+1] > thresh_s) & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) 
        
        # identify triples
        if idx > 0 and idx < len(peaksT) - 4:
            APlists[idx][2] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+3] - peaksT[idx+2] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) 
        
        # identify quadruples
        if idx > 0 and idx < len(peaksT) - 5:
            APlists[idx][3] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+4] - peaksT[idx+3] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) \
            & (peaksT[idx+3] - peaksT[idx+2] < thresh_double_s)
        
        # identify pent
        if idx > 0 and idx < len(peaksT) - 6:
            APlists[idx][4] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+5] - peaksT[idx+4] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) \
            & (peaksT[idx+3] - peaksT[idx+2] < thresh_double_s) & (peaksT[idx+4] - peaksT[idx+3] < thresh_double_s)
        
        # identify 6-tuples
        if idx > 0 and idx < len(peaksT) - 7:
            APlists[idx][5] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+6] - peaksT[idx+5] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) \
            & (peaksT[idx+3] - peaksT[idx+2] < thresh_double_s) & (peaksT[idx+4] - peaksT[idx+3] < thresh_double_s) \
            & (peaksT[idx+5] - peaksT[idx+4] < thresh_double_s)
        
        # identify 7-tuples
        if idx > 0 and idx < len(peaksT) - 8:
            APlists[idx][6] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+7] - peaksT[idx+6] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) \
            & (peaksT[idx+3] - peaksT[idx+2] < thresh_double_s) & (peaksT[idx+4] - peaksT[idx+3] < thresh_double_s) \
            & (peaksT[idx+5] - peaksT[idx+4] < thresh_double_s) & (peaksT[idx+6] - peaksT[idx+5] < thresh_double_s)
        
        # identify 8-tuples
        if idx > 0 and idx < len(peaksT) - 9:
            APlists[idx][7] = (peaksT[idx] - peaksT[idx-1] > thresh_s) & (peaksT[idx+8] - peaksT[idx+7] > thresh_s) \
            & (peaksT[idx+1] - peaksT[idx] < thresh_double_s) & (peaksT[idx+2] - peaksT[idx+1] < thresh_double_s) \
            & (peaksT[idx+3] - peaksT[idx+2] < thresh_double_s) & (peaksT[idx+4] - peaksT[idx+3] < thresh_double_s) \
            & (peaksT[idx+5] - peaksT[idx+4] < thresh_double_s) & (peaksT[idx+6] - peaksT[idx+5] < thresh_double_s) \
            & (peaksT[idx+7] - peaksT[idx+6] < thresh_double_s)
            
    
    #%%
    APlists = APlists.astype(bool)
    
    for i in range(len(APlists[0])):
        numAPsInBucket = sum(APlists[:,i])
        print(str(i+1) + ' APs: ' + str(numAPsInBucket))
        
        if numAPsInBucket > 0:
            AP_t = peaksT[APlists[:,i]]
            
            # fig = plt.figure()
            # fig.suptitle(str(i+1) + 'APs')
            
            # ax1 = plt.subplot(211)
            # ax2 = plt.subplot(212)
            for j,t in enumerate(AP_t):
                t_idx = np.nonzero(tArray == t)[0][0]
                t_v_segment = (t_idx + np.arange(-1*thresh_samples_v, thresh_samples_v)).astype(int)
                t_v_segment_s = (t_v_segment - np.min(t_v_segment))/sRate_v
                
                v_segment = v_filt[t_v_segment]
                # ax1.plot(t_v_segment_s, v_segment)
                
                f_idx = np.argmin(np.abs(tArray_F - t)) # find closest frame to AP index
                t_f_segment = (f_idx + np.arange(-1*thresh_samples_f, thresh_samples_f)).astype(int)
                t_f_segment_s = (t_f_segment - np.min(t_f_segment))/sRate_F
                
                F_segment = F[t_f_segment]
                # ax2.plot(t_f_segment_s, F_segment)
                # plt.show()
                
                # populate bucketAP1, bucketAP2, ..., bucketAP8 here
                sRates = {'sr_v': sRate_v, 'sr_f': sRate_F}
                
                if i == 0: # AP = 1
                    bucket_1AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 1: # AP = 2
                    bucket_2AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 2: # AP = 3
                    bucket_3AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 3: # AP = 4
                    bucket_4AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 4: # AP = 5
                    bucket_5AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 5: # AP = 6
                    bucket_6AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 6: # AP = 7
                    bucket_7AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                if i == 7: # AP = 8
                    bucket_8AP.append(APsegment(b, F_segment, v_segment, sRates, i+1))
                
                
                # calculate F parameters here
                # Fparams = calcFparams(v_segment, F_segment, sRate_v, sRate_F)
            
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_1AP'), bucket_1AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_2AP'), bucket_2AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_3AP'), bucket_3AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_4AP'), bucket_4AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_5AP'), bucket_5AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_6AP'), bucket_6AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_7AP'), bucket_7AP)
np.save(os.path.join(r'F:\ufGCaMP2pData\Analysis', 'bucket_8AP'), bucket_8AP)