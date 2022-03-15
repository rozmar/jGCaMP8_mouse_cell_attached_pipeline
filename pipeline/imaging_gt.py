import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_cell_attached as ephys_cell_attached
import pipeline.ephysanal_cell_attached as ephysanal_cell_attached
import pipeline.imaging as imaging
#import pipeline.imaging_gt as imaging_gt
from pipeline.pipeline_tools import get_schema_name
import scipy.ndimage as ndimage
import scipy.signal as signal
schema = dj.schema(get_schema_name('imaging_gt'),locals()) # TODO ez el van baszva
import numpy as np

import scipy
def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f


def rollingfun(y, window = 10, func = 'mean'):
    """
    rollingfun
        rolling average, min, max or std
    
    @input:
        y = array, window, function (mean,min,max,std)
    """
    
    if window >len(y):
        window = len(y)-1
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

def lpFilter(sig, LPfreq, order, sRate, padding = True):
    """
    High pass filter
    -enable padding to avoid edge artefact
    """
    padlength=10000
    sos = signal.butter(order, LPfreq, 'lp', fs=sRate, output='sos')
    if padding: 
        sig_out = signal.sosfilt(sos, np.concatenate([sig[padlength-1::-1],sig,sig[:-padlength-1:-1]]))
        sig_out = sig_out[padlength:-padlength]
    else:
        sig_out = signal.sosfilt(sos, sig)
    return sig_out

@schema
class LFPNeuropilCorrelation(dj.Computed):
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    lfp_neuropil_xcorr = null              : longblob #
    lfp_neuropil_xcorr_time = null         : longblob #
    lfp_neuropil_xcorr_max = null          : double
    lfp_neuropil_xcorr_max_t = null          : double
    lfp_fft=null                           : longblob
    lfp_fft_time=null                           : longblob
    lfp_fft_max = null                     :double
    lfp_fft_max_t = null                     :double
    
    """
    
    def make(self,key):
        #key = {'subject_id': 478348, 'session': 1, 'cell_number': 2, 'sweep_number': 1}
        sweep_now = ephys_cell_attached.Sweep()*ephys_cell_attached.SweepResponse()*ephys_cell_attached.SweepMetadata()&'recording_mode = "current clamp"'&key
        try:
            if len(sweep_now )>0:
                neuropil_key = {'motion_correction_method': 'Suite2P',
                                'roi_type':'Suite2P',
                                'neuropil_number':1,
                                'channel_number':1}
                sweep_start_time =  sweep_now.fetch1('sweep_start_time')
                neuropil_f,frame_times  = (sweep_now*ROISweepCorrespondance()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()&neuropil_key).fetch1('neuropil_f','frame_times')
                
               # print(i)
                trace,sample_rate = sweep_now.fetch1('response_trace','sample_rate')
                ap_max_index = (ephysanal_cell_attached.ActionPotential()*sweep_now).fetch('ap_max_index')
                trace_no_ap = trace.copy()
                step_back = int(sample_rate*.001)
                step_forward = int(sample_rate*.01)
                for ap_max_index_now in ap_max_index:
                    start_idx = np.max([0,ap_max_index_now-step_back])
                    end_idx = np.min([len(trace),ap_max_index_now+step_forward])
                    trace_no_ap[start_idx:end_idx] = trace[start_idx]
                
                cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()&sweep_now).fetch1('cell_recording_start', 'session_time')
                #ephys_time = np.arange(len(trace))/sample_rate
                frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))
                frame_rate = 1/np.diff(frame_times)[0]
                trace_highpass = hpFilter(trace_no_ap, .3, 3, sample_rate, padding = True)
                trace_bandpass = lpFilter(trace_highpass, 200, 3, sample_rate, padding = True)
                trace_truncated = trace_highpass[int(sample_rate)*10:-int(sample_rate)*10]
                percentiles = np.percentile(trace_truncated,[10,90])
                trace_truncated = trace_truncated-percentiles[0]
                trace_truncated = trace_truncated/(np.diff(percentiles))
                out = scipy.signal.welch(trace_truncated, fs=sample_rate,nperseg = 5*sample_rate)
                needed = (out[0]>.1) & (out[0]<50)
                #%
                downsample_factor = int(round(sample_rate / frame_rate))
                trace_bandpass_downsampled = trace_bandpass[int(downsample_factor/2)::downsample_factor]
                trace_bandpass_downsampled = trace_bandpass_downsampled[:len(neuropil_f)]
                std_window = int(frame_rate*.5)
                ephys_std = rollingfun(trace_bandpass_downsampled, window = std_window, func = 'std')
                xcorr = scipy.signal.correlate(scipy.stats.zscore(neuropil_f),scipy.stats.zscore(ephys_std), mode='full', method='direct')
                xcorr_t = np.arange(len(xcorr))/frame_rate-len(frame_times)/frame_rate
                xcorr_needed = (xcorr_t>-5) & (xcorr_t<5)
                key['lfp_neuropil_xcorr'] = xcorr[xcorr_needed]
                key['lfp_neuropil_xcorr_time'] = xcorr_t[xcorr_needed]
                key['lfp_fft'] = out[1][needed]*out[0][needed]
                key['lfp_fft_time'] = out[0][needed]
                
                lfp_idx = np.argmax(key['lfp_fft'])
                xcorr_idx = np.argmax(key['lfp_neuropil_xcorr'])
                
                key['lfp_fft_max'] = key['lfp_fft'][lfp_idx]
                key['lfp_fft_max_t'] = key['lfp_fft_time'][lfp_idx]
                key['lfp_neuropil_xcorr_max'] = key['lfp_neuropil_xcorr'][xcorr_idx]
                key['lfp_neuropil_xcorr_max_t'] = key['lfp_neuropil_xcorr_time'][xcorr_idx]
                
            self.insert1(key,skip_duplicates=True)
        except:
            print('error with {}'.format(key))


        



@schema
class CellBaselineFluorescence(dj.Computed):
    definition = """# 
    -> ephys_cell_attached.Cell
    ---
    cell_baseline_roi_f0_median = null          : double # power corrected median fluorescence during resting - neuropil subtracted    
    cell_baseline_roi_f0_mean = null          : double # power corrected mean fluorescence during resting - neuropil subtracted    
    cell_baseline_roi_f0_std = null          : double # std of power corrected fluorescence during resting - neuropil subtracted    
    """
    def make(self, key):
        #%
        #key = {'subject_id':471991,'session':1,'cell_number':3}
        key_extra = {'channel_number':1,
                     'roi_type':'Suite2P',
                     'motion_correction_method':'Suite2P',
                     'roi_type':'Suite2P',
                     'neuropil_number':1,
                     'neuropil_subtraction':'0.8',
                     'gaussian_filter_sigma':10
                     }
        isis, baselines, power= (ephysanal_cell_attached.APGroup()*CalciumWave.CalciumWaveProperties()*imaging.MoviePowerPostObjective()&key&key_extra).fetch('ap_group_pre_isi','cawave_baseline_f','movie_power') #imaging_gt
        power = np.asarray(power,float)
        baselines_corrected = (baselines*power**2)/40**2
        if sum(isis>2)>5:
            order = np.argsort(isis)[::-1]
            key['cell_baseline_roi_f0_median'] = np.median(baselines_corrected[order][:np.min([10,sum(isis>2)])])
            key['cell_baseline_roi_f0_mean'] = np.mean(baselines_corrected[order][:np.min([10,sum(isis>2)])])
            key['cell_baseline_roi_f0_std'] = np.std(baselines_corrected[order][:np.min([10,sum(isis>2)])])
        self.insert1(key,skip_duplicates=True)
        #plt.plot(isis,baselines_corrected,'ko')
        #%
@schema
class CellDynamicRange(dj.Computed):
    definition = """# 
    -> ephys_cell_attached.Cell
    ---
    cell_f0_median = null          : double # median F0 - neuropil subtracted    
    cell_dff_percentile_100  = null : double
    cell_dff_percentile_99  = null : double
    cell_dff_percentile_95  = null : double
    cell_dff_percentile_90  = null : double
    cell_dff_percentile_80  = null : double
    cell_dff_percentile_70  = null : double
    cell_dff_percentile_60  = null : double
    cell_dff_percentile_50  = null : double
    """
    def make(self, key):
        #%
        fs = 122
        sig_baseline = 40 
        win_baseline = int(60*fs)
        filter_sigma = 0.01
        #key = {'subject_id':471991,'session':1,'cell_number':3}
        key_extra = {'channel_number':1,
                     'roi_type':'Suite2P',
                     'motion_correction_method':'Suite2P',
                     'roi_type':'Suite2P',
                     'neuropil_number':1,
                     'neuropil_subtraction':'0.8',
                     'gaussian_filter_sigma':10
                     }
        #%
        F_list,Fneu_list = (imaging.ROINeuropilTrace()*imaging.ROITrace()*ROISweepCorrespondance()&key&key_extra).fetch('roi_f','neuropil_f')
        dFF_list = []
        f0_list = []
        try:
                
            for F,Fneu in zip(F_list,Fneu_list):
                
                #noncell = iscell[:,0] ==0
                Fcorr= F-Fneu*.7
    
                Flow = ndimage.filters.gaussian_filter(Fcorr,    sig_baseline)
                Flow = ndimage.filters.minimum_filter1d(Flow,    win_baseline)
                Flow = ndimage.filters.maximum_filter1d(Flow,    win_baseline)
                #%
               # dF = Fcorr-Flow
                dFF = (Fcorr-Flow)/Flow
                dFF = ndimage.filters.gaussian_filter(dFF,    filter_sigma*fs)
                dFF_list.append(dFF)
                f0_list.append(Flow)
            dff_percentiles = np.percentile(np.concatenate(dFF_list),[50,60,70,80,90,95,99,100])
            for p,pval in zip([50,60,70,80,90,95,99,100],dff_percentiles):
                key['cell_dff_percentile_{}'.format(p)]=pval
                key['cell_f0_median'] = np.median(np.concatenate(f0_list))
            self.insert1(key,skip_duplicates=True)
        except:
            pass
                

@schema
class MovieDepth(dj.Computed):
    definition = """
    -> imaging.Movie
    ---
    movie_depth=null          : double # depth from pia in microns
    """
    def make(self, key):
        #key = {'subject_id':471991,'session':1,'movie_number':2}
        try:
            movie_hmotors_sample_z = float((imaging.MovieMetaData()&key).fetch1('movie_hmotors_sample_z'))
            if movie_hmotors_sample_z<0:
                try:
                    depth = (ephys_cell_attached.Cell()*ROISweepCorrespondance()&key).fetch1('depth')#imaging_gt.
                    key['movie_depth'] = depth
                except:
                    pass
            else:
                key['movie_depth'] = movie_hmotors_sample_z
        except:
            print('could not extract movie depth for {}'.format(key))
        
        self.insert1(key,skip_duplicates=True)

@schema
class MovieCalciumWaveSNR(dj.Computed):  
    definition = """ # the aim of this table is to find Movies with different SNR
    -> imaging.Movie
    ---
    movie_event_num                   : int     #only events with given minimum inter-event intervals
    movie_ap_num                      : double  #only in the events
    movie_total_ap_num                : double  # all APs in the movie
    movie_mean_cawave_snr_per_ap      : double  
    movie_median_cawave_snr_per_ap    : double
    movie_mean_ap_snr                 : double
    movie_median_ap_snr               : double
    """     
    def make(self, key):
        #%%
        min_pre_isi = .5
        min_post_isi = .5
        preisi_cond = 'ap_group_pre_isi>{}'.format(min_pre_isi)
        postisi_cond = 'ap_group_post_isi>{}'.format(min_post_isi)
# =============================================================================
#         key = {'subject_id':472181,
#                'movie_number':0,
#                }
# =============================================================================
        key_extra = {'channel_number':1,
                     'roi_type':'Suite2P',
                     'motion_correction_method':'Suite2P',
                     'roi_type':'Suite2P',
                     'neuropil_number':1,
                     'neuropil_subtraction':'0.8',
                     'gaussian_filter_sigma':10
                     }
        cawaves = ephysanal_cell_attached.APGroup()*CalciumWave.CalciumWaveProperties()&key&key_extra&preisi_cond&postisi_cond#imaging_gt.
        cawave_snr,ap_group_ap_num,ap_group_min_snr_dv=(cawaves).fetch('cawave_snr','ap_group_ap_num','ap_group_min_snr_dv')
        all_aps = (ephysanal_cell_attached.ActionPotential()*ROISweepCorrespondance())&key#imaging_gt
        #%%
        if len(cawave_snr)>0:
            cawave_snr_per_ap = cawave_snr/ap_group_ap_num
            key['movie_event_num'] = len(cawave_snr)
            key['movie_total_ap_num'] = len(all_aps)
            key['movie_ap_num'] = np.sum(ap_group_ap_num)
            key['movie_mean_cawave_snr_per_ap'] = np.nanmean(cawave_snr_per_ap)
            key['movie_median_cawave_snr_per_ap'] = np.nanmedian(cawave_snr_per_ap)
            key['movie_mean_ap_snr'] = np.nanmean(ap_group_min_snr_dv)
            key['movie_median_ap_snr'] = np.nanmedian(ap_group_min_snr_dv)
            self.insert1(key,skip_duplicates=True)
        
        #%%

@schema
class SessionROIFPercentile(dj.Computed):
    definition = """
    -> imaging.ROI
    -> imaging.MovieChannel
    ---
    roi_f0_percentile          : double
    roi_f0_percentile_value    : double
    """

    
@schema
class SessionROIFDistribution(dj.Computed): # after correcting for power but not depth
    definition = """
    -> experiment.Session
    -> imaging.ROIType
    -> imaging.MotionCorrectionMethod
    ---
    session_f0_percentiles         :blob # 1,2,3,4,5,6...
    session_f0_percentile_values   :blob # F0 values corresponding to each precentile
    """
    def make(self, key):
        #%%
        
        #key = {'subject_id':472181,'channel_number':1,'roi_type':'Suite2P','motion_correction_method':'Suite2P'}
        
        session_f0_percentiles = np.arange(0,101,1)
        rois = imaging.MoviePowerPostObjective()*imaging.ROITrace()&key&'channel_number = 1'
        if len(rois)==0:
            return None
        #print(key)    
        f,power,movie_number,roi_number= rois.fetch('roi_f_min','movie_power','movie_number','roi_number')#
        power = np.asarray(power,float)
        f = np.asarray(f,float)
        f_corr = f/(power**2)
        #needed = np.isnan(f_corr)== False
        session_f0_percentile_values=np.percentile(f_corr,session_f0_percentiles)
        order = np.argsort(f_corr)
        
        f_corr = f_corr[order]
        movie_number = movie_number[order]
        roi_number = roi_number[order]
        percentiles = np.arange(len(f_corr))/len(f_corr)*100
        key_list = list()
        for f_corr_now,movie_number_now,roi_number_now,percentile in zip(f_corr,movie_number,roi_number,percentiles):
            key_now = key.copy()
            key_now['channel_number'] = 1
            key_now['roi_f0_percentile'] = percentile
            key_now['roi_f0_percentile_value'] = f_corr_now
            key_now['movie_number'] = movie_number_now
            key_now['roi_number'] = roi_number_now
            key_list.append(key_now)
        key['session_f0_percentiles'] = session_f0_percentiles
        key['session_f0_percentile_values'] = session_f0_percentile_values
        #%
        SessionROIFPercentile().insert(key_list, allow_direct_insert=True)#i
        self.insert1(key,skip_duplicates=True)
    #%%



@schema
class SessionCalciumSensor(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_calcium_sensor          : varchar(100)
    session_virus_name              : varchar(200)
    session_sensor_expression_time  : int # expression time in days
    session_sensor_max_titer        : double
    session_sensor_min_titer        : double
    session_age                     : int #age of mouse at experiment
    """
    def make(self, key):
        #%
        #key = {'subject_id':471991}
        dob = (lab.Subject()&key).fetch1('date_of_birth')
        surgeries = lab.Surgery()&key
        for surgery in surgeries:
            if len(lab.Surgery.VirusInjection()&surgery)>0:
                virus_injection_time = surgery['start_time']
                virus_ids,dilutions = (lab.Surgery.VirusInjection()&surgery).fetch('virus_id','dilution')
                virus_id = virus_ids[0]
                dilution_min = float(np.min(dilutions))
                dilution_max = float(np.max(dilutions))
                subject_virus_name, original_titer = (lab.Virus()&'virus_id = {}'.format(virus_id)).fetch1('virus_name','titer')
                key['session_sensor_min_titer'] = float(original_titer)/dilution_min
                key['session_sensor_max_titer'] = float(original_titer)/dilution_max
                if '921' in subject_virus_name:
                    subject_calcium_sensor = 'GCaMP7F'
                elif '538' in subject_virus_name:
                    subject_calcium_sensor = 'XCaMPgf'
                elif '456' in subject_virus_name:
                    subject_calcium_sensor = '456'
                elif '686' in subject_virus_name:
                    subject_calcium_sensor = '686'
                elif '688' in subject_virus_name:
                    subject_calcium_sensor = '688'
            else:
                experiment_time = surgery['start_time']
        key['session_virus_name'] = subject_virus_name
        key['session_sensor_expression_time'] = (experiment_time-virus_injection_time).days
        key['session_age'] = (virus_injection_time.date()-dob).days
        key['session_calcium_sensor'] = subject_calcium_sensor
        self.insert1(key,skip_duplicates=True)
                
                
@schema
class ROIDwellTime(dj.Computed):
    definition = """
    -> imaging.ROI
    ---
    roi_dwell_time_multiplier       : double #number of dwell times the roi sees
    """      
    def make(self, key):   
        #%%
        dummy_mask = [13, 12, 13, 12, 12, 12, 12, 11, 12, 11, 11, 11, 11, 11, 11, 10, 11,
                      10, 10, 10, 11, 10,  9, 10, 10, 10,  9, 10,  9,  9, 10,  9,  9,  9,
                      9,  9,  9,  9,  8,  9,  9,  8,  9,  8,  9,  8,  9,  8,  8,  8,  8,
                      9,  8,  8,  8,  8,  8,  7,  8,  8,  8,  8,  7,  8,  8,  7,  8,  7,
                      8,  7,  8,  7,  8,  7,  7,  8,  7,  7,  7,  8,  7,  7,  7,  7,  7,
                      7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  7,  7,  7,  6,  7,
                      7,  7,  6,  7,  7,  6,  7,  6,  7,  7,  6,  7,  6,  7,  6,  7,  6,
                      6,  7,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  6,  7,
                      6,  6,  6,  6,  7,  6,  6,  6,  6,  6,  6,  7,  6,  6,  6,  6,  6,
                      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                      6,  6,  5,  6,  6,  6,  6,  6,  6,  6,  5,  6,  6,  6,  6,  6,  6,
                      5,  6,  6,  6,  6,  5,  6,  6,  6,  5,  6,  6,  6,  6,  5,  6,  6,
                      5,  6,  6,  6,  5,  6,  6,  6,  5,  6,  6,  5,  6,  6,  6,  5,  6,
                      6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,
                      6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  5,  6,  6,  5,  6,  6,  5,
                      6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  5,  6,  6,  5,  6,  6,  5,
                      6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,
                      5,  6,  6,  5,  6,  6,  6,  5,  6,  6,  5,  6,  6,  6,  5,  6,  6,
                      6,  5,  6,  6,  5,  6,  6,  6,  6,  5,  6,  6,  6,  5,  6,  6,  6,
                      6,  5,  6,  6,  6,  6,  6,  6,  5,  6,  6,  6,  6,  6,  6,  6,  5,
                      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
                      6,  6,  6,  6,  6,  6,  6,  7,  6,  6,  6,  6,  6,  6,  7,  6,  6,
                      6,  6,  7,  6,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,
                      7,  6,  6,  7,  6,  7,  6,  7,  6,  7,  7,  6,  7,  6,  7,  7,  6,
                      7,  7,  7,  6,  7,  7,  7,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,
                      7,  7,  7,  7,  7,  7,  7,  8,  7,  7,  7,  8,  7,  7,  8,  7,  8,
                      7,  8,  7,  8,  7,  8,  8,  7,  8,  8,  8,  8,  7,  8,  8,  8,  8,
                      8,  9,  8,  8,  8,  8,  9,  8,  9,  8,  9,  8,  9,  9,  8,  9,  9,
                      9,  9,  9,  9,  9, 10,  9,  9, 10,  9, 10, 10, 10,  9, 10, 11, 10,
                      10, 10, 11, 10, 11, 11, 11, 11, 11, 11, 12, 11, 12, 12, 12, 12, 13,
                      12, 13]
# =============================================================================
#         key = {'subject_id':472179,
#                'session':1,
#                'movie_number':1,
#                'motion_correction_method':'Suite2P',
#                'roi_type':'Suite2P',
#                'roi_number':0}
# =============================================================================
        roi = imaging.ROI()&key
        roi_weights,roi_xpix = roi.fetch1('roi_weights','roi_xpix')
        roi_weights = roi_weights/sum(roi_weights)
        movie_hscan2d_mask=(imaging.MovieMetaData()&key).fetch1('movie_hscan2d_mask')
        if  type(movie_hscan2d_mask) == type(None):
            movie_hscan2d_mask = np.asarray(dummy_mask)
        key['roi_dwell_time_multiplier'] = sum(movie_hscan2d_mask[roi_xpix]*roi_weights)*len(roi_weights)
        self.insert1(key,skip_duplicates=True)
    #%
@schema
class ROISweepCorrespondance(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    -> imaging.ROI
    """

@schema
class ROINeuropilCorrelation(dj.Computed):
    definition = """
    -> ROISweepCorrespondance
    -> imaging.MovieChannel
    -> imaging.ROINeuropil
    ---
    r_neu=null                   : double
    r_neu_corrcoeff=null         : double
    r_neu_pixelnum=null          : int 
    """
    def make(self, key):
        #%%
        
        #print(key)
        try: # try PCA instead of regression and low pass filtering

            neuropil_estimation_decay_time = 3 # - after each AP this interval is skipped for estimating the contribution of the neuropil
            neuropil_estimation_filter_sigma = .05
            #neuropil_estimation_median_filter_time = 4 #seconds
            neuropil_estimation_min_filter_time = 10 #seconds
            session_start_time = (experiment.Session&key).fetch1('session_time')
            cell_start_time = (ephys_cell_attached.Cell&key).fetch1('cell_recording_start')
            sweep_start_time = (ephys_cell_attached.Sweep&key).fetch1('sweep_start_time')
            sweep_timeoffset = (cell_start_time -session_start_time).total_seconds()
            sweep_start_time = float(sweep_start_time)+sweep_timeoffset
            apmaxtimes = np.asarray((ephysanal_cell_attached.ActionPotential()&key).fetch('ap_max_time'),float) +sweep_start_time
            try:
                F = (imaging.ROITrace()&key).fetch1('roi_f')
            except:
                print('roi trace not found for  {}'.format(key))
                #return None
            Fneu = (imaging.ROINeuropilTrace()&key).fetch1('neuropil_f')
            roi_timeoffset = (imaging.ROI()&key).fetch1('roi_time_offset')
            frame_times = (imaging.MovieFrameTimes()&key).fetch1('frame_times') + roi_timeoffset
            framerate = 1/np.median(np.diff(frame_times))
            
            decay_step = int(neuropil_estimation_decay_time*framerate)
            F_activity = np.zeros(len(frame_times))
            for ap_time in apmaxtimes:
                F_activity[np.argmax(frame_times>ap_time)] = 1
            F_activitiy_conv= np.convolve(F_activity,np.concatenate([np.zeros(decay_step),np.ones(decay_step)]),'same')
            F_activitiy_conv= np.convolve(F_activitiy_conv[::-1],np.concatenate([np.zeros(int((neuropil_estimation_filter_sigma*framerate*5))),np.ones(int((neuropil_estimation_filter_sigma*framerate*5)))]),'same')[::-1]
            F_activitiy_conv[:decay_step]=1
            needed =F_activitiy_conv==0
            if sum(needed)> 100:
                F_orig_filt = gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma)
                F_orig_filt_ultra_low =  ndimage.minimum_filter(F_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))

                F_orig_filt_highpass = F_orig_filt-F_orig_filt_ultra_low + F_orig_filt_ultra_low[0]
                Fneu_orig_filt = gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma)
                Fneu_orig_filt_ultra_low =  ndimage.minimum_filter(Fneu_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))

                Fneu_orig_filt_highpass = Fneu_orig_filt-Fneu_orig_filt_ultra_low + Fneu_orig_filt_ultra_low[0]
                p=np.polyfit(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],1)
                R_neuopil_fit = np.corrcoef(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed])
                r_neu_now =p[0]
                key['r_neu'] = r_neu_now
                key['r_neu_corrcoeff'] = R_neuopil_fit[0][1]
                key['r_neu_pixelnum'] = sum(needed)
                
                self.insert1(key,skip_duplicates=True)
        except:
            print('ERROR!!!!! - {}'.format(key))
            self.insert1(key,skip_duplicates=True)

@schema
class CalciumWaveNeuropilHandling(dj.Lookup): #dye
    definition = """
    neuropil_subtraction  :  varchar(20)
    """
    contents = zip(('none','0.7','calculated','0.8'))

@schema
class CalciumWaveFilter(dj.Lookup): #dye
    definition = """
    gaussian_filter_sigma  :  double # in ms
    """
    contents = zip((0,5,10,20,50))

@schema
class CalciumWave(dj.Imported):
    definition = """
    -> ROISweepCorrespondance
    -> ephysanal_cell_attached.APGroup
    -> imaging.MovieChannel # separately for the two channels..
    ---
    cawave_f       : blob # raw f
    cawave_time    : blob # seconds
    """
    class CalciumWaveNeuropil(dj.Part):
        definition = """
        -> master
        -> imaging.ROINeuropil # so ROI and neuropil is also included - neuropil can be different
        ---
        cawave_fneu    : blob # raw f
        """
    class CalciumWaveProperties(dj.Part):
        definition = """
        -> master
        -> imaging.ROINeuropil # so ROI and neuropil is also included - neuropil can be different
        -> CalciumWaveNeuropilHandling
        -> CalciumWaveFilter
        ---
        cawave_rneu                : double
        cawave_baseline_f          : double
        cawave_baseline_f_std      : double
        cawave_baseline_dff_global : double
        cawave_peak_amplitude_f    : double
        cawave_peak_amplitude_dff  : double
        cawave_rise_time           : double
        cawave_snr                 : double
        cawave_half_decay_time     : double # peak to half peak time
        """
@schema
class IngestCalciumWave(dj.Computed):
    definition = """
    -> imaging.Movie
    ---
    cawave_complete : int #1
    """
    def make(self, key):
        
        rise_time_dict = {'GCaMP7F':23,
                          'XCaMPgf':23,
                          '456':4,
                          '686':5,
                          '688':8.5,
                              }
        
        max_neuropil_num = 3 # only neuropil 0 and 1 are needed right now..
        max_channel_num = 1 # only channel 1
        neuropil_correlation_min_corr_coeff=.5
        neuropil_correlation_min_pixel_number=100
        neuropil_r_fallback = .8
        ca_wave_time_back = 1
        ca_wave_time_back_for_baseline = .2
        ca_wave_time_forward = 3
        global_f0_time = 10 #seconds
        max_rise_time = .1 #seconds to find maximum after last AP in a group # 0.1 for pyr
        gaussian_filter_sigmas = CalciumWaveFilter().fetch('gaussian_filter_sigma')#imaging_gt.
        neuropil_subtractions = CalciumWaveNeuropilHandling().fetch('neuropil_subtraction')#imaging_gt.
        groundtruth_rois = ROISweepCorrespondance()&key#imaging_gt.
        
        calcium_wave_list = list()
        calcium_wave_neuropil_list = list()
        calcium_wave_properties_list = list()
        #%
        for groundtruth_roi in groundtruth_rois:
            channels = imaging.MovieChannel()&groundtruth_roi
            apgroups = ephysanal_cell_attached.APGroup()&groundtruth_roi
            session_start_time = (experiment.Session&groundtruth_roi).fetch1('session_time')
            cell_start_time = (ephys_cell_attached.Cell&groundtruth_roi).fetch1('cell_recording_start')
            sweep_start_time = (ephys_cell_attached.Sweep&groundtruth_roi).fetch1('sweep_start_time')
            sweep_timeoffset = (cell_start_time -session_start_time).total_seconds()
            sweep_start_time = float(sweep_start_time)+sweep_timeoffset
            session_calcium_sensor = (SessionCalciumSensor()&groundtruth_roi).fetch1('session_calcium_sensor')
            roi_timeoffset = (imaging.ROI()&groundtruth_roi).fetch1('roi_time_offset')
            frame_times = (imaging.MovieFrameTimes()&groundtruth_roi).fetch1('frame_times') + roi_timeoffset
            framerate = 1/np.median(np.diff(frame_times))
            ca_wave_step_back = int(ca_wave_time_back*framerate)
            ca_wave_step_back_for_baseline = int(ca_wave_time_back_for_baseline*framerate)
            ca_wave_step_back_for_baseline_amplitude = 3
            ca_wave_step_forward =int(ca_wave_time_forward*framerate)
            global_f0_step = int(global_f0_time*framerate)
            ophys_time = np.arange(-ca_wave_step_back,ca_wave_step_forward)/framerate*1000
            #break
            #%
            for channel in channels:
                groundtruth_roi_now = groundtruth_roi.copy()
                groundtruth_roi_now['channel_number'] = channel['channel_number']
                if channel['channel_number']>max_channel_num:
                    continue
                F = (imaging.ROITrace()&groundtruth_roi_now).fetch1('roi_f')
                F_filt_list = list()
                for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered ROI traces
                    if gaussian_filter_sigma>0:
                        F_filt_list.append(gaussFilter(F,sRate = framerate,sigma = gaussian_filter_sigma/1000))
                    else:
                        F_filt_list.append(F)
                #F_filt = np.asarray(F_filt)
                neuropils = np.asarray(list(imaging.ROINeuropil()&groundtruth_roi_now))
                Fneu_list = list()
                Fneu_filt_list = list()
                for neuropil in neuropils: # extracting neuropils
                    groundtruth_roi_neuropil = groundtruth_roi_now.copy()
                    groundtruth_roi_neuropil['neuropil_number'] = neuropil['neuropil_number']
                    try:
                        Fneu_list.append((imaging.ROINeuropilTrace()&groundtruth_roi_neuropil).fetch1('neuropil_f'))   
                        Fneu_filt_list_now = list()
                        for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered neuropil traces
                            if gaussian_filter_sigma>0:
                                Fneu_filt_list_now.append(gaussFilter(Fneu_list[-1],sRate = framerate,sigma = gaussian_filter_sigma/1000))
                            else:
                                Fneu_filt_list_now.append(Fneu_list[-1])
                        Fneu_filt_list.append(np.asarray(Fneu_filt_list_now))
                    except:
                        neuropils = neuropils[:len(Fneu_list)]
                for apgroup in apgroups:
                    groundtruth_roi_now['ap_group_number'] = apgroup['ap_group_number']
                    ap_group_start_time = float(apgroup['ap_group_start_time'])+sweep_timeoffset
                    ap_group_length = float(apgroup['ap_group_end_time']-apgroup['ap_group_start_time'])
                    ca_wave_now = groundtruth_roi_now.copy()
                    cawave_f = np.ones(len(ophys_time))*np.nan
                    start_idx = np.argmin(np.abs(frame_times-ap_group_start_time))
                    idxs = np.arange(start_idx-ca_wave_step_back,start_idx+ca_wave_step_forward)
                    idxs_valid = (idxs>=0) & (idxs<len(F))
                    if sum(idxs_valid)<len(idxs_valid):#np.abs(time_diff)>1/framerate:
                        #print('calcium wave outside recording, skipping {}'.format(groundtruth_roi_now))
                        continue
                    cawave_f[idxs_valid] = F[idxs[idxs_valid]]
                    time_diff = frame_times[start_idx]-ap_group_start_time
                    
                    cawave_time = ophys_time+time_diff*1000
                    ca_wave_now['cawave_f'] = cawave_f
                    ca_wave_now['cawave_time'] = cawave_time
                    calcium_wave_list.append(ca_wave_now)
                    
                    for neuropil,Fneu,Fneu_filt in zip(neuropils,Fneu_list,Fneu_filt_list):
                        ca_wave_neuropil_now = groundtruth_roi_now.copy()
                        ca_wave_neuropil_now['neuropil_number'] = neuropil['neuropil_number']
                        if neuropil['neuropil_number']>max_neuropil_num:# way too many neuropils, don't need it right now
                            continue
                        cawave_fneu = np.ones(len(ophys_time))*np.nan
                        cawave_fneu[idxs_valid] = Fneu[idxs[idxs_valid]]
                        ca_wave_neuropil_now['cawave_fneu'] = cawave_fneu
                        calcium_wave_neuropil_list.append(ca_wave_neuropil_now)
                        
                        ca_wave_properties = groundtruth_roi_now.copy()
                        ca_wave_properties['neuropil_number'] = neuropil['neuropil_number']                    
                        for gaussian_filter_sigma,F_filt,Fneu_filt in zip(gaussian_filter_sigmas,F_filt_list,Fneu_filt): # create filtered ROI traces
                            for neuropil_subtraction in neuropil_subtractions:
                                ca_wave_properties_now = ca_wave_properties.copy()
                                ca_wave_properties_now['gaussian_filter_sigma']=gaussian_filter_sigma
                                ca_wave_properties_now['neuropil_subtraction']=neuropil_subtraction
                                if neuropil_subtraction == 'none':
                                    r_neu = 0
                                elif neuropil_subtraction == 'calculated':
                                    try:
                                        r_neu,r_neu_corrcoeff,r_neu_pixelnum = (ROINeuropilCorrelation()&ca_wave_properties).fetch1('r_neu','r_neu_corrcoeff','r_neu_pixelnum')##imaging_gt.
                                        if r_neu_corrcoeff<neuropil_correlation_min_corr_coeff or r_neu_pixelnum <= neuropil_correlation_min_pixel_number:
                                            1/0 # just to get to the exception
                                    except:
                                        #print('no calculated r_neuropil or low correlation coefficient for {}, resetting to .7'.format(ca_wave_properties_now))
                                        r_neu = neuropil_r_fallback
                                        
                                else:
                                    r_neu = float(neuropil_subtraction)
                                
                                F_subtracted = F_filt-Fneu_filt*r_neu
                                F0_gobal = np.percentile(F_subtracted[np.max([0,int(start_idx-global_f0_step/2)]):np.min([len(F_subtracted),int(start_idx+global_f0_step/2)])],10)
                                cawave_f_now = np.ones(len(ophys_time))*np.nan
                                cawave_f_now[idxs_valid] = F_subtracted[idxs[idxs_valid]]
                                #%
# =============================================================================
#                                 if max_rise_time+ap_group_length>cawave_time[-1]/1000:
#                                     end_idx = len(cawave_time)
#                                 else:
#                                     end_idx =np.argmax(max_rise_time+ap_group_length<cawave_time/1000)
#                                 peak_amplitude_idx = np.argmax(cawave_f_now[ca_wave_step_back:end_idx])+ca_wave_step_back
# =============================================================================
                                peak_amplitude_idx_start = np.argmax(cawave_time>rise_time_dict[session_calcium_sensor])
                                #%
                                baseline_step_num = ca_wave_step_back - int((gaussian_filter_sigma*4)/1000)
                                baseline_step_start = ca_wave_step_back-ca_wave_step_back_for_baseline- int((gaussian_filter_sigma*4)/1000)
                                baseline_step_start_amplitude = ca_wave_step_back-ca_wave_step_back_for_baseline_amplitude- int((gaussian_filter_sigma*4)/1000) 
                                ca_wave_properties_now['cawave_rneu'] =r_neu
                                ca_wave_properties_now['cawave_baseline_f'] = np.nanmean(cawave_f_now[baseline_step_start_amplitude:baseline_step_num])
                                ca_wave_properties_now['cawave_baseline_f_std'] = np.nanstd(cawave_f_now[baseline_step_start:baseline_step_num])
                                ca_wave_properties_now['cawave_baseline_dff_global'] = (ca_wave_properties_now['cawave_baseline_f']-F0_gobal)/F0_gobal
                                ca_wave_properties_now['cawave_peak_amplitude_f'] = np.mean(cawave_f_now[peak_amplitude_idx_start:peak_amplitude_idx_start+3])-ca_wave_properties_now['cawave_baseline_f']
                                ca_wave_properties_now['cawave_peak_amplitude_dff'] = ca_wave_properties_now['cawave_peak_amplitude_f']/ca_wave_properties_now['cawave_baseline_f']
                                ca_wave_properties_now['cawave_rise_time'] = cawave_time[peak_amplitude_idx_start]
                                ca_wave_properties_now['cawave_snr'] = ca_wave_properties_now['cawave_peak_amplitude_f']/ca_wave_properties_now['cawave_baseline_f_std']
                                
                                cawave_f_now_normalized = (cawave_f_now-ca_wave_properties_now['cawave_baseline_f'])/ca_wave_properties_now['cawave_peak_amplitude_f']
                                half_decay_time_idx = np.argmax(cawave_f_now_normalized[peak_amplitude_idx_start+1:]<.5)+peak_amplitude_idx_start+1
                                ca_wave_properties_now['cawave_half_decay_time'] = cawave_time[half_decay_time_idx]-ca_wave_properties_now['cawave_rise_time']
                                
                                calcium_wave_properties_list.append(ca_wave_properties_now)
         #%            
        #with dj.conn().transaction: #inserting one movie
        #print('uploading calcium waves for {}'.format(key)) #movie['movie_name']
        CalciumWave().insert(calcium_wave_list, allow_direct_insert=True)#imaging_gt.
        dj.conn().ping()
        CalciumWave.CalciumWaveNeuropil().insert(calcium_wave_neuropil_list, allow_direct_insert=True)#imaging_gt.
        dj.conn().ping()
        CalciumWave.CalciumWaveProperties().insert(calcium_wave_properties_list, allow_direct_insert=True)#imaging_gt.
        key['cawave_complete'] = 1
        self.insert1(key,skip_duplicates=True)
#%%
        
        
        
@schema
class DoubletCalciumWave(dj.Imported):
    definition = """
    -> ROISweepCorrespondance
    -> ephysanal_cell_attached.APDoublet
    -> imaging.MovieChannel # separately for the two channels..
    ---
    doublet_cawave_f       : blob # raw f
    doublet_cawave_time    : blob # seconds
    """
    class DoubletCalciumWaveNeuropil(dj.Part):
        definition = """
        -> master
        -> imaging.ROINeuropil # so ROI and neuropil is also included - neuropil can be different
        ---
        doublet_cawave_fneu    : blob # raw f
        """
    class DoubletCalciumWaveProperties(dj.Part):
        definition = """
        -> master
        -> imaging.ROINeuropil # so ROI and neuropil is also included - neuropil can be different
        -> CalciumWaveNeuropilHandling
        -> CalciumWaveFilter
        ---
        doublet_cawave_rneu                : double
        doublet_cawave_baseline_f          : double
        doublet_cawave_baseline_f_std      : double
        doublet_cawave_baseline_dff_global : double
        doublet_cawave_peak_amplitude_f    : double
        doublet_cawave_peak_amplitude_dff  : double
        doublet_cawave_rise_time           : double
        doublet_cawave_snr                 : double
        doublet_cawave_half_decay_time     : double # peak to half peak time
        """
@schema
class IngestDoubletCalciumWave(dj.Computed):
    definition = """
    -> imaging.Movie
    ---
    doublet_cawave_complete : int #1
    """
    def make(self, key):
        max_neuropil_num = 3 # only neuropil 0 and 1 are needed right now..
        max_channel_num = 1 # only channel 1
        neuropil_correlation_min_corr_coeff=.5
        neuropil_correlation_min_pixel_number=100
        neuropil_r_fallback = .8
        ca_wave_time_back = 1
        ca_wave_time_back_for_baseline = .2
        ca_wave_time_forward = 3
        global_f0_time = 10 #seconds
        max_rise_time = .8 #seconds to find maximum after last AP in a group # 0.1 for pyr
        gaussian_filter_sigmas = CalciumWaveFilter().fetch('gaussian_filter_sigma')#imaging_gt.
        neuropil_subtractions = CalciumWaveNeuropilHandling().fetch('neuropil_subtraction')#imaging_gt.
        groundtruth_rois = ROISweepCorrespondance()&key#imaging_gt.
        
        calcium_wave_list = list()
        calcium_wave_neuropil_list = list()
        calcium_wave_properties_list = list()
        #%
        for groundtruth_roi in groundtruth_rois:
            channels = imaging.MovieChannel()&groundtruth_roi
            apgroups = ephysanal_cell_attached.APDoublet()&groundtruth_roi
            session_start_time = (experiment.Session&groundtruth_roi).fetch1('session_time')
            cell_start_time = (ephys_cell_attached.Cell&groundtruth_roi).fetch1('cell_recording_start')
            sweep_start_time = (ephys_cell_attached.Sweep&groundtruth_roi).fetch1('sweep_start_time')
            sweep_timeoffset = (cell_start_time -session_start_time).total_seconds()
            sweep_start_time = float(sweep_start_time)+sweep_timeoffset
            roi_timeoffset = (imaging.ROI()&groundtruth_roi).fetch1('roi_time_offset')
            frame_times = (imaging.MovieFrameTimes()&groundtruth_roi).fetch1('frame_times') + roi_timeoffset
            framerate = 1/np.median(np.diff(frame_times))
            ca_wave_step_back = int(ca_wave_time_back*framerate)
            ca_wave_step_back_for_baseline = int(ca_wave_time_back_for_baseline*framerate)
            ca_wave_step_forward =int(ca_wave_time_forward*framerate)
            global_f0_step = int(global_f0_time*framerate)
            ophys_time = np.arange(-ca_wave_step_back,ca_wave_step_forward)/framerate*1000
            #break
            #%
            for channel in channels:
                groundtruth_roi_now = groundtruth_roi.copy()
                groundtruth_roi_now['channel_number'] = channel['channel_number']
                if channel['channel_number']>max_channel_num:
                    continue
                F = (imaging.ROITrace()&groundtruth_roi_now).fetch1('roi_f')
                F_filt_list = list()
                for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered ROI traces
                    if gaussian_filter_sigma>0:
                        F_filt_list.append(gaussFilter(F,sRate = framerate,sigma = gaussian_filter_sigma/1000))
                    else:
                        F_filt_list.append(F)
                #F_filt = np.asarray(F_filt)
                neuropils = np.asarray(list(imaging.ROINeuropil()&groundtruth_roi_now))
                Fneu_list = list()
                Fneu_filt_list = list()
                for neuropil in neuropils: # extracting neuropils
                    groundtruth_roi_neuropil = groundtruth_roi_now.copy()
                    groundtruth_roi_neuropil['neuropil_number'] = neuropil['neuropil_number']
                    try:
                        Fneu_list.append((imaging.ROINeuropilTrace()&groundtruth_roi_neuropil).fetch1('neuropil_f'))   
                        Fneu_filt_list_now = list()
                        for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered neuropil traces
                            if gaussian_filter_sigma>0:
                                Fneu_filt_list_now.append(gaussFilter(Fneu_list[-1],sRate = framerate,sigma = gaussian_filter_sigma/1000))
                            else:
                                Fneu_filt_list_now.append(Fneu_list[-1])
                        Fneu_filt_list.append(np.asarray(Fneu_filt_list_now))
                    except:
                        neuropils = neuropils[:len(Fneu_list)]
                for apgroup in apgroups:
                    groundtruth_roi_now['ap_doublet_number'] = apgroup['ap_doublet_number']
                    ap_doublet_start_time = float(apgroup['ap_doublet_start_time'])+sweep_timeoffset
                    ap_doublet_length = float(apgroup['ap_doublet_length'])
                    ca_wave_now = groundtruth_roi_now.copy()
                    cawave_f = np.ones(len(ophys_time))*np.nan
                    start_idx = np.argmin(np.abs(frame_times-ap_doublet_start_time))
                    idxs = np.arange(start_idx-ca_wave_step_back,start_idx+ca_wave_step_forward)
                    idxs_valid = (idxs>=0) & (idxs<len(F))
                    if sum(idxs_valid)<len(idxs_valid):#np.abs(time_diff)>1/framerate:
                        #print('calcium wave outside recording, skipping {}'.format(groundtruth_roi_now))
                        continue
                    cawave_f[idxs_valid] = F[idxs[idxs_valid]]
                    time_diff = frame_times[start_idx]-ap_doublet_start_time
                    
                    cawave_time = ophys_time+time_diff*1000
                    ca_wave_now['doublet_cawave_f'] = cawave_f
                    ca_wave_now['doublet_cawave_time'] = cawave_time
                    calcium_wave_list.append(ca_wave_now)
                    
                    for neuropil,Fneu,Fneu_filt in zip(neuropils,Fneu_list,Fneu_filt_list):
                        ca_wave_neuropil_now = groundtruth_roi_now.copy()
                        ca_wave_neuropil_now['neuropil_number'] = neuropil['neuropil_number']
                        if neuropil['neuropil_number']>max_neuropil_num:# way too many neuropils, don't need it right now
                            continue
                        cawave_fneu = np.ones(len(ophys_time))*np.nan
                        cawave_fneu[idxs_valid] = Fneu[idxs[idxs_valid]]
                        ca_wave_neuropil_now['doublet_cawave_fneu'] = cawave_fneu
                        calcium_wave_neuropil_list.append(ca_wave_neuropil_now)
                        
                        ca_wave_properties = groundtruth_roi_now.copy()
                        ca_wave_properties['neuropil_number'] = neuropil['neuropil_number']                    
                        for gaussian_filter_sigma,F_filt,Fneu_filt in zip(gaussian_filter_sigmas,F_filt_list,Fneu_filt): # create filtered ROI traces
                            for neuropil_subtraction in neuropil_subtractions:
                                ca_wave_properties_now = ca_wave_properties.copy()
                                ca_wave_properties_now['gaussian_filter_sigma']=gaussian_filter_sigma
                                ca_wave_properties_now['neuropil_subtraction']=neuropil_subtraction
                                if neuropil_subtraction == 'none':
                                    r_neu = 0
                                elif neuropil_subtraction == 'calculated':
                                    try:
                                        r_neu,r_neu_corrcoeff,r_neu_pixelnum = (ROINeuropilCorrelation()&ca_wave_properties).fetch1('r_neu','r_neu_corrcoeff','r_neu_pixelnum')##imaging_gt.
                                        if r_neu_corrcoeff<neuropil_correlation_min_corr_coeff or r_neu_pixelnum <= neuropil_correlation_min_pixel_number:
                                            1/0 # just to get to the exception
                                    except:
                                        #print('no calculated r_neuropil or low correlation coefficient for {}, resetting to .7'.format(ca_wave_properties_now))
                                        r_neu = neuropil_r_fallback
                                        
                                else:
                                    r_neu = float(neuropil_subtraction)
                                
                                F_subtracted = F_filt-Fneu_filt*r_neu
                                F0_gobal = np.percentile(F_subtracted[np.max([0,int(start_idx-global_f0_step/2)]):np.min([len(F_subtracted),int(start_idx+global_f0_step/2)])],10)
                                cawave_f_now = np.ones(len(ophys_time))*np.nan
                                cawave_f_now[idxs_valid] = F_subtracted[idxs[idxs_valid]]
                                #%
                                if max_rise_time+ap_doublet_length>cawave_time[-1]/1000:
                                    end_idx = len(cawave_time)
                                else:
                                    end_idx =np.argmax(max_rise_time+ap_doublet_length<cawave_time/1000)
                                peak_amplitude_idx = np.argmax(cawave_f_now[ca_wave_step_back:end_idx])+ca_wave_step_back
                                #%
                                baseline_step_num = ca_wave_step_back - int((gaussian_filter_sigma*4)/1000)
                                baseline_step_start = ca_wave_step_back-ca_wave_step_back_for_baseline- int((gaussian_filter_sigma*4)/1000)
                                ca_wave_properties_now['doublet_cawave_rneu'] =r_neu
                                ca_wave_properties_now['doublet_cawave_baseline_f'] = np.nanmean(cawave_f_now[baseline_step_start:baseline_step_num])
                                ca_wave_properties_now['doublet_cawave_baseline_f_std'] = np.nanstd(cawave_f_now[baseline_step_start:baseline_step_num])
                                ca_wave_properties_now['doublet_cawave_baseline_dff_global'] = (ca_wave_properties_now['doublet_cawave_baseline_f']-F0_gobal)/F0_gobal
                                ca_wave_properties_now['doublet_cawave_peak_amplitude_f'] = cawave_f_now[peak_amplitude_idx]-ca_wave_properties_now['doublet_cawave_baseline_f']
                                ca_wave_properties_now['doublet_cawave_peak_amplitude_dff'] = ca_wave_properties_now['doublet_cawave_peak_amplitude_f']/ca_wave_properties_now['doublet_cawave_baseline_f']
                                ca_wave_properties_now['doublet_cawave_rise_time'] = cawave_time[peak_amplitude_idx]
                                ca_wave_properties_now['doublet_cawave_snr'] = ca_wave_properties_now['doublet_cawave_peak_amplitude_f']/ca_wave_properties_now['doublet_cawave_baseline_f_std']
                                
                                cawave_f_now_normalized = (cawave_f_now-ca_wave_properties_now['doublet_cawave_baseline_f'])/ca_wave_properties_now['doublet_cawave_peak_amplitude_f']
                                half_decay_time_idx = np.argmax(cawave_f_now_normalized[peak_amplitude_idx:]<.5)+peak_amplitude_idx
                                ca_wave_properties_now['doublet_cawave_half_decay_time'] = cawave_time[half_decay_time_idx]-ca_wave_properties_now['doublet_cawave_rise_time']
                                
                                calcium_wave_properties_list.append(ca_wave_properties_now)
         #%            
        #with dj.conn().transaction: #inserting one movie
        #print('uploading calcium waves for {}'.format(key)) #movie['movie_name']
        DoubletCalciumWave().insert(calcium_wave_list, allow_direct_insert=True)#imaging_gt.
        dj.conn().ping()
        DoubletCalciumWave.DoubletCalciumWaveNeuropil().insert(calcium_wave_neuropil_list, allow_direct_insert=True)#imaging_gt.
        dj.conn().ping()
        DoubletCalciumWave.DoubletCalciumWaveProperties().insert(calcium_wave_properties_list, allow_direct_insert=True)#imaging_gt.
        key['doublet_cawave_complete'] = 1
        self.insert1(key,skip_duplicates=True)       
@schema
class CellMovieDynamicRange(dj.Computed):
    definition = """# 
    -> ROISweepCorrespondance
    ---
    movie_f0_median = null          : double # median F0 - neuropil subtracted    
    movie_dff_percentile_100  = null : double
    movie_dff_percentile_99  = null : double
    movie_dff_percentile_95  = null : double
    movie_dff_percentile_90  = null : double
    movie_dff_percentile_80  = null : double
    movie_dff_percentile_70  = null : double
    movie_dff_percentile_60  = null : double
    movie_dff_percentile_50  = null : double
    """
    def make(self, key):
        #%
        fs = 122
        sig_baseline = 40 
        win_baseline = int(60*fs)
        filter_sigma = 0.01
        #key = {'subject_id':471991,'session':1,'cell_number':3}
        key_extra = {'channel_number':1,
                     'roi_type':'Suite2P',
                     'motion_correction_method':'Suite2P',
                     'roi_type':'Suite2P',
                     'neuropil_number':1,
                     'neuropil_subtraction':'0.8',
                     'gaussian_filter_sigma':10
                     }
        #%
        F_list,Fneu_list = (imaging.ROINeuropilTrace()*imaging.ROITrace()&key&key_extra).fetch('roi_f','neuropil_f')
        dFF_list = []
        f0_list = []
        try:
                
            for F,Fneu in zip(F_list,Fneu_list):
                
                #noncell = iscell[:,0] ==0
                Fcorr= F-Fneu*.7
    
                Flow = ndimage.filters.gaussian_filter(Fcorr,    sig_baseline)
                Flow = ndimage.filters.minimum_filter1d(Flow,    win_baseline)
                Flow = ndimage.filters.maximum_filter1d(Flow,    win_baseline)
                #%
               # dF = Fcorr-Flow
                dFF = (Fcorr-Flow)/Flow
                dFF = ndimage.filters.gaussian_filter(dFF,    filter_sigma*fs)
                dFF_list.append(dFF)
                f0_list.append(Flow)
            dff_percentiles = np.percentile(np.concatenate(dFF_list),[50,60,70,80,90,95,99,100])
            for p,pval in zip([50,60,70,80,90,95,99,100],dff_percentiles):
                key['movie_dff_percentile_{}'.format(p)]=pval
                key['movie_f0_median'] = np.median(np.concatenate(f0_list))
            self.insert1(key,skip_duplicates=True)
        except:
            pass
            
        #plt.plot(isis,baselines_corrected,'ko')
        #%