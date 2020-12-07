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
def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f
#%%
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
    movie_total_ap_num                    : double  #only in the events
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
# =============================================================================
# @schema
# class ROINeuropilCorrelationCorrected(dj.Computed):   
# definition = """
#     -> ROISweepCorrespondance
#     -> imaging.MovieChannel
#     -> imaging.ROINeuropil
#     ---
#     r_neu_corrected=null          : double
#     """
# =============================================================================
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
            #%%
            #key =  {'subject_id': 479571, 'session': 1, 'cell_number': 2, 'sweep_number': 2, 'movie_number': 10, 'motion_correction_method': 'Suite2P', 'roi_type': 'Suite2P', 'roi_number': 1, 'channel_number': 1, 'neuropil_number': 0}
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
# =============================================================================
#                 try:
#                     F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#                 except:
#                     F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
# =============================================================================
                F_orig_filt_highpass = F_orig_filt-F_orig_filt_ultra_low + F_orig_filt_ultra_low[0]
                Fneu_orig_filt = gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma)
                Fneu_orig_filt_ultra_low =  ndimage.minimum_filter(Fneu_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))
# =============================================================================
#                 try:
#                     Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#                 except:
#                     Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
# =============================================================================
                Fneu_orig_filt_highpass = Fneu_orig_filt-Fneu_orig_filt_ultra_low + Fneu_orig_filt_ultra_low[0]
                p=np.polyfit(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],1)
                R_neuopil_fit = np.corrcoef(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed])
                r_neu_now =p[0]
                key['r_neu'] = r_neu_now
                key['r_neu_corrcoeff'] = R_neuopil_fit[0][1]
                key['r_neu_pixelnum'] = sum(needed)
                
                #print(key)
                #%%
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
        max_neuropil_num = 3 # only neuropil 0 and 1 are needed right now..
        max_channel_num = 1 # only channel 1
        neuropil_correlation_min_corr_coeff=.5
        neuropil_correlation_min_pixel_number=100
        neuropil_r_fallback = .8
        ca_wave_time_back = 1
        ca_wave_time_back_for_baseline = .2
        ca_wave_time_forward = 3
        global_f0_time = 10 #seconds
        max_rise_time = .02 #seconds to find maximum after last AP in a group # 0.1 for pyr
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
                                if max_rise_time+ap_group_length>cawave_time[-1]/1000:
                                    end_idx = len(cawave_time)
                                else:
                                    end_idx =np.argmax(max_rise_time+ap_group_length<cawave_time/1000)
                                peak_amplitude_idx = np.argmax(cawave_f_now[ca_wave_step_back:end_idx])+ca_wave_step_back
                                #%
                                baseline_step_num = ca_wave_step_back - int((gaussian_filter_sigma*4)/1000)
                                baseline_step_start = ca_wave_step_back-ca_wave_step_back_for_baseline- int((gaussian_filter_sigma*4)/1000)
                                ca_wave_properties_now['cawave_rneu'] =r_neu
                                ca_wave_properties_now['cawave_baseline_f'] = np.nanmean(cawave_f_now[baseline_step_start:baseline_step_num])
                                ca_wave_properties_now['cawave_baseline_f_std'] = np.nanstd(cawave_f_now[baseline_step_start:baseline_step_num])
                                ca_wave_properties_now['cawave_baseline_dff_global'] = (ca_wave_properties_now['cawave_baseline_f']-F0_gobal)/F0_gobal
                                ca_wave_properties_now['cawave_peak_amplitude_f'] = cawave_f_now[peak_amplitude_idx]-ca_wave_properties_now['cawave_baseline_f']
                                ca_wave_properties_now['cawave_peak_amplitude_dff'] = ca_wave_properties_now['cawave_peak_amplitude_f']/ca_wave_properties_now['cawave_baseline_f']
                                ca_wave_properties_now['cawave_rise_time'] = cawave_time[peak_amplitude_idx]
                                ca_wave_properties_now['cawave_snr'] = ca_wave_properties_now['cawave_peak_amplitude_f']/ca_wave_properties_now['cawave_baseline_f_std']
                                
                                cawave_f_now_normalized = (cawave_f_now-ca_wave_properties_now['cawave_baseline_f'])/ca_wave_properties_now['cawave_peak_amplitude_f']
                                half_decay_time_idx = np.argmax(cawave_f_now_normalized[peak_amplitude_idx:]<.5)+peak_amplitude_idx
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
    