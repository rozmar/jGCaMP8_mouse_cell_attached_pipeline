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
    """
    def make(self, key):
        #%%
        
        #print(key)
        try:
            #%%
            #key =  {'subject_id': 479120, 'session': 1, 'cell_number': 8, 'sweep_number': 0, 'movie_number': 29, 'motion_correction_method': 'Suite2P', 'roi_type': 'Suite2P', 'roi_number': 89, 'channel_number': 2, 'neuropil_number': 0}
            neuropil_estimation_decay_time = 3 # - after each AP this interval is skipped for estimating the contribution of the neuropil
            neuropil_estimation_filter_sigma = .05
            neuropil_estimation_median_filter_time = 4 #seconds
            
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
                try:
                    F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
                except:
                    F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
                F_orig_filt_highpass = F_orig_filt-F_orig_filt_ultra_low + F_orig_filt_ultra_low[0]
                Fneu_orig_filt = gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma)
                try:
                    Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
                except:
                    Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
                Fneu_orig_filt_highpass = Fneu_orig_filt-Fneu_orig_filt_ultra_low + Fneu_orig_filt_ultra_low[0]
                p=np.polyfit(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],1)
                R_neuopil_fit = np.corrcoef(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed])
                r_neu_now =p[0]
                key['r_neu'] = r_neu_now
                key['r_neu_corrcoeff'] = R_neuopil_fit[0][1]
                #print(key)
                #%%
                self.insert1(key,skip_duplicates=True)
        except:
            print('ERROR!!!!! - {}'.format(key))

@schema
class CalciumWaveNeuropilHandling(dj.Lookup): #dye
    definition = """
    neuropil_subtraction  :  varchar(20)
    """
    contents = zip(('none','0.7','calculated'))

@schema
class CalciumWaveFilter(dj.Lookup): #dye
    definition = """
    gaussian_filter_sigma  :  double # in ms
    """
    contents = zip((0,5,10))

@schema
class CalciumWave(dj.Imported):
    definition = """
    -> ephysanal_cell_attached.APGroup
    -> imaging.ROI
    -> imaging.MovieChannel
    ---
    cawave_f       : blob # raw f
    cawave_fneu    : blob # raw f
    cawave_time    : blob # seconds
    cawave_r_neu   : double #calculated neuropil value
    """
    class CalciumWaveProperties(dj.Part):
        definition = """
        -> master
        -> CalciumWaveNeuropilHandling
        -> CalciumWaveFilter
        ---
        cawave_baseline_f          : double
        cawave_baseline_dff        : double
        cawave_peak_amplitude_f    : double
        cawave_peak_amplitude_dff  : double
        cawave_rise_time           : double
        cawave_snr                 : double
        """
    