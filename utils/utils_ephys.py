from scipy import signal, interpolate
from scipy.io import loadmat
import scipy.ndimage as ndimage
import numpy as np
from pywavesurfer import ws
import matplotlib.pyplot as plt

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
    

def remove_stim_artefacts_without_stim(trace, sample_rate):
    #%%
    #trace_old = trace
    trace = trace.copy()
    cut_out_time = .005
    cut_out_step = int(cut_out_time*sample_rate)
    sRate_needed = 500
    downsample_factor = int(sample_rate/sRate_needed)
    thresh = 10 #pA/ms
    v_down = trace[:len(trace):downsample_factor]
    v_filt = gaussFilter(v_down,sRate_needed,sigma = .01)
    stim_amplitudes = list()
    stim_idxs = list()
    if (max(v_filt)-min(v_filt))*10**12>200:# there is a stimulus probably
        dv=np.diff(v_filt)*sRate_needed*10**9
        peaks,ampl = signal.find_peaks(np.abs(dv), height=thresh)
        peaks = peaks*downsample_factor +int(downsample_factor/2)
        for peak in peaks:
            difi  = trace[peak-cut_out_step] - trace[peak+cut_out_step]
            trace[peak-cut_out_step:] =trace[peak-cut_out_step:]+ difi
            trace[peak-cut_out_step:peak+cut_out_step] =trace[peak-cut_out_step-1]
            stim_amplitudes.append(difi)
            stim_idxs.append(peak)
     #%%       
    return trace,stim_idxs,stim_amplitudes

def remove_stim_artefacts_with_stim(trace,sample_rate,squarepulse_indices,time_back = .002, time_width = .001):
# =============================================================================
#     time_back = .002
#     time_width = .001
# =============================================================================
    window_steps = [int((time_back+time_width/2)*sample_rate),int((time_back-time_width/2)*sample_rate)]
    step_back = int(time_back*sample_rate)
    trace_corrected = trace.copy()
    for idx in squarepulse_indices:
        baseline_value = np.median(trace_corrected[idx-np.max(window_steps):idx-np.min(window_steps)])
        step_value = np.median(trace_corrected[idx+np.min(window_steps):idx+np.max(window_steps)])
        difi = step_value-baseline_value
        trace_corrected[idx-step_back:idx+step_back]=step_value
        trace_corrected[idx-step_back:]=trace_corrected[idx-step_back:]-difi
    return trace_corrected


def findAPs(trace, sample_rate,recording_mode, SN_min = 5,method = 'diff'):
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
# =============================================================================
#     method = 'diff'
#     SN_min = 5
# =============================================================================
    
    if recording_mode=='voltage clamp':
        min_amplitude = 30*10**-12
        min_ap_integral = 2e-14 #As
        min_halfwidth = 0.025
    else:
        min_amplitude = .0002
        min_ap_integral = 1e-7 #Vs
        min_halfwidth = 0.05
    if method == 'diff':
        if recording_mode=='voltage clamp':
            trace=trace*-1
            #%
        trace_orig = trace.copy()
        trace_filt = gaussFilter(trace,sample_rate,sigma = .00005)  
        trace_diff=np.diff(trace_filt)*sample_rate
        window = int(sample_rate*.1)
        step=int(window/2)
        starts = np.arange(0,len(trace_diff)-window,step)
        stds = list()
        for start in starts:
            stds.append(np.std(trace_diff[start:start+window]))
        stds_roll = rollingfun(stds,100,'min')
        stds_roll = rollingfun(stds_roll,500,'mean')
        #%
        trace_diff_scaled = np.copy(trace_diff)
        noise_level = np.ones(len(trace_diff)+1)
        for start,std in zip(starts,stds_roll):
            trace_diff_scaled[start:start+window]=trace_diff[start:start+window]/std
            noise_level[start:start+window]=std
        trace_diff_scaled[start:]=trace_diff[start:]/std
        noise_level[start:]=std
        
        
        baseline_length = .001#s long
        baseline_step = int(sample_rate*baseline_length)
        baseline_idxs = list()
        peak_idxs = list()
        peak_snrs = list()
        peak_snrs_v = list()
        peak_snrs_dv = list()
        peak_amplitudes_left = list()
        peak_amplitudes_right = list()
        half_widths = list()
        full_widths = list()
        rise_times = list()
        ahps_amplitudes=list()
        peak_to_trough = list()
        ap_integrals = list()
        rise_time = list()
        trace_diff_scaled_temp = trace_diff_scaled.copy()
        while np.max(trace_diff_scaled_temp)>SN_min:
            peak_idx = np.argmax(trace_diff_scaled_temp)
            start_idx = np.max([peak_idx-1,0])
            
            while start_idx >5 and np.min(trace_filt[start_idx-5:start_idx])<trace_filt[start_idx]:
                start_idx -= 2
            end_idx = peak_idx+1
            while end_idx < len(trace_filt)-5 and np.max(trace_filt[end_idx:end_idx+5])>trace_filt[end_idx]:
                end_idx += 1
            if end_idx > 5:
                end_idx=start_idx+np.argmax(trace[start_idx:end_idx+5])
            peak_idx_real = end_idx
            peak_idxs.append(end_idx)
            peak_snrs.append(np.max(trace_diff_scaled_temp))
            end_idx = np.min([end_idx+5,len(trace)-1])
            while end_idx < len(trace_filt)-5 and np.min(trace_filt[end_idx:end_idx+5])<trace_filt[end_idx]:
                end_idx += 2
            ahp_end_idx = end_idx + 2
            while ahp_end_idx < len(trace_filt)-10 and np.max(trace_filt[ahp_end_idx:ahp_end_idx+10])>trace_filt[ahp_end_idx]:
                ahp_end_idx += 2
            
            trace_diff_scaled_temp[start_idx:end_idx] = 0
            peak_to_trough.append((end_idx-peak_idx_real)/sample_rate)
            peak_value_filt = np.max(trace_filt[start_idx:end_idx])
            peak_value_raw = np.max(trace[start_idx:end_idx])
            baseline_idxs.append(np.arange(np.max([start_idx-baseline_step,0]),start_idx))
            baseline_left = np.mean(trace_filt[np.max([start_idx-baseline_step,0]):start_idx])
            baseline_right =  np.mean(trace_filt[end_idx:np.min([end_idx+baseline_step,len(trace_filt)])])
            amplitude_left = peak_value_filt-baseline_left
            amplitude_right =  peak_value_filt-baseline_right
            baseline_v = trace_filt[np.max([start_idx-baseline_step,0]):start_idx]
            if len(baseline_v)>10:
                p = np.polyfit(np.arange(len(baseline_v)),baseline_v,1)
                baseline_v =baseline_v -np.polyval(p,np.arange(len(baseline_v)))
            peak_snrs_v.append(amplitude_left/np.std(baseline_v))
            peak_snrs_dv.append((trace_diff[peak_idx])/np.std(trace_diff[np.max([start_idx-baseline_step,0]):start_idx]))
            ahp = trace_filt[end_idx]
            ahps_amplitudes.append(ahp-baseline_left)
            wave_for_integral = np.abs((trace[start_idx:ahp_end_idx]-baseline_left))
            wave_for_integral[wave_for_integral<.05*amplitude_left] = 0
            wave_integral = sum(wave_for_integral)/sample_rate            
            wave_normalized = (trace[start_idx:ahp_end_idx]-baseline_left)/(peak_value_raw-baseline_left)
            f = interpolate.interp1d(np.arange(0,len(wave_normalized),1),wave_normalized)
            wave_normalized_upsampled = f(np.arange(0,len(wave_normalized)-1,.1))            
            half_width = sum(wave_normalized_upsampled>.5)/sample_rate*100
            full_width = sum(wave_normalized_upsampled>.05)/sample_rate*100            
            wave_normalized_for_rise = (trace[start_idx:peak_idxs[-1]]-baseline_left)/(peak_value_raw-baseline_left)
            f = interpolate.interp1d(np.arange(0,len(wave_normalized_for_rise),1),wave_normalized_for_rise)
            wave_normalized_for_rise_upsampled = f(np.arange(0,len(wave_normalized_for_rise)-1,.1))  
            rise_time = sum(wave_normalized_for_rise_upsampled>.05)/sample_rate*100
            peak_amplitudes_left.append(amplitude_left)
            peak_amplitudes_right.append(amplitude_right)
            half_widths.append(half_width)
            full_widths.append(full_width)
            rise_times.append(rise_time)
            ap_integrals.append(wave_integral)
            #break
            #plt.plot(trace_filt[start_idx:end_idx])
        #%%
        baseline_idxs = np.asarray(baseline_idxs)
        peak_idxs = np.asarray(peak_idxs)
        peak_snrs = np.asarray(peak_snrs)
        peak_snrs_v = np.asarray(peak_snrs_v)
        peak_snrs_dv = np.asarray(peak_snrs_dv)
        peak_amplitudes_left = np.asarray(peak_amplitudes_left)
        peak_amplitudes_right = np.asarray(peak_amplitudes_right)
        half_widths = np.asarray(half_widths)
        ahps_amplitudes=np.asarray(ahps_amplitudes)
        peak_to_trough_time = np.asarray(peak_to_trough)
        full_widths = np.asarray(full_widths)
        rise_times = np.asarray(rise_times)
        ap_integrals = np.asarray(ap_integrals)
        amplitude_ratios = peak_amplitudes_left/peak_amplitudes_right
        needed = (peak_snrs_v>SN_min) & (peak_snrs_dv>SN_min) & (peak_amplitudes_left>min_amplitude) & (amplitude_ratios<=2) & (amplitude_ratios>0)& (half_widths>=min_halfwidth) & (ap_integrals>min_ap_integral)
        #%%
        #get rid of obviously not APs
        order = np.argsort(peak_idxs[needed])
        baseline_idxs = baseline_idxs[needed][order]
        peak_idxs = peak_idxs[needed][order]
        peak_snrs = peak_snrs[needed][order]
        peak_snrs_v = peak_snrs_v[needed][order]
        peak_snrs_dv = peak_snrs_dv[needed][order]
        peak_amplitudes_left = peak_amplitudes_left[needed][order]
        peak_amplitudes_right = peak_amplitudes_right[needed][order]
        half_widths = half_widths[needed][order]
        full_widths = full_widths[needed][order]
        rise_times = rise_times[needed][order]
        ap_integrals = ap_integrals[needed][order]
        ahps_amplitudes= ahps_amplitudes[needed][order]
        peak_to_trough_time = peak_to_trough_time[needed][order]
        amplitude_ratios = amplitude_ratios[needed][order]
        #%%
        if len(peak_idxs)>0:
            #enforce similar amplitude and halfwidth APs
            isis = np.concatenate([[1],np.diff(peak_idxs)/sample_rate])
            validamplitude = np.median(peak_amplitudes_left[peak_snrs_dv>SN_min*2])
            validhw = np.median(half_widths[peak_snrs_dv>SN_min*2])
            idx_to_keep = list()
            idx_to_del=list()
            for idx, (amplitude,hw,isi) in enumerate(zip(peak_amplitudes_left,half_widths,isis)):
                if isi<.005:
                    minratio = .25
                else:
                    minratio =.5
                if amplitude<validamplitude*minratio or hw>validhw*(1/minratio):
                    idx_to_del.append(idx)
                else:
                    idx_to_keep.append(idx)
                    validamplitude=amplitude
                    validhw=hw
           
            baseline_idxs = baseline_idxs[idx_to_keep]         
            peak_idxs = peak_idxs[idx_to_keep]
            peak_snrs = peak_snrs[idx_to_keep]
            peak_snrs_v = peak_snrs_v[idx_to_keep]
            peak_snrs_dv = peak_snrs_dv[idx_to_keep]
            peak_amplitudes_left = peak_amplitudes_left[idx_to_keep]
            peak_amplitudes_right = peak_amplitudes_right[idx_to_keep]
            half_widths = half_widths[idx_to_keep]
            
            full_widths = full_widths[idx_to_keep]
            rise_times = rise_times[idx_to_keep]
            ap_integrals = ap_integrals[idx_to_keep]
            
            ahps_amplitudes= ahps_amplitudes[idx_to_keep]
            peak_to_trough_time = peak_to_trough_time[idx_to_keep]
            amplitude_ratios = amplitude_ratios[idx_to_keep]
            #%
            ap_dict = {'peak_idx':peak_idxs,
                       'peak_times':peak_idxs/sample_rate,
                       'peak_snr_v':peak_snrs_v,
                       'peak_snr_dv':peak_snrs_dv,
                       'peak_amplitude_left':peak_amplitudes_left,
                       'peak_amplitude_right':peak_amplitudes_right,
                       'half_width':half_widths,                   
                       'full_width':full_widths,
                       'rise_time':rise_times,
                       'ap_integral':ap_integrals,
                       'ahp_amplitude':ahps_amplitudes*-1,
                       'peak_to_through':peak_to_trough_time*1000,
                       'isi_left':np.concatenate([[None],np.diff(peak_idxs)/sample_rate]),
                       'isi_right':np.concatenate([np.diff(peak_idxs)/sample_rate,[None]]),
                       'baseline_idxs':baseline_idxs}
        else:
            ap_dict = dict()
            ap_dict['peak_idx'] = []

# =============================================================================
#                                           #%%
#             time = np.arange(len(trace))/sample_rate
#             plt.plot(time,trace)
#             plt.plot(time[peak_idxs],trace[peak_idxs],'ro')
# =============================================================================
    
    #%%
    return ap_dict
    


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

def load_wavesurfer_file(WS_path):
    #%%
    ws_data = ws.loadDataFile(filename=WS_path, format_string='double' )
    units = np.array(ws_data['header']['AIChannelUnits']).astype(str)
    channelnames = np.array(ws_data['header']['AIChannelNames']).astype(str)
    commandchannel = np.asarray(ws_data['header']['AOChannelNames'],str) =='Command'
    commandunit = np.asarray(ws_data['header']['AOChannelUnits'],str)[np.where(commandchannel)[0][0]]
    ephysidx = (channelnames == 'Voltage').argmax()
    if 'A' in commandunit:
        recording_mode = 'CC'
    else:
        recording_mode = 'VC'
    framatimesidx = (channelnames == 'FrameTrigger').argmax()
    

    
    keys = list(ws_data.keys())
    del keys[keys=='header']
    if len(keys)>1:
        print('MULTIPLE SWEEPS! HANDLE ME!!')
        time.sleep(1000)
    sweep = ws_data[keys[0]]
    sRate = ws_data['header']['AcquisitionSampleRate'][0][0]
    voltage = sweep['analogScans'][ephysidx,:]
   #%
    if units[ephysidx] == 'mV':
        voltage =voltage/1000
        units[ephysidx] = 'V'
    if units[ephysidx] == 'pA':
        voltage =voltage/10**12
        units[ephysidx] = 'A'
    if units[ephysidx] == 'nA':
        voltage =voltage/10**9
        units[ephysidx] = 'A'   

    
    frame_trigger = sweep['analogScans'][framatimesidx,:]
    timestamp = ws_data['header']['ClockAtRunStart']
    
      #% 
# =============================================================================
#             plt.plot(voltage)
#             plt.plot(peaks,voltage[peaks],'ro')
# =============================================================================
            
            
    #%%
    return voltage, frame_trigger, sRate, recording_mode, timestamp  , units[ephysidx]
