import numpy as np
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
import scipy.ndimage as ndimage
import scipy.stats as stats
from scipy import signal
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f

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

def remove_nans_from_trace(bin_mean):
    while sum(np.isnan(bin_mean))>0:
        startpos = np.argmax(np.isnan(bin_mean))
        endpos = np.argmin(np.isnan(bin_mean[startpos:]))+startpos-1
        nanlen = endpos-startpos+1
        startval = bin_mean[startpos-1]
        endval = bin_mean[endpos+1]
        if startpos ==0: 
            startval = endval
        if endpos == len(bin_mean)-1:
            endval = startval
        vals = (np.arange(nanlen+2)/(nanlen+1))*(endval-startval)+startval
        
        bin_mean[startpos:endpos+1]=vals[1:-1]
    return bin_mean
    

def generate_superresolution_trace(x_conc,y_conc,bin_step,bin_size,function = 'mean',dogaussian = False):
    # TODO use gaussian averaging instead of boxcar
    bin_centers = np.arange(np.min(x_conc),np.max(x_conc),bin_step)
    bin_mean = list()
    bin_gauss = list()
    bin_median = list()
    bin_std = list()
    bin_n = list()
    for bin_center in bin_centers:
        idxes = (x_conc>bin_center-bin_size/2) & (x_conc<bin_center+bin_size/2)
        bin_mean.append(np.mean(y_conc[idxes]))
        bin_median.append(np.median(y_conc[idxes]))
        bin_std.append(np.std(y_conc[idxes]))
        bin_n.append(np.sum(idxes))
        if  dogaussian:
            idxes_gauss = (x_conc>bin_center-bin_size) & (x_conc<bin_center+bin_size)
            gauss_f = stats.norm(bin_center,bin_size/4)
            gauss_w = gauss_f.pdf(x_conc[idxes_gauss])
            gauss_w = gauss_w/sum(gauss_w)
            bin_gauss.append(np.sum(y_conc[idxes_gauss]*gauss_w))
        
    bin_mean = remove_nans_from_trace(bin_mean)
    bin_median = remove_nans_from_trace(bin_median)
    bin_gauss = remove_nans_from_trace(bin_gauss)
    if function== 'median':
        trace = bin_median
    else:
        trace = bin_mean
    out_dict = {'bin_centers':bin_centers,
                'trace_mean':bin_mean,
                'trace_median':bin_median,
                'trace_std':bin_std,
                'trace_gauss':bin_gauss,
                'trace_n':bin_n}
    if function == 'all':
        return out_dict
    else:
        return bin_centers,trace

def add_scalebar_to_fig(ax,
                        axis = 'x',
                        size = 50,
                        conversion = 0.59,
                        unit = r'$\mu$m',
                        color = 'white',
                        location = 'top right',
                        linewidth = 3):
    
    
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    realsize = size/conversion
    scalevalues = np.asarray([0,realsize])
    
    if axis =='x':
        if 'top' in location:
            yoffset_text = ylim[1]-np.diff(ylim)[0]*.1*2
        elif 'bottom' in location:
            yoffset_text = ylim[0]+np.diff(ylim)[0]*.1*2 
        else:
            yoffset_text = np.mean(ylim)
        if 'right' in location:
            xoffset_text = xlim[1]-np.diff(xlim)[0]*.1-realsize/2
        elif 'left' in location:
            xoffset_text = xlim[0]+np.diff(xlim)[0]*.1+realsize/2
        else:
            xoffset_text = np.mean(xlim)
        
    if axis =='y':
        if 'top' in location:
            yoffset_text = ylim[1]-np.diff(ylim)[0]*.1-realsize/2
        elif 'bottom' in location:
            yoffset_text = ylim[0]+np.diff(ylim)[0]*.1+realsize/2
        else:
            yoffset_text = np.mean(ylim)
        if 'right' in location:
            xoffset_text = xlim[1]-np.diff(xlim)[0]*.1*2
        elif 'left' in location:
            xoffset_text = xlim[0]+np.diff(xlim)[0]*.1*2
        else:
            xoffset_text = np.mean(xlim)
    if 'top' in location:
        if axis == 'y':
            yoffset = ylim[1]-np.diff(ylim)[0]*.1-realsize
            #yoffset_text = yoffset+realsize/2
        else:
            yoffset = ylim[1]-np.diff(ylim)[0]*.1
            #yoffset_text = yoffset-np.diff(ylim)[0]*.1
    elif 'bottom' in location:
        #if axis == 'y':  
        yoffset = ylim[0]+np.diff(ylim)[0]*.1#yoffset_text = yoffset+realsize/2
    else:
        if axis == 'y':
            yoffset = np.mean(ylim) - realsize/2
        else:
            yoffset = np.mean(ylim)
    
    if axis == 'x':
        horizontalalignment = 'center'
    
    if 'right' in location:
        if axis == 'x':
            xoffset = xlim[1]-np.diff(xlim)[0]*.1-realsize
        else:
            xoffset = xlim[1]-np.diff(xlim)[0]*.1
            horizontalalignment = 'right'
    elif 'left' in location:
        xoffset = xlim[0]+np.diff(xlim)[0]*.1
        if axis == 'y':
            horizontalalignment = 'left'
    else:
        if axis == 'x':
            xoffset = np.mean(xlim) - realsize/2
        else:
            xoffset = np.mean(xlim)
            
    if axis == 'x':    
        xvalues = scalevalues + xoffset
        yvalues = np.asarray([0,0]) + yoffset
    else:
        xvalues = np.asarray([0,0])+xoffset
        yvalues = scalevalues + yoffset
        
    ax.plot(xvalues,yvalues,linewidth = linewidth, color = color) 
    ax.text(xoffset_text,yoffset_text, '{} {}'.format(size,unit),color=color,horizontalalignment=horizontalalignment)
    print([xvalues,yvalues])

def collect_ca_wave_parameters(ca_wave_parameters,cawave_properties_needed,ephys_properties_needed):
    parameters_must_be_downloaded = ['subject_id','session','cell_number','sweep_number','ap_group_number']
    cawave_properties_all = cawave_properties_needed.copy()
    cawave_properties_all.extend(parameters_must_be_downloaded)
    cawave_properties_all.extend(ephys_properties_needed)
    key= {'neuropil_number': ca_wave_parameters['neuropil_number'],
          'neuropil_subtraction': ca_wave_parameters['neuropil_subtraction'],#'none','0.7','calculated'
          'session_calcium_sensor':None,#'GCaMP7F',#'GCaMP7F','XCaMPgf','456','686','688'
          'gaussian_filter_sigma' : ca_wave_parameters['gaussian_filter_sigma'],
          'channel_number':1,
          }
    snr_cond = 'ap_group_min_snr_dv>{}'.format(ca_wave_parameters['min_ap_snr'])
    preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
    postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
    basline_dff_cond = 'cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
    basline_dff_cond_2 = 'cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
    ephys_snr_cond = 'sweep_ap_snr_dv_median>={}'.format(ca_wave_parameters['min_sweep_ap_snr_dv_median'])
    ephys_hwstd_cond = 'sweep_ap_hw_std_per_median<={}'.format(ca_wave_parameters['max_sweep_ap_hw_std_per_median'])
    recording_mode_cond = 'recording_mode = "{}"'.format(ca_wave_parameters['ephys_recording_mode'])
    baseline_f_cond = 'cawave_baseline_f > {}'.format(ca_wave_parameters['minimum_f0_value'])
    channel_offset_cond = 'channel_offset > {}'.format(ca_wave_parameters['min_channel_offset'])
    #cawaves_all = imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()
    ca_waves_dict = dict()
    for sensor in ca_wave_parameters['sensors']:
        print(sensor)
        key['session_calcium_sensor'] = sensor
        cawaves_all = imaging.MovieChannel()*imaging_gt.SessionROIFPercentile()*ephysanal_cell_attached.SweepAPQC()*ephysanal_cell_attached.CellSpikeParameters()*imaging_gt.ROIDwellTime()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging.MovieMetaData()*imaging.MovieNote()*imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.CalciumWave()*imaging_gt.CalciumWave.CalciumWaveNeuropil()&'movie_note_type = "quality"'
        cellwave_by_cell_list = list()
        for cellkey in ephys_cell_attached.Cell():
# =============================================================================
#             cell_ephys_data = dict()
#             ephys_properties_list = (ephysanal_cell_attached.CellSpikeParameters()&cellkey).fetch(*ephys_properties_needed)
#             for data,data_name in zip(ephys_properties_list,ephys_properties_needed):
#                 cell_ephys_data[data_name] = data 
# =============================================================================
            cawaves = cawaves_all& key & snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 &ephys_snr_cond & ephys_hwstd_cond & cellkey  &recording_mode_cond&baseline_f_cond & channel_offset_cond
            if ca_wave_parameters['only_manually_labelled_movies']:
                cawaves = cawaves&'movie_note = "1"'
            if len(cawaves) == 0:
                continue
            #print(cellkey)
            cell_data = dict()
            #cell_data['cell_ephys_data']=cell_ephys_data
            cawave_properties_list = cawaves.fetch(*cawave_properties_all)
            for data,data_name in zip(cawave_properties_list,cawave_properties_all):
                cell_data[data_name] = data 
            if np.nanmean(cell_data['ap_peak_to_trough_time_median'])<ca_wave_parameters['max_ap_peak_to_trough_time_median'] and np.nanmean(cell_data['ap_ahp_amplitude_median'])>ca_wave_parameters['min_ap_ahp_amplitude_median']:
                continue # if interneurons are skipped
            if np.nanmean(cell_data['ap_peak_to_trough_time_median'])>ca_wave_parameters['min_ap_peak_to_trough_time_median'] or np.nanmean(cell_data['ap_ahp_amplitude_median'])<ca_wave_parameters['max_ap_ahp_amplitude_median']:
                continue # if pyramidal cells are skipped
            cell_data['movie_hmotors_sample_z'] = np.asarray(cell_data['movie_hmotors_sample_z'],float)
            cell_data['movie_power'] = np.asarray(cell_data['movie_power'],float)
            cell_data['movie_hmotors_sample_z'][cell_data['movie_hmotors_sample_z']<0] = cellkey['depth']
            ap_num_list = np.unique(cell_data['ap_group_ap_num'])
            for property_name in cawave_properties_needed:
                cell_data['{}_mean_per_ap'.format(property_name)]=list()
                cell_data['{}_median_per_ap'.format(property_name)]=list()
                cell_data['{}_std_per_ap'.format(property_name)]=list()
                cell_data['{}_values_per_ap'.format(property_name)]=list()
            cell_data['event_num_per_ap'] = list()
            for ap_num_now in ap_num_list:
                idx = cell_data['ap_group_ap_num']==ap_num_now
                cell_data['event_num_per_ap'].append(sum(idx))
                for property_name in cawave_properties_needed:
                    cell_data['{}_mean_per_ap'.format(property_name)].append(np.nanmean(cell_data[property_name][idx]))
                    cell_data['{}_median_per_ap'.format(property_name)].append(np.nanmedian(cell_data[property_name][idx]))
                    cell_data['{}_std_per_ap'.format(property_name)].append(np.nanstd(cell_data[property_name][idx]))
                    cell_data['{}_values_per_ap'.format(property_name)].append(cell_data[property_name][idx])
    
            for property_name in cell_data.keys():
                if 'values' not in property_name:
                    cell_data[property_name] = np.asarray(cell_data[property_name])        
            #moviedepth = imaging.MovieMetaData().fetch('movie_hmotors_sample_z')        
            cell_data['cell_key'] = cellkey
            cellwave_by_cell_list.append(cell_data.copy())        
        ca_waves_dict[sensor] = cellwave_by_cell_list
    return(ca_waves_dict)
    
def collect_ca_wave_traces( ca_wave_parameters,ca_waves_dict,parameters,use_doublets = False   ):
    ca_traces_dict = dict()
    ca_traces_dict_by_cell = dict()
    for sensor in ca_wave_parameters['sensors']:
        ca_traces_dict[sensor] = {
                'cawave_rise_times_list':list(),
                'cawave_peak_amplitudes_dff_list':list(),
                'cawave_f_corr_list':list(),
                'cawave_dff_list':list(),
                'cawave_time_list':list(),
                'cawave_lengths':list(),
                'cawave_f0_list':list(),
                'cawave_ap_group_length_list':list(),
                'cawave_snr_list':list(),
                'cawave_f_corr_normalized_list':list()}

        key= {'neuropil_number': ca_wave_parameters['neuropil_number'],
              'neuropil_subtraction': ca_wave_parameters['neuropil_subtraction'],#'none','0.7','calculated'
              'session_calcium_sensor':sensor,#'GCaMP7F',#'GCaMP7F','XCaMPgf','456','686','688'
              'gaussian_filter_sigma' : ca_wave_parameters['gaussian_filter_sigma'],
              'channel_number':1,
              }
        snr_cond = 'ap_group_min_snr_dv>{}'.format(ca_wave_parameters['min_ap_snr'])
        preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
        postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
        if use_doublets:
            basline_dff_cond = 'doublet_cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
            basline_dff_cond_2 = 'doublet_cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
            baseline_f_cond = 'doublet_cawave_baseline_f > {}'.format(ca_wave_parameters['minimum_f0_value'])
            snr_cond = 'ap_doublet_min_snr_dv>{}'.format(ca_wave_parameters['min_ap_snr'])
            preisi_cond = 'ap_doublet_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
            postisi_cond = 'ap_doublet_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
        else:
            basline_dff_cond = 'cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
            basline_dff_cond_2 = 'cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
            baseline_f_cond = 'cawave_baseline_f > {}'.format(ca_wave_parameters['minimum_f0_value'])
            snr_cond = 'ap_group_min_snr_dv>{}'.format(ca_wave_parameters['min_ap_snr'])
            preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
            postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
        ephys_snr_cond = 'sweep_ap_snr_dv_median>={}'.format(ca_wave_parameters['min_sweep_ap_snr_dv_median'])
        ephys_hwstd_cond = 'sweep_ap_hw_std_per_median<={}'.format(ca_wave_parameters['max_sweep_ap_hw_std_per_median'])
        
        channel_offset_cond = 'channel_offset > {}'.format(ca_wave_parameters['min_channel_offset'])
        
        if use_doublets:
            cawaves_all = imaging.MovieChannel()*ephysanal_cell_attached.SweepAPQC()*imaging_gt.ROIDwellTime()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging.MovieMetaData()*imaging.MovieNote()*imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APDoublet()*imaging_gt.DoubletCalciumWave.DoubletCalciumWaveProperties()*imaging_gt.DoubletCalciumWave()*imaging_gt.DoubletCalciumWave.DoubletCalciumWaveNeuropil()&'movie_note_type = "quality"'
        else:
            cawaves_all = imaging.MovieChannel()*ephysanal_cell_attached.SweepAPQC()*imaging_gt.ROIDwellTime()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging.MovieMetaData()*imaging.MovieNote()*imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.CalciumWave()*imaging_gt.CalciumWave.CalciumWaveNeuropil()&'movie_note_type = "quality"'
    
        cellwave_by_cell_list = np.asarray(ca_waves_dict[sensor])
        cellvals = list()
        for cell_data in cellwave_by_cell_list: # set order
            apidx = cell_data['ap_group_ap_num_mean_per_ap']==parameters['ap_num_to_plot']
            if cell_data['event_num_per_ap'][apidx]>=parameters['min_ap_per_apnum']:
                cellvals.append(cell_data['cawave_snr_mean_per_ap'][apidx][0])#cawave_snr_mean_per_ap#cawave_peak_amplitude_dff_mean_per_ap
            else:
                cellvals.append(np.nan)
        order = np.argsort(cellvals)
        cellwave_by_cell_list = cellwave_by_cell_list[order]
        cellvals = np.asarray(cellvals)[order]
        ca_traces_by_cell_list = list()
        for cell_data,cellval in zip(cellwave_by_cell_list,cellvals): # plot
            if np.isnan(cellval):#
                continue
            if 'ignore_cells_below_median_snr' in parameters.keys():
                if parameters['ignore_cells_below_median_snr'] and cellval<np.nanmedian(cellvals): # restriction for only SNR above median
                    continue
            cawaves = cawaves_all& key & snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 & ephys_snr_cond & ephys_hwstd_cond & cell_data['cell_key']  &baseline_f_cond & channel_offset_cond
            if ca_wave_parameters['only_manually_labelled_movies']:
                cawaves = cawaves&'movie_note = "1"'
            if len(cawaves) == 0:
                continue
            cawave_rise_times_list = list()
            cawave_peak_amplitudes_dff_list = list()
            cawave_f_corr_list = list()
            cawave_dff_list  = list()
            cawave_time_list = list()
            cawave_lengths = list()
            cawave_f0_list = list()
            cawave_ap_group_length_list = list()
            cawave_snr_list = list()
            cawave_f_corr_normalized_list = list()
            if use_doublets:
                cawave_fs,cawave_times,cawave_fneus,cawave_f0s,cawave_rneus,ap_group_start_time,ap_group_length,cawave_snrs,cawave_peak_amplitude_fs,cawave_peak_amplitude_dffs,cawave_rise_times = cawaves.fetch('doublet_cawave_f','doublet_cawave_time','doublet_cawave_fneu','doublet_cawave_baseline_f','doublet_cawave_rneu','ap_doublet_start_time','ap_doublet_length','doublet_cawave_snr','doublet_cawave_peak_amplitude_f','doublet_cawave_peak_amplitude_dff','doublet_cawave_rise_time')
                ap_group_end_time = np.asarray(ap_group_start_time,float)+np.asarray(ap_group_length,float)
            else:
                cawave_fs,cawave_times,cawave_fneus,cawave_f0s,cawave_rneus,ap_group_start_time,ap_group_end_time,cawave_snrs,cawave_peak_amplitude_fs,cawave_peak_amplitude_dffs,cawave_rise_times = (cawaves&'ap_group_ap_num={}'.format(parameters['ap_num_to_plot'])).fetch('cawave_f','cawave_time','cawave_fneu','cawave_baseline_f','cawave_rneu','ap_group_start_time','ap_group_end_time','cawave_snr','cawave_peak_amplitude_f','cawave_peak_amplitude_dff','cawave_rise_time')
            ap_group_lengths = np.asarray(ap_group_end_time,float)-np.asarray(ap_group_start_time,float)
            for cawave_f,cawave_time,cawave_fneu,cawave_f0,cawave_rneu,ap_group_length,cawave_snr,cawave_peak_amplitude_f, cawave_peak_amplitude_dff,cawave_rise_time in zip(cawave_fs,cawave_times,cawave_fneus,cawave_f0s,cawave_rneus,ap_group_lengths,cawave_snrs,cawave_peak_amplitude_fs,cawave_peak_amplitude_dffs ,cawave_rise_times):
                cawave_f_corr = cawave_f-cawave_fneu*cawave_rneu-cawave_f0
                cawave_dff = (cawave_f_corr)/cawave_f0
                #zeroidx = np.argmin(np.abs(cawave_time))
                cawave_peak_amplitudes_dff_list.append(cawave_peak_amplitude_dff)
                cawave_lengths.append(len(cawave_dff))
                cawave_f_corr_list.append(np.asarray(cawave_f_corr,float))
                cawave_dff_list.append(np.asarray(cawave_dff,float))
                cawave_time_list.append(np.asarray(cawave_time,float))
                cawave_f0_list.append(cawave_f0)
                cawave_ap_group_length_list.append(ap_group_length)
                cawave_snr_list.append(cawave_snr)
                cawave_f_corr_normalized_list.append(np.asarray(cawave_f_corr,float)/cawave_peak_amplitude_f)
                cawave_rise_times_list.append(cawave_rise_time)
            if len(np.unique(cawave_lengths))>1:
                print('not equal traces')
                #time.sleep(1000)
            cell_dict = {'cawave_rise_times_list':cawave_rise_times_list,
                         'cawave_peak_amplitudes_dff_list':cawave_peak_amplitudes_dff_list,
                         'cawave_f_corr_list':cawave_f_corr_list,
                         'cawave_dff_list':cawave_dff_list,
                         'cawave_time_list':cawave_time_list,
                         'cawave_lengths':cawave_lengths,
                         'cell_key':cell_data['cell_key'],
                         'cawave_f0_list':cawave_f0_list,
                         'cawave_ap_group_length_list':cawave_ap_group_length_list,
                         'cawave_snr_list':cawave_snr_list,
                         'cawave_f_corr_normalized_list':cawave_f_corr_normalized_list}
            ca_traces_by_cell_list.append(cell_dict)
            ca_traces_dict[sensor]['cawave_rise_times_list'].extend(cawave_rise_times_list)
            ca_traces_dict[sensor]['cawave_peak_amplitudes_dff_list'].extend(cawave_peak_amplitudes_dff_list)
            ca_traces_dict[sensor]['cawave_f_corr_list'].extend(cawave_f_corr_list)
            ca_traces_dict[sensor]['cawave_dff_list'].extend(cawave_dff_list)
            ca_traces_dict[sensor]['cawave_time_list'].extend(cawave_time_list)
            ca_traces_dict[sensor]['cawave_lengths'].extend(cawave_lengths)
            ca_traces_dict[sensor]['cawave_f0_list'].extend(cawave_f0_list)
            ca_traces_dict[sensor]['cawave_ap_group_length_list'].extend(cawave_ap_group_length_list)
            ca_traces_dict[sensor]['cawave_snr_list'].extend(cawave_snr_list)
            ca_traces_dict[sensor]['cawave_f_corr_normalized_list'].extend(cawave_f_corr_normalized_list)
        ca_traces_dict_by_cell[sensor] = ca_traces_by_cell_list
    return ca_traces_dict,ca_traces_dict_by_cell


def calculate_superresolution_traces_for_all_sensors(ca_traces_dict,ca_traces_dict_by_cell,ca_wave_parameters,plot_parameters):
    superresolution_traces = dict()
    for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
        superresolution_traces[sensor] = dict()
        print(sensor)
        for tracename in plot_parameters['traces_to_use']:
            x_conc = np.concatenate(ca_traces_dict[sensor]['cawave_time_list'])
            y_conc= np.concatenate(ca_traces_dict[sensor]['cawave_{}_list'.format(tracename)])
            needed_idx = (x_conc>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc<=plot_parameters['end_time']+plot_parameters['bin_size'])
            x_conc = x_conc[needed_idx]
            y_conc = y_conc[needed_idx]
            if plot_parameters['double_bin_size_for_old_sensors'] and ('gcamp7f' in sensor or 'xcamp' in sensor):
                trace_dict = generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size']*3,function = 'all', dogaussian = plot_parameters['dogaussian']) 
            else:
                trace_dict = generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'all', dogaussian = plot_parameters['dogaussian']) 
            x = np.asarray(trace_dict['bin_centers'])
            y_mean = np.asarray(trace_dict['trace_mean'])
            y_median = np.asarray(trace_dict['trace_median'])
            y_gauss = np.asarray(trace_dict['trace_gauss'])
            y_sd = np.asarray(trace_dict['trace_std'])
            y_n = np.asarray(trace_dict['trace_n'])
            if not plot_parameters['normalize_raw_trace_offset'] and tracename == 'f':
                y_mean = y_mean+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
                y_median = y_median+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
                y_gauss = y_gauss+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
            superresolution_traces[sensor]['{}_time'.format(tracename)] = x
            superresolution_traces[sensor]['{}_mean'.format(tracename)] = y_mean
            superresolution_traces[sensor]['{}_median'.format(tracename)] = y_median
            superresolution_traces[sensor]['{}_gauss'.format(tracename)] = y_gauss
            superresolution_traces[sensor]['{}_sd'.format(tracename)] = y_sd
            superresolution_traces[sensor]['{}_n'.format(tracename)] = y_n
            superresolution_traces[sensor]['{}_raw_time'.format(tracename)] = x_conc
            superresolution_traces[sensor]['{}_raw'.format(tracename)] = y_conc
            
            superresolution_traces[sensor]['{}_time_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_mean_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_median_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_gauss_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_sd_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_n_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_raw_time_per_cell'.format(tracename)] = list()
            superresolution_traces[sensor]['{}_raw_per_cell'.format(tracename)] = list()
        if   plot_parameters['cell_wise_analysis']:
            for cell in ca_traces_dict_by_cell[sensor]:
                if len(cell['cawave_time_list'])>=plot_parameters['min_ap_per_apnum']:
                    
                    for tracename in plot_parameters['traces_to_use']:
                        x_conc = np.concatenate(cell['cawave_time_list'])
                        y_conc= np.concatenate(cell['cawave_{}_list'.format(tracename)])
                        needed_idx = (x_conc>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc<=plot_parameters['end_time']+plot_parameters['bin_size'])
                        x_conc = x_conc[needed_idx]
                        y_conc = y_conc[needed_idx]
                        if plot_parameters['double_bin_size_for_old_sensors'] and ('gcamp7f' in sensor or 'xcamp' in sensor):
                            trace_dict = generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size']*3,function = 'all', dogaussian = plot_parameters['dogaussian']) 
                        else:
                            trace_dict = generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'all', dogaussian = plot_parameters['dogaussian']) 
                        x = np.asarray(trace_dict['bin_centers'])
                        y_mean = np.asarray(trace_dict['trace_mean'])
                        y_median = np.asarray(trace_dict['trace_median'])
                        y_gauss = np.asarray(trace_dict['trace_gauss'])
                        y_sd = np.asarray(trace_dict['trace_std'])
                        y_n = np.asarray(trace_dict['trace_n'])
                        if not plot_parameters['normalize_raw_trace_offset'] and tracename == 'f':
                            y_mean = y_mean+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
                            y_median = y_median+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])              
                        superresolution_traces[sensor]['{}_time_per_cell'.format(tracename)].append(x)
                        superresolution_traces[sensor]['{}_mean_per_cell'.format(tracename)].append(y_mean)
                        superresolution_traces[sensor]['{}_median_per_cell'.format(tracename)].append(y_median)
                        superresolution_traces[sensor]['{}_gauss_per_cell'.format(tracename)].append(y_gauss)
                        superresolution_traces[sensor]['{}_sd_per_cell'.format(tracename)].append(y_sd)
                        superresolution_traces[sensor]['{}_n_per_cell'.format(tracename)].append(y_n)
                        superresolution_traces[sensor]['{}_raw_time_per_cell'.format(tracename)].append(x_conc)
                        superresolution_traces[sensor]['{}_raw_per_cell'.format(tracename)].append(y_conc)
    return superresolution_traces