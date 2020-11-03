import numpy as np
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt

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
    

def generate_superresolution_trace(x_conc,y_conc,bin_step,bin_size,function = 'mean'):
    bin_centers = np.arange(np.min(x_conc),np.max(x_conc),bin_step)
    bin_mean = list()
    bin_median = list()
    for bin_center in bin_centers:
        bin_mean.append(np.mean(y_conc[(x_conc>bin_center-bin_size/2) & (x_conc<bin_center+bin_size/2)]))
        bin_median.append(np.median(y_conc[(x_conc>bin_center-bin_size/2) & (x_conc<bin_center+bin_size/2)]))
    bin_mean = bin_median
    bin_mean = remove_nans_from_trace(bin_mean)
    bin_median = remove_nans_from_trace(bin_median)
    if function== 'median':
        trace = bin_median
    else:
        trace = bin_mean
    return bin_centers,trace

def collect_ca_wave_parameters(ca_wave_parameters,cawave_properties_needed):
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
    #cawaves_all = imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()
    ca_waves_dict = dict()
    for sensor in ca_wave_parameters['sensors']:
        print(sensor)
        key['session_calcium_sensor'] = sensor
        cawaves_all = imaging_gt.ROIDwellTime()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging.MovieMetaData()*imaging.MovieNote()*imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.CalciumWave()*imaging_gt.CalciumWave.CalciumWaveNeuropil()&'movie_note_type = "quality"'
        cellwave_by_cell_list = list()
        for cellkey in ephys_cell_attached.Cell():
            
            cawaves = cawaves_all& key & snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 & cellkey  
            if ca_wave_parameters['only_manually_labelled_movies']:
                cawaves = cawaves&'movie_note = "1"'
            if len(cawaves) == 0:
                continue
            #print(cellkey)
            cell_data = dict()
            cawave_properties_list = cawaves.fetch(*cawave_properties_needed)
            for data,data_name in zip(cawave_properties_list,cawave_properties_needed):
                cell_data[data_name] = data 
            cell_data['movie_hmotors_sample_z'] = np.asarray(cell_data['movie_hmotors_sample_z'],float)
            cell_data['movie_power'] = np.asarray(cell_data['movie_power'],float)
            cell_data['movie_hmotors_sample_z'][cell_data['movie_hmotors_sample_z']<0] = cellkey['depth']
            ap_num_list = np.unique(cell_data['ap_group_ap_num'])
            for property_name in cawave_properties_needed:
                cell_data['{}_mean_per_ap'.format(property_name)]=list()
                cell_data['{}_std_per_ap'.format(property_name)]=list()
                cell_data['{}_values_per_ap'.format(property_name)]=list()
            cell_data['event_num_per_ap'] = list()
            for ap_num_now in ap_num_list:
                idx = cell_data['ap_group_ap_num']==ap_num_now
                cell_data['event_num_per_ap'].append(sum(idx))
                for property_name in cawave_properties_needed:
                    cell_data['{}_mean_per_ap'.format(property_name)].append(np.nanmean(cell_data[property_name][idx]))
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
    
def collect_ca_wave_traces( ca_wave_parameters,ca_waves_dict,parameters   ):
    ca_traces_dict = dict()
    ca_traces_dict_by_cell = dict()
    for sensor in ca_wave_parameters['sensors']:
        ca_traces_dict[sensor] = {
                'cawave_f_corr_list':list(),
                'cawave_dff_list':list(),
                'cawave_time_list':list(),
                'cawave_lengths':list(),
                'cawave_f0_list':list()}

        key= {'neuropil_number': ca_wave_parameters['neuropil_number'],
              'neuropil_subtraction': ca_wave_parameters['neuropil_subtraction'],#'none','0.7','calculated'
              'session_calcium_sensor':sensor,#'GCaMP7F',#'GCaMP7F','XCaMPgf','456','686','688'
              'gaussian_filter_sigma' : ca_wave_parameters['gaussian_filter_sigma'],
              'channel_number':1,
              }
        snr_cond = 'ap_group_min_snr_dv>{}'.format(ca_wave_parameters['min_ap_snr'])
        preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
        postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
        basline_dff_cond = 'cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
        basline_dff_cond_2 = 'cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
        cawaves_all = imaging_gt.ROIDwellTime()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging.MovieMetaData()*imaging.MovieNote()*imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.CalciumWave()*imaging_gt.CalciumWave.CalciumWaveNeuropil()&'movie_note_type = "quality"'
    
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
            if np.isnan(cellval):#cellval<np.nanmedian(cellvals): # restriction for only SNR above median
                continue
            cawaves = cawaves_all& key & snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 & cell_data['cell_key']  
            if ca_wave_parameters['only_manually_labelled_movies']:
                cawaves = cawaves&'movie_note = "1"'
            if len(cawaves) == 0:
                continue
            cawave_f_corr_list = list()
            cawave_dff_list  = list()
            cawave_time_list = list()
            cawave_lengths = list()
            cawave_f0_list = list()
            cawave_fs,cawave_times,cawave_fneus,cawave_f0s,cawave_rneus = (cawaves&'ap_group_ap_num={}'.format(parameters['ap_num_to_plot'])).fetch('cawave_f','cawave_time','cawave_fneu','cawave_baseline_f','cawave_rneu')
            for cawave_f,cawave_time,cawave_fneu,cawave_f0,cawave_rneu in zip(cawave_fs,cawave_times,cawave_fneus,cawave_f0s,cawave_rneus ):
                cawave_f_corr = cawave_f-cawave_fneu*cawave_rneu-cawave_f0
                cawave_dff = (cawave_f_corr)/cawave_f0
                #zeroidx = np.argmin(np.abs(cawave_time))
                cawave_lengths.append(len(cawave_dff))
                cawave_f_corr_list.append(np.asarray(cawave_f_corr,float))
                cawave_dff_list.append(np.asarray(cawave_dff,float))
                cawave_time_list.append(np.asarray(cawave_time,float))
                cawave_f0_list.append(cawave_f0)
            if len(np.unique(cawave_lengths))>1:
                print('not equal traces')
                #time.sleep(1000)
            cell_dict = {'cawave_f_corr_list':cawave_f_corr_list,
                         'cawave_dff_list':cawave_dff_list,
                         'cawave_time_list':cawave_time_list,
                         'cawave_lengths':cawave_lengths,
                         'cell_key':cell_data['cell_key'],
                         'cawave_f0_list':cawave_f0_list}
            ca_traces_by_cell_list.append(cell_dict)
            ca_traces_dict[sensor]['cawave_f_corr_list'].extend(cawave_f_corr_list)
            ca_traces_dict[sensor]['cawave_dff_list'].extend(cawave_dff_list)
            ca_traces_dict[sensor]['cawave_time_list'].extend(cawave_time_list)
            ca_traces_dict[sensor]['cawave_lengths'].extend(cawave_lengths)
            ca_traces_dict[sensor]['cawave_f0_list'].extend(cawave_f0_list)
        ca_traces_dict_by_cell[sensor] = ca_traces_by_cell_list
    return ca_traces_dict,ca_traces_dict_by_cell