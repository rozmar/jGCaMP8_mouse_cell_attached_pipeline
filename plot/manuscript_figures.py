import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
from plot import manuscript_figures_core , cell_attached_ephys_plot, imaging_gt_plot
from utils import utils_plot,utils_ephys
import pandas as pd
import seaborn as sns
import matplotlib
sensor_colors ={'GCaMP7F':'green',
                'XCaMPgf':'orange',
                '456':'blue',
                '686':'red',
                '688':'black'}
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}
figures_dir = '/home/rozmar/Data/shared_dir/GCaMP8/'
matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
%matplotlib qt
# =============================================================================
# #%%  gaussian
# #figures_dir = '/home/rozmar/Data/shared_dir/'
# import scipy.stats as stats
# fig = plt.figure()
# ax = fig.add_subplot(111)
# gauss_x = np.arange(-3,3,.1)
# gauss_f = stats.norm()
# gauss_y = gauss_f.pdf(gauss_x)
# #%
# ax.plot(gauss_x,gauss_y,'k-')
# figfile = os.path.join(figures_dir,'gauss.{}')
# inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
# subprocess.run(inkscape_cmd )
# =============================================================================

#%%#######################################################################Grand averages-scatter plots - START################################################################################
#################################################################################%%############################################################################################
#%% Select events for putative pyramidal cells 
leave_out_putative_interneurons = True
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':.5,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':1,
                      'min_post_isi':.5,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':5,#0,5,10,20,50
                      'neuropil_subtraction':'0.8',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['XCaMPgf','GCaMP7F','456','686','688'], # this determines the order of the sensors,#'XCaMPgf'
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100 # PMT error
                      }
if leave_out_putative_interneurons:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['min_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=-100
    ca_wave_parameters['min_ap_ahp_amplitude_median']=100
if leave_out_putative_pyramidal_cells:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['max_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=100
    ca_wave_parameters['max_ap_ahp_amplitude_median']=-100    
cawave_properties_needed = ['cawave_baseline_dff_global',
                            'ap_group_ap_num',
                            'cawave_snr', 
                            'cawave_peak_amplitude_dff',
                            'cawave_peak_amplitude_f',
                            'cawave_baseline_f',
                            'cawave_baseline_f_std',
                            'cawave_rneu',
                            'cawave_rise_time',
                            'cawave_half_decay_time',
                            'movie_hmotors_sample_z',
                            'movie_power',
                            'session_sensor_expression_time',
                            'roi_pixel_num',
                            'roi_dwell_time_multiplier',
                            'roi_f0_percentile',
                            'roi_f0_percentile_value']
ephys_properties_needed = ['ap_ahp_amplitude_median',
                           'ap_halfwidth_median',
                           'ap_full_width_median',
                           'ap_rise_time_median',
                           'ap_peak_to_trough_time_median',
                           'recording_mode']
#%
ca_waves_dict_pyr = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)



#%% download sensor kinetics
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':1} # filters both grand average and cell-wise traces
ca_traces_dict_1ap_pyr,ca_traces_dict_by_cell_1ap_pyr  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,
                                                                                           ca_waves_dict_pyr,
                                                                                           wave_parameters)

#%
#%%
from utils import utils_plot
superres_parameters={'sensor_colors':sensor_colors,
                     'normalize_raw_trace_offset':True,#set to 0
                     'min_ap_per_apnum':10, # filters only cell-wise trace
                     'bin_step_for_cells':.5,
                     'bin_size_for_cells':8,
                     'bin_step':.5,
                     'bin_size':8, # 8 means 2ms sigma for gaussian
                     'start_time':-50, # ms
                     'end_time':500,#ms
                     'traces_to_use':['f_corr','dff','f_corr_normalized'],
                     'cell_wise_analysis':True,
                     'dogaussian':True,
                     'double_bin_size_for_old_sensors':True
                     }
superresolution_traces_1ap_pyr_low_resolution = utils_plot.calculate_superresolution_traces_for_all_sensors(ca_traces_dict_1ap_pyr,
                                                                                                             ca_traces_dict_by_cell_1ap_pyr,
                                                                                                             ca_wave_parameters,
                                                                                                             superres_parameters)
#%
from utils import utils_plot
superres_parameters={'sensor_colors':sensor_colors,
                     'normalize_raw_trace_offset':True,#set to 0
                     'min_ap_per_apnum':10, # filters only cell-wise trace
                     'bin_step_for_cells':1,
                     'bin_size_for_cells':8,
                     'bin_step':.5,
                     'bin_size':2, #2 means 0.5 sigma for gaussian
                     'start_time':-50, # ms
                     'end_time':500,#ms
                     'traces_to_use':['f_corr','dff','f_corr_normalized'],
                     'cell_wise_analysis':False,
                     'dogaussian':True,
                     'double_bin_size_for_old_sensors':True
                     }
superresolution_traces_1ap_pyr_high_resolution = utils_plot.calculate_superresolution_traces_for_all_sensors(ca_traces_dict_1ap_pyr,
                                                                                                             ca_traces_dict_by_cell_1ap_pyr,
                                                                                                             ca_wave_parameters,
                                                                                                             superres_parameters)
#%% grand averages
from plot import manuscript_figures_core
plot_parameters={'sensor_colors':sensor_colors,
                 'sensors':ca_wave_parameters['sensors'],
                 'start_time':-50,#-50, # ms
                 'end_time':500,#100,#ms
                 'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'gauss',#'mean',#'median'#'gauss'#
                 'normalize_to_max':False,
                 'linewidth':3,
                 'error_bar':'none',#se,sd
                 'figures_dir':figures_dir,
                 'fig_name':'fig_grand_average_traces_pyr',
                 'xscalesize':100,
                 'yscalesize':10,
                 }
manuscript_figures_core.plot_superresolution_grand_averages(superresolution_traces_1ap_pyr_low_resolution,plot_parameters)
plot_parameters={'sensor_colors':sensor_colors,
                 'sensors':ca_wave_parameters['sensors'],
                 'start_time':-10,#-50, # ms
                 'end_time':100,#100,#ms
                 'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'gauss',#'mean',#'median'#'gauss'#
                 'normalize_to_max':False,
                 'linewidth':3,
                 'error_bar':'none',#se,sd
                 'figures_dir':figures_dir,
                 'fig_name':'fig_grand_average_traces_pyr_zoom',
                 'xscalesize':10,
                 'yscalesize':10,
                 }
manuscript_figures_core.plot_superresolution_grand_averages(superresolution_traces_1ap_pyr_high_resolution,plot_parameters)
#%% each cell separately
superresolution_traces = superresolution_traces_1ap_pyr_low_resolution
plot_parameters={'sensor_colors':sensor_colors,
                 'start_time':-50, # ms
                 'end_time':200,#ms
                 'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'gauss',#'mean',#'median'#
                 'normalize_to_max':False,
                 'linewidth':2,
                 'alpha':.5
                 }

sensornum = len(ca_wave_parameters['sensors'])
fig_cell_wise_trace = plt.figure(figsize = [5,15])
ax_dict = dict()
for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
    try:
        ax_dict[sensor] = fig_cell_wise_trace.add_subplot(sensornum,1,sensor_i+1,sharey = ax_dict[ca_wave_parameters['sensors'][0]],sharex = ax_dict[ca_wave_parameters['sensors'][0]])
    except:
        ax_dict[sensor] = fig_cell_wise_trace.add_subplot(sensornum,1,sensor_i+1)
    color = plot_parameters['sensor_colors'][sensor]
    xs = superresolution_traces[sensor]['{}_time_per_cell'.format(plot_parameters['trace_to_use'])]
    ys = superresolution_traces[sensor]['{}_{}_per_cell'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
    for x,y in zip(xs,ys):
        if plot_parameters['normalize_to_max']:
            y = y/np.max(y)
        ax_dict[sensor].plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = plot_parameters['alpha'])
    ax_dict[sensor].set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
    ax_dict[sensor].axis('off')
utils_plot.add_scalebar_to_fig(ax_dict[ca_wave_parameters['sensors'][0]],
                               axis = 'y',
                               size = 50,
                               conversion = 100,
                               unit = r'dF/F%',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5)

utils_plot.add_scalebar_to_fig(ax_dict[ca_wave_parameters['sensors'][0]],
                               axis = 'x',
                               size = 100,
                               conversion = 1,
                               unit = r'ms',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5)    
figfile = os.path.join(figures_dir,'fig_fig_cell_wise_traces_pyr.{}')
fig_cell_wise_trace.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )
#%% calculate rise times, half decay times and amplitudes from superresolution traces
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':10,
                   'trace_to_use':'dff',#'f','dff'
                   'function_to_use':'median',#'median'#'mean'
                   'superresolution_function':'mean',#'median',
                   'max_rise_time':100,#ms
                   }
plot_parameters={'sensor_colors':sensor_colors,
                 'partial_risetime_target':.8,
                 'figures_dir':figures_dir,
                 'plot_name':'fig_amplitude_rise_time_scatter_pyr_1ap'}
manuscript_figures_core.plot_cell_wise_scatter_plots(ca_wave_parameters,
                                                     ca_waves_dict_pyr,
                                                     superresolution_traces_1ap_pyr_low_resolution,
                                                     wave_parameters,
                                                     plot_parameters)


#%%#######################################################################Spike doublets - START################################################################################
#################################################################################%%############################################################################################
#%% download sensor kinetics for 2 AP - doublets
wave_parameters = {'ap_num_to_plot':2,#ap_num_to_plot
                   'min_ap_per_apnum':1} # filters both grand average and cell-wise traces
ca_wave_parameters['min_pre_isi']=.2
ca_wave_parameters['min_post_isi']=.1
ca_wave_parameters['max_baseline_dff']=3
ca_traces_dict_2ap_pyr,ca_traces_dict_by_cell_2ap_pyr  = utils_plot.collect_ca_wave_traces(ca_wave_parameters,
                                                                                           ca_waves_dict_pyr,
                                                                                           wave_parameters,
                                                                                           use_doublets = True)

#%% 2 AP kinetics

from plot import manuscript_figures_core
plot_parameters={'trace_to_use':'f_corr',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'gauss',#'mean',#'median'#'gauss'
                 'normalize_to_max':False,
                 'linewidth':3,
                 'time_range_step' : .005,
                 'time_range_window' : .001,
                 'time_range_offset' : .003,
                 'time_range_step_number' : 7,#9,
                 'bin_step' : 1,
                 'bin_size' : 8,
                 'sensor_colors':sensor_colors,
                 'start_time' : -25,
                 'end_time' : 100,
                 'min_event_number' : 10,#20,
                 'exclude_outliers' : True, # 5-95 percentiles included only by snr
                 'sensors_to_plot' : ['XCaMPgf','GCaMP7F','456','686','688'],#
                 'slow_sensors' : ['GCaMP7F'],#'XCaMPgf'
                 'figures_dir':figures_dir
                 }
manuscript_figures_core.plot_2ap_superresolution_traces(ca_traces_dict_2ap_pyr,superresolution_traces_1ap_pyr_low_resolution,plot_parameters)
#%% doublet linearity vs inter-spike interval
plot_parameters = {'sensors':['GCaMP7F','456','686','688'],
                   'step_size':0.5,
                   'bin_size':20}
fig = plt.figure()
ax_list = list()
ax_list_normalized = list()
for sensor_i, sensor in enumerate(plot_parameters['sensors']):
    if len(ax_list) == 0:
        ax_now = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+1)    
        ax_sum = fig.add_subplot(len(plot_parameters['sensors'])+1,3,len(plot_parameters['sensors'])*3+1,sharex = ax_now)    
        ax_now.set_title('AP doublet dF/F amplitude vs ISI')
    else:
        ax_now = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+1,sharex = ax_list[0])    
    ax_list.append(ax_now)
    isi_bulk = np.asarray(ca_traces_dict_2ap_pyr[sensor]['cawave_ap_group_length_list'])*1000
    dff_bulk = np.asarray(ca_traces_dict_2ap_pyr[sensor]['cawave_peak_amplitudes_dff_list'])
    ax_now.plot(isi_bulk,dff_bulk,'o', color =  sensor_colors[sensor], alpha = .3)
    superres_data_dict = utils_plot.generate_superresolution_trace(isi_bulk,dff_bulk,plot_parameters['step_size'],plot_parameters['bin_size'],function = 'all',dogaussian = False)
    ax_now.plot(superres_data_dict['bin_centers'],superres_data_dict['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_sum.plot(superres_data_dict['bin_centers'],superres_data_dict['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_now.set_ylabel('Event amplitude (dF/F)')
    isi_cellwise_normalized_list = list()
    dff_cellwise_normalized_list = list()
    dff_cellwise_normalized_decay_corrected_list = list()
    for cell_data_doublets in ca_traces_dict_by_cell_2ap_pyr[sensor]:
        dff_1ap = np.nan
        for cell_data_events in ca_waves_dict_pyr[sensor]:
            if cell_data_doublets['cell_key'] == cell_data_events['cell_key']:
                try:
                    dff_1ap = cell_data_events['cawave_peak_amplitude_dff_mean_per_ap'][cell_data_events['ap_group_ap_num_mean_per_ap']==1][0]
                    break
                except:
                    pass
        try:
            
            dff_cellwise_normalized_list.extend(cell_data_doublets['cawave_peak_amplitudes_dff_list']/dff_1ap)
            
            isi_cellwise_normalized_list.extend(cell_data_doublets['cawave_ap_group_length_list'])
            #%
            ap_decay_y = superresolution_traces_1ap_pyr_low_resolution[sensor]['dff_gauss']/np.max(superresolution_traces_1ap_pyr_low_resolution[sensor]['dff_gauss'])
            ap_decay_x = superresolution_traces_1ap_pyr_low_resolution[sensor]['dff_time']
            risetimes = cell_data_doublets['cawave_rise_times_list']
            amplitudes_corrected_normalized = list()
            for risetime, amplitude in zip(risetimes,cell_data_doublets['cawave_peak_amplitudes_dff_list']):
                idx = np.argmax(ap_decay_x>risetime)
                amplitude_corrected = amplitude+dff_1ap*(1-ap_decay_y[idx])
                amplitudes_corrected_normalized.append(amplitude_corrected/dff_1ap)
                #break
               #%
            dff_cellwise_normalized_decay_corrected_list.extend(amplitudes_corrected_normalized)
        except:
            pass
    if len(ax_list_normalized) == 0:
        ax_now_normalized = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+2,sharex = ax_list[0])
        ax_sum_normalized = fig.add_subplot(len(plot_parameters['sensors'])+1,3,len(plot_parameters['sensors'])*3+2,sharex = ax_list[0],sharey = ax_now_normalized)  
        ax_now_normalized.set_title('AP doublet normalized amplitude vs ISI')
        ax_now_normalized_decay_corrected = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+3,sharex = ax_list[0],sharey = ax_now_normalized)
        ax_sum_normalized_decay_corrected = fig.add_subplot(len(plot_parameters['sensors'])+1,3,len(plot_parameters['sensors'])*3+3,sharex = ax_list[0],sharey = ax_now_normalized)  
        ax_now_normalized_decay_corrected.set_title('AP doublet normalized decay corrected amplitude vs ISI')
    else:
        ax_now_normalized = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+2,sharex = ax_list[0],sharey = ax_list_normalized[0])
        ax_now_normalized_decay_corrected = fig.add_subplot(len(plot_parameters['sensors'])+1,3,sensor_i*3+3,sharex = ax_list[0],sharey = ax_now_normalized)
    ax_list_normalized.append(ax_now_normalized)
    isi_bulk_normalized = np.asarray(isi_cellwise_normalized_list)*1000
    dff_bulk_normalized = np.asarray(dff_cellwise_normalized_list)
    ax_now_normalized.plot(isi_bulk_normalized,dff_bulk_normalized,'o', color =  sensor_colors[sensor], alpha = .3)
    superres_data_dict_norm = utils_plot.generate_superresolution_trace(isi_bulk_normalized,dff_bulk_normalized,plot_parameters['step_size'],plot_parameters['bin_size'],function = 'all',dogaussian = False)
    ax_now_normalized.plot(superres_data_dict_norm['bin_centers'],superres_data_dict_norm['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_sum_normalized.plot(superres_data_dict_norm['bin_centers'],superres_data_dict_norm['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_now_normalized.set_ylabel('Normalized amplitude')
    
    
    isi_bulk_normalized_corrected = np.asarray(isi_cellwise_normalized_list)*1000
    dff_bulk_normalized_corrected = np.asarray(dff_cellwise_normalized_decay_corrected_list)
    ax_now_normalized_decay_corrected.plot(isi_bulk_normalized_corrected,dff_bulk_normalized_corrected,'o', color =  sensor_colors[sensor], alpha = .3)
    superres_data_dict_norm = utils_plot.generate_superresolution_trace(isi_bulk_normalized_corrected,dff_bulk_normalized_corrected,plot_parameters['step_size'],plot_parameters['bin_size'],function = 'all',dogaussian = False)
    ax_now_normalized_decay_corrected.plot(superres_data_dict_norm['bin_centers'],superres_data_dict_norm['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_sum_normalized_decay_corrected.plot(superres_data_dict_norm['bin_centers'],superres_data_dict_norm['trace_mean'], color =  sensor_colors[sensor], linewidth=4)
    ax_now_normalized_decay_corrected.set_ylabel('Normalized & corrected ampl')
    
ax_sum_normalized_decay_corrected.plot([0,100],[2,2],'k--')
ax_sum_normalized.plot([0,100],[2,2],'k--')
ax_sum.set_ylabel('Event amplitude (dF/F)')
ax_sum_normalized.set_ylabel('Normalized amplitude')
ax_sum_normalized_decay_corrected.set_ylabel('Normalized & corrected ampl')
ax_now_normalized.set_ylim([1,4.5])
ax_sum_normalized.set_xlabel('ISI (ms)')
ax_sum.set_xlabel('ISI (ms)')
ax_sum_normalized_decay_corrected.set_xlabel('ISI (ms)')
# =============================================================================
#         break
#     break
# =============================================================================
    
    #break
    



#%%#######################################################################Spike doublets - END################################################################################
#################################################################################%%############################################################################################

#%%#######################################################################Multiple AP pyr - START################################################################################
#################################################################################%%############################################################################################
#%% Select events for putative pyramidal cells - multiple AP
leave_out_putative_interneurons = True
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':3,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':.5,
                      'min_post_isi':.5,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':10,#0,5,10,20,50
                      'neuropil_subtraction':'0.8',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100,
                      }
if leave_out_putative_interneurons:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['min_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=-100
    ca_wave_parameters['min_ap_ahp_amplitude_median']=100
if leave_out_putative_pyramidal_cells:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['max_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=100
    ca_wave_parameters['max_ap_ahp_amplitude_median']=-100    
cawave_properties_needed = ['cawave_baseline_dff_global',
                            'ap_group_ap_num',
                            'cawave_snr', 
                            'cawave_peak_amplitude_dff',
                            'cawave_peak_amplitude_f',
                            'cawave_baseline_f',
                            'cawave_baseline_f_std',
                            'cawave_rneu',
                            'cawave_rise_time',
                            'movie_hmotors_sample_z',
                            'movie_power',
                            'session_sensor_expression_time',
                            'roi_pixel_num',
                            'roi_dwell_time_multiplier',
                            'roi_f0_percentile',
                            'roi_f0_percentile_value']
ephys_properties_needed = ['ap_ahp_amplitude_median',
                           'ap_halfwidth_median',
                           'ap_full_width_median',
                           'ap_rise_time_median',
                           'ap_peak_to_trough_time_median',
                           'recording_mode']
#%
ca_waves_dict_pyr_multiple_AP = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)
#%%
plot_parameters = {'min_ap_per_apnum': 3,
                   'min_cell_num':1,
                   'min_event_num':10, # for bulk plot
                   'max_apnum':5,
                   'sensor_colors':sensor_colors}
fig_dff_ap_dependence = plt.figure(figsize = [10,10])
ax_amplitude = fig_dff_ap_dependence.add_subplot(3,2,1)
ax_amplitude_bulk = fig_dff_ap_dependence.add_subplot(3,2,2,sharex = ax_amplitude,sharey = ax_amplitude)
#ax_amplitude_swarm = ax_amplitude.twinx()
ax_amplitude_norm = fig_dff_ap_dependence.add_subplot(3,2,3,sharex = ax_amplitude)
ax_amplitude_norm_bulk = fig_dff_ap_dependence.add_subplot(3,2,4,sharex = ax_amplitude,sharey = ax_amplitude_norm)
ax_cellnum = fig_dff_ap_dependence.add_subplot(3,2,5,sharex = ax_amplitude)
ax_eventnum = fig_dff_ap_dependence.add_subplot(3,2,6,sharex = ax_amplitude)
#ax_amplitude_norm_swarm = ax_amplitude_norm.twinx()
for sensor in ca_wave_parameters['sensors']:
    cellwave_by_cell_list = ca_waves_dict_pyr_multiple_AP[sensor]
    sensor_apnums = np.arange(1,100,1)
    sensor_y_vals = list()
    sensor_y_vals_norm = list()
    sensor_y_vals_bulk = list()
    sensor_y_vals_norm_bulk = list()
    for sensor_apnum in sensor_apnums:
        sensor_y_vals.append(list())
        sensor_y_vals_norm.append(list())
        sensor_y_vals_bulk.append(list())
        sensor_y_vals_norm_bulk.append(list())
    for cell_data in cellwave_by_cell_list:
        needed_apnum_idxes = cell_data['event_num_per_ap']>=plot_parameters['min_ap_per_apnum']
        apnums = cell_data['ap_group_ap_num_mean_per_ap'][needed_apnum_idxes]
        if 1 not in apnums:
            continue
        ys = cell_data['cawave_peak_amplitude_dff_mean_per_ap'][needed_apnum_idxes]
        
        y_1ap = ys[apnums==1][0]
        y_norms = ys/y_1ap
        for y,y_norm,apnum in zip(ys,y_norms,apnums):
            idx = np.where(sensor_apnums == apnum)[0][0]
            sensor_y_vals[idx].append(y)
            sensor_y_vals_norm[idx].append(y_norm)
        #% same thing for all APs and not for cell-wise averages            
        apnums = cell_data['ap_group_ap_num']
        ys = cell_data['cawave_peak_amplitude_dff']
        y_norms = ys/y_1ap
        for y,y_norm,apnum in zip(ys,y_norms,apnums):
            idx = np.where(sensor_apnums == apnum)[0][0]
            sensor_y_vals_bulk[idx].append(y)
            sensor_y_vals_norm_bulk[idx].append(y_norm)    
            
        
# =============================================================================
#         ax_amplitude.plot(apnums,ys,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .3)
#         ax_amplitude_norm.plot(apnums,y_norms,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .3)
# =============================================================================
    apnum_concat = list()
    y_mean = list()
    y_std = list()
    y_median = list()
    y_norm_mean = list()
    y_norm_median = list()
    y_norm_std = list()
    
    y_mean_bulk = list()
    y_std_bulk = list()
    y_median_bulk = list()
    y_norm_mean_bulk = list()
    y_norm_median_bulk = list()
    y_norm_std_bulk = list()
    
    eventnum = list()
    cellnum = list()
    for ys,ys_norm,apnum,ys_bulk,ys_norm_bulk in zip(sensor_y_vals,sensor_y_vals_norm,sensor_apnums,sensor_y_vals_bulk,sensor_y_vals_norm_bulk):
        apnum_concat.append(np.ones(len(ys))*apnum)
        y_mean.append(np.nanmean(ys))
        y_median.append(np.nanmedian(ys))
        y_std.append(np.nanstd(ys))
        y_norm_mean.append(np.nanmean(ys_norm))
        y_norm_median.append(np.nanmedian(ys_norm))
        y_norm_std.append(np.nanstd(ys_norm))
        cellnum.append(len(ys))
        
        y_mean_bulk.append(np.nanmean(ys_bulk))
        y_median_bulk.append(np.nanmedian(ys_bulk))
        y_std_bulk.append(np.nanstd(ys_bulk))
        y_norm_mean_bulk.append(np.nanmean(ys_norm_bulk))
        y_norm_median_bulk.append(np.nanmedian(ys_norm_bulk))
        y_norm_std_bulk.append(np.nanstd(ys_norm_bulk))
        eventnum.append(len(ys_bulk))
        #break
    #%
    y_mean = np.asarray(y_mean)
    y_median = np.asarray(y_median)
    y_std = np.asarray(y_std)
    y_norm_mean = np.asarray(y_norm_mean)
    y_norm_median = np.asarray(y_norm_median)
    y_norm_std = np.asarray(y_norm_std)
    cellnum = np.asarray(cellnum)
    y_mean_bulk = np.asarray(y_mean_bulk)
    y_median_bulk = np.asarray(y_median_bulk)
    y_std_bulk = np.asarray(y_std_bulk)
    y_norm_mean_bulk = np.asarray(y_norm_mean_bulk)
    y_norm_median_bulk = np.asarray(y_norm_median_bulk)
    y_norm_std_bulk = np.asarray(y_norm_std_bulk)
    eventnum = np.asarray(eventnum)
    needed_points = (cellnum >= plot_parameters['min_cell_num']) & (sensor_apnums<= plot_parameters['max_apnum'])
    needed_points_bulk = (eventnum >= plot_parameters['min_event_num'])  & (sensor_apnums<= plot_parameters['max_apnum'])
    ax_amplitude.errorbar(sensor_apnums[needed_points],
                          y_mean[needed_points],
                          yerr = y_std[needed_points]/np.sqrt(cellnum[needed_points]),
                          fmt='-',
                          color = plot_parameters['sensor_colors'][sensor])
    #sns.swarmplot(x =np.concatenate(apnum_concat),y = np.concatenate(sensor_y_vals),color=plot_parameters['sensor_colors'][sensor],ax = ax_amplitude,alpha = .9,size = 3)     
    ax_amplitude_norm.errorbar(sensor_apnums[needed_points],
                               y_norm_mean[needed_points],
                               yerr = y_norm_std[needed_points]/np.sqrt(cellnum[needed_points]),
                               fmt='-',
                               color = plot_parameters['sensor_colors'][sensor])
    #sns.swarmplot(x =np.concatenate(apnum_concat),y = np.concatenate(sensor_y_vals_norm),color=plot_parameters['sensor_colors'][sensor],ax = ax_amplitude_norm,alpha = .9,size = 3)   
    ax_cellnum.plot(sensor_apnums[needed_points],cellnum[needed_points],'o',color = plot_parameters['sensor_colors'][sensor],alpha = .8)
    
    ax_amplitude_bulk.errorbar(sensor_apnums[needed_points_bulk],
                          y_median_bulk[needed_points_bulk],
                          yerr = y_std_bulk[needed_points_bulk]/np.sqrt(eventnum[needed_points_bulk]),
                          fmt='-',
                          color = plot_parameters['sensor_colors'][sensor])
    
    ax_amplitude_norm_bulk.errorbar(sensor_apnums[needed_points_bulk],
                               y_norm_mean_bulk[needed_points_bulk],
                               yerr = y_norm_std_bulk[needed_points_bulk]/np.sqrt(eventnum[needed_points_bulk]),
                               fmt='-',
                               color = plot_parameters['sensor_colors'][sensor])
    
    ax_eventnum.semilogy(sensor_apnums[needed_points_bulk],eventnum[needed_points_bulk],'o',color = plot_parameters['sensor_colors'][sensor],alpha = .8)
    
    #break
    #%
ax_amplitude_norm_bulk.plot([0,8],[0,8],'k--')
ax_eventnum.set_xlabel('Number of spikes') 
ax_cellnum.set_ylabel('Number of events in dataset') 
ax_amplitude_norm_bulk.set_ylabel('Normalized amplitude')

ax_amplitude_norm.plot([0,6],[0,6],'k--')
ax_cellnum.set_xlim([0.5,5.5])
ax_cellnum.set_xlabel('Number of spikes') 
ax_cellnum.set_ylabel('Number of cells in dataset') 
#ax_amplitude_norm.set_xlabel('Number of spikes') 
ax_amplitude_norm.set_ylabel('Normalized amplitude')
#ax_amplitude.set_xlabel('Number of spikes') 
ax_amplitude.set_ylabel('Amplitude (dF/F)')
ax_amplitude.set_ylim([0,5])
ax_amplitude_norm.set_ylim([0,16])
ax_amplitude.set_title('cell-wise analysis')
ax_amplitude_bulk.set_title('event-wise analysis')
#%
figfile = os.path.join(figures_dir,'fig_multiple_ap_scatter_0ms_integration.{}')
fig_dff_ap_dependence.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )

#%%#######################################################################Multiple AP pyr - END################################################################################
#################################################################################%%############################################################################################

#%%#######################################################################Decay times - START################################################################################
#################################################################################%%############################################################################################
#%% Select events for putative pyramidal cells - Decay time stuff
leave_out_putative_interneurons = True
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':.5,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':1,
                      'min_post_isi':1,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':0,#0,5,10,20,50
                      'neuropil_subtraction':'0.8',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100
                      }
if leave_out_putative_interneurons:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['min_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=-100
    ca_wave_parameters['min_ap_ahp_amplitude_median']=100
if leave_out_putative_pyramidal_cells:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['max_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=100
    ca_wave_parameters['max_ap_ahp_amplitude_median']=-100    
cawave_properties_needed = ['cawave_baseline_dff_global',
                            'ap_group_ap_num',
                            'cawave_snr', 
                            'cawave_peak_amplitude_dff',
                            'cawave_peak_amplitude_f',
                            'cawave_baseline_f',
                            'cawave_baseline_f_std',
                            'cawave_rneu',
                            'cawave_rise_time',
                            'movie_hmotors_sample_z',
                            'movie_power',
                            'session_sensor_expression_time',
                            'roi_pixel_num',
                            'roi_dwell_time_multiplier',
                            'roi_f0_percentile',
                            'roi_f0_percentile_value']
ephys_properties_needed = ['ap_ahp_amplitude_median',
                           'ap_halfwidth_median',
                           'ap_full_width_median',
                           'ap_rise_time_median',
                           'ap_peak_to_trough_time_median',
                           'recording_mode']
#%
ca_waves_dict_pyr_decay_time = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)
#%
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':5,
                   'ignore_cells_below_median_snr':True} # filters both grand average and cell-wise traces
ca_traces_dict_1ap_pyr_decay,ca_traces_dict_by_cell_1ap_pyr_decay  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,
                                                                                                       ca_waves_dict_pyr_decay_time,
                                                                                                       wave_parameters)

#%
superres_parameters={'sensor_colors':sensor_colors,
                     'normalize_raw_trace_offset':True,#set to 0
                     'min_ap_per_apnum':10, # filters only cell-wise trace
                     'bin_step_for_cells':1,
                     'bin_size_for_cells':8,
                     'bin_step':1,
                     'bin_size':8,
                     'start_time':-50, # ms
                     'end_time':1000,#ms
                     'traces_to_use':['f_corr','dff','f_corr_normalized'],
                     'cell_wise_analysis':True,
                     'dogaussian':False,
                     'double_bin_size_for_old_sensors':False,
                     }
superresolution_traces_1ap_pyr_decay = utils_plot.calculate_superresolution_traces_for_all_sensors(ca_traces_dict_1ap_pyr_decay,
                                                                                                             ca_traces_dict_by_cell_1ap_pyr_decay,
                                                                                                             ca_wave_parameters,
                                                                                                             superres_parameters)
#%% fitting decay times
import scipy
rise_times = {'GCaMP7F':30,
              'XCaMPgf':30,
              '456':10,
              '686':10,
              '688':20}
end_times = {'GCaMP7F':750,
              'XCaMPgf':750,
              '456':770,
              '686':750,
              '688':750}
fit_parameters = {'rise_times':rise_times,
                  'end_times':end_times,
                  'trace_to_use':'f_corr_normalized',#'f_corr','dff','f_corr_normalized'
                  'double_exponential':True
                  }
rise_time_amplitude_ratio = 1
fig = plt.figure(figsize = [5,15])
ax_dict = dict()

for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
    y = superresolution_traces_1ap_pyr_decay[sensor]['{}_raw'.format(fit_parameters['trace_to_use'])]
    x = superresolution_traces_1ap_pyr_decay[sensor]['{}_raw_time'.format(fit_parameters['trace_to_use'])]
    if sensor_i == 0:
        ax_dict[sensor] = fig.add_subplot(len(ca_wave_parameters['sensors']),1,sensor_i+1)
        master_ax = ax_dict[sensor]
    else:
        ax_dict[sensor] = fig.add_subplot(len(ca_wave_parameters['sensors']),1,sensor_i+1,sharex = master_ax,sharey = master_ax)
# =============================================================================
#     x = superresolution_traces_1ap_pyr_decay[sensor]['{}_time'.format(fit_parameters['trace_to_use'])]
#     y = superresolution_traces_1ap_pyr_decay[sensor]['{}_mean'.format(fit_parameters['trace_to_use'])]
# =============================================================================
    idx_needed = (x>fit_parameters['rise_times'][sensor]) &(x<fit_parameters['end_times'][sensor])
    
    xvals = x[idx_needed]-fit_parameters['rise_times'][sensor]
    yvals = y[idx_needed]#/np.max(y[idx_needed])
    x_range = np.arange(np.min(xvals),np.max(xvals),1)
    if fit_parameters['double_exponential']:
        out = scipy.optimize.curve_fit(lambda t,a,b,c,d,e: a*np.exp(-t/b) + c + d*np.exp(-t/e),  xvals,  yvals,maxfev = 12000,bounds=((0,0,-0.05,0,0),(np.inf,2000,0.05,np.inf,2000)))#,bounds=((0,0,-0.000001,0,0),(np.inf,np.inf,0.00000001,np.inf,np.inf))
        #yfit = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2] +out[0][3]*np.exp(-xvals/out[0][4])
        yrange = out[0][0]*np.exp(-x_range/out[0][1])+out[0][2]+out[0][3]*np.exp(-x_range/out[0][4])
        yrange_norm = yrange/np.max(yrange)
        half_idx = np.argmax(yrange_norm<.5)
        half_time = x_range[half_idx]
        ax_dict[sensor].set_title('tau1: {:.2f} ms, tau2: {:.2f} ms, w_tau1: {:.2f}%, 0-{}% \n risetime: {:.2f} ms, half decay time: {:.2f}'.format(out[0][1],out[0][4],100*out[0][0]/(out[0][0]+out[0][3]),int(rise_time_amplitude_ratio*100),fit_parameters['rise_times'][sensor],half_time))
    else:
        out = scipy.optimize.curve_fit(lambda t,a,b,c: a*np.exp(-t/b) + c,  xvals,  yvals,bounds=((0,0,-.1),(np.inf,np.inf,.1)))#,bounds=((0,0,-0.000001),(np.inf,np.inf,0.00000001))
        #yfit = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2]
        yrange = out[0][0]*np.exp(-x_range/out[0][1])+out[0][2]
        yrange_norm = yrange/np.max(yrange)
        half_idx = np.argmax(yrange_norm<.5)
        half_time = x_range[half_idx]
        ax_dict[sensor].set_title('tau: {:.2f} ms, 0-{}% risetime: {:.2f} ms, half decay time: {:.2f}'.format(out[0][1],int(rise_time_amplitude_ratio*100),fit_parameters['rise_times'][sensor],half_time))
    
    
    
    ax_dict[sensor].plot(x,y,'ko',markersize = 3,alpha = .2,color = sensor_colors[sensor])
    ax_dict[sensor].plot(x_range+fit_parameters['rise_times'][sensor],yrange,'-',color = 'white',linewidth =6)#sensor_colors[sensor]
    ax_dict[sensor].axis('off')
    #
    
    #break
ax_dict[sensor].set_xlim([-50,600])    
ax_dict[sensor].set_ylim([-.5,1])    

utils_plot.add_scalebar_to_fig(ax_dict[sensor],
                               axis = 'x',
                               size = 100,
                               conversion = 1,
                               unit = r'ms',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5)
    
figfile = os.path.join(figures_dir,'fig_decay_fit.{}')
fig.savefig(figfile.format('png'),dpi=300)
# =============================================================================
# inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
# subprocess.run(inkscape_cmd )    
# =============================================================================
#%%#######################################################################Decay times - END################################################################################
#################################################################################%%############################################################################################
#%%#######################################################################INTERNEURONS - START################################################################################
#################################################################################%%############################################################################################
#%% Select events for putative interneurons cells a
leave_out_putative_interneurons = False
leave_out_putative_pyramidal_cells = True
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':.5,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':.5,
                      'min_post_isi':.5,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':5,#0,5,10,20,50
                      'neuropil_subtraction':'none',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100,
                      }
if leave_out_putative_interneurons:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['min_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=-100
    ca_wave_parameters['min_ap_ahp_amplitude_median']=100
if leave_out_putative_pyramidal_cells:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['max_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=100
    ca_wave_parameters['max_ap_ahp_amplitude_median']=-100    
cawave_properties_needed = ['cawave_baseline_dff_global',
                            'ap_group_ap_num',
                            'cawave_snr', 
                            'cawave_peak_amplitude_dff',
                            'cawave_peak_amplitude_f',
                            'cawave_baseline_f',
                            'cawave_baseline_f_std',
                            'cawave_rneu',
                            'cawave_rise_time',
                            'cawave_half_decay_time',
                            'movie_hmotors_sample_z',
                            'movie_power',
                            'session_sensor_expression_time',
                            'roi_pixel_num',
                            'roi_dwell_time_multiplier',
                            'roi_f0_percentile',
                            'roi_f0_percentile_value']
ephys_properties_needed = ['ap_ahp_amplitude_median',
                           'ap_halfwidth_median',
                           'ap_full_width_median',
                           'ap_rise_time_median',
                           'ap_peak_to_trough_time_median',
                           'recording_mode']
#%
ca_waves_dict_int = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)

#%%
#%% compare sensors with 1AP - scatter plots
import plot.imaging_gt_plot as imaging_plot
ap_num_to_plot = 1
min_ap_per_apnum = 1   
plot_parameters={'sensor_colors':sensor_colors,
                 'min_ap_per_apnum':min_ap_per_apnum,
                 'ap_num_to_plot':ap_num_to_plot,
                 'event_histogram_step' : 5,
                 'scatter_plot_alpha':.1,
                 'scatter_marker_size':3,
                 'order_cells_by':'cawave_snr_median_per_ap',#'roi_f0_percentile_median_per_ap',#
                 'correct_f0_with_power_depth':True}#cawave_snr_median_per_ap#cawave_peak_amplitude_dff_median_per_ap
#%
imaging_plot.scatter_plot_all_event_properties(ca_waves_dict_int,ca_wave_parameters,plot_parameters)
#%% download sensor kinetics
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':1} # filters both grand average and cell-wise traces
ca_traces_dict_1ap_int,ca_traces_dict_by_cell_1ap_int  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,
                                                                                           ca_waves_dict_int,
                                                                                           wave_parameters)

#%%
superres_parameters={'sensor_colors':sensor_colors,
                     'normalize_raw_trace_offset':True,#set to 0
                     'min_ap_per_apnum':10, # filters only cell-wise trace
                     'bin_step_for_cells':1,
                     'bin_size_for_cells':8,
                     'bin_step':1,
                     'bin_size':20,
                     'start_time':-100, # ms
                     'end_time':1000,#ms
                     'traces_to_use':['f_corr','dff','f_corr_normalized'],
                     'cell_wise_analysis':True,
                     'dogaussian':False
                     }
superresolution_traces_1ap_int_low_resolution = utils_plot.calculate_superresolution_traces_for_all_sensors(ca_traces_dict_1ap_int,
                                                                                                             ca_traces_dict_by_cell_1ap_int,
                                                                                                             ca_wave_parameters,
                                                                                                             superres_parameters)

#%% grand average - interneurons
import utils.utils_plot as utils_plot
superresolution_traces = superresolution_traces_1ap_int_low_resolution

fig_grand_average_traces_pyr = plt.figure()
ax_grand_average = fig_grand_average_traces_pyr.add_subplot(1,1,1)
plot_parameters={'sensor_colors':sensor_colors,
                 'start_time':-100, # ms
                 'end_time':500,#ms
                 'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'mean',#'mean',#'median'#'gauss'#
                 'normalize_to_max':False,
                 'linewidth':3,
                 'error_bar':'none',#se,sd
                 }
for sensor in ca_wave_parameters['sensors']:
    x = superresolution_traces[sensor]['{}_time'.format(plot_parameters['trace_to_use'])]
    y = superresolution_traces[sensor]['{}_{}'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
    if plot_parameters['error_bar'] == 'sd':
        error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]
    elif plot_parameters['error_bar'] == 'se':
        error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]/np.sqrt(superresolution_traces[sensor]['{}_n'.format(plot_parameters['trace_to_use'])])
    else:
        error =np.zeros(len(y))
    if plot_parameters['normalize_to_max']:
        error = error/np.nanmax(y)
        y = y/np.nanmax(y)
    color = plot_parameters['sensor_colors'][sensor]
    ax_grand_average.plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.9)
    if plot_parameters['error_bar'] != 'none':
        ax_grand_average.fill_between(x, y-error, y+error,color =color,alpha = .3)
ax_grand_average.set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
ax_grand_average.plot(0,-.05,'r|',markersize = 18)
utils_plot.add_scalebar_to_fig(ax_grand_average,
                    axis = 'y',
                    size = 5,
                    conversion = 100,
                    unit = r'dF/F%',
                    color = 'black',
                    location = 'top left',
                    linewidth = 5)

utils_plot.add_scalebar_to_fig(ax_grand_average,
                    axis = 'x',
                    size = 50,
                    conversion = 1,
                    unit = r'ms',
                    color = 'black',
                    location = 'top left',
                    linewidth = 5)
figfile = os.path.join(figures_dir,'fig_grand_average_traces_int.{}')
fig_grand_average_traces_pyr.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )

#%% each cell separately
superresolution_traces = superresolution_traces_1ap_int_low_resolution
plot_parameters={'sensor_colors':sensor_colors,
                 'start_time':-100, # ms
                 'end_time':500,#ms
                 'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'superresolution_function':'median',#'mean',#'median'#
                 'normalize_to_max':False,
                 'linewidth':2,
                 'alpha':.5
                 }

sensornum = len(ca_wave_parameters['sensors'])
fig_cell_wise_trace = plt.figure(figsize = [5,15])
ax_dict = dict()
for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
    try:
        ax_dict[sensor] = fig_cell_wise_trace.add_subplot(sensornum,1,sensor_i+1,sharey = ax_dict[ca_wave_parameters['sensors'][0]],sharex = ax_dict[ca_wave_parameters['sensors'][0]])
    except:
        ax_dict[sensor] = fig_cell_wise_trace.add_subplot(sensornum,1,sensor_i+1)
    color = plot_parameters['sensor_colors'][sensor]
    xs = superresolution_traces[sensor]['{}_time_per_cell'.format(plot_parameters['trace_to_use'])]
    ys = superresolution_traces[sensor]['{}_{}_per_cell'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
    for x,y in zip(xs,ys):
        ax_dict[sensor].plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = plot_parameters['alpha'])
    ax_dict[sensor].set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
    ax_dict[sensor].axis('off')
utils_plot.add_scalebar_to_fig(ax_dict[ca_wave_parameters['sensors'][0]],
                               axis = 'y',
                               size = 50,
                               conversion = 100,
                               unit = r'dF/F%',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5)

utils_plot.add_scalebar_to_fig(ax_dict[ca_wave_parameters['sensors'][0]],
                               axis = 'x',
                               size = 100,
                               conversion = 1,
                               unit = r'ms',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5)    
# =============================================================================
# figfile = os.path.join(figures_dir,'fig_fig_cell_wise_traces_pyr.{}')
# fig_cell_wise_trace.savefig(figfile.format('svg'))
# inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
# subprocess.run(inkscape_cmd )
# =============================================================================
#%% multiple ap
plot_parameters = {'min_ap_per_apnum': 1,
                   'min_cell_num':2,
                   'sensor_colors':sensor_colors}
fig_dff_ap_dependence = plt.figure(figsize = [5,10])
ax_amplitude = fig_dff_ap_dependence.add_subplot(3,1,1)
#ax_amplitude_swarm = ax_amplitude.twinx()
ax_amplitude_norm = fig_dff_ap_dependence.add_subplot(3,1,2,sharex = ax_amplitude)
ax_cellnum = fig_dff_ap_dependence.add_subplot(3,1,3,sharex = ax_amplitude)
for sensor in ca_wave_parameters['sensors']:
    cellwave_by_cell_list = ca_waves_dict_int[sensor]
    sensor_apnums = np.arange(1,100,1)
    sensor_y_vals = list()
    sensor_y_vals_norm = list()
    for sensor_apnum in sensor_apnums:
        sensor_y_vals.append(list())
        sensor_y_vals_norm.append(list())
    for cell_data in cellwave_by_cell_list:
        needed_apnum_idxes = cell_data['event_num_per_ap']>=plot_parameters['min_ap_per_apnum']
        apnums = cell_data['ap_group_ap_num_mean_per_ap'][needed_apnum_idxes]
        if 1 not in apnums:
            continue
        ys = cell_data['cawave_peak_amplitude_dff_mean_per_ap'][needed_apnum_idxes]
        
        y_norms = ys/ys[apnums==1][0]
        for y,y_norm,apnum in zip(ys,y_norms,apnums):
           
            idx = np.where(sensor_apnums == apnum)[0][0]
            sensor_y_vals[idx].append(y)
            sensor_y_vals_norm[idx].append(y_norm)
# =============================================================================
#         ax_amplitude.plot(apnums,ys,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .3)
#         ax_amplitude_norm.plot(apnums,y_norms,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .3)
# =============================================================================
    apnum_concat = list()
    y_mean = list()
    y_std = list()
    y_median = list()
    y_norm_mean = list()
    y_norm_median = list()
    y_norm_std = list()
    cellnum = list()
    for ys,ys_norm,apnum in zip(sensor_y_vals,sensor_y_vals_norm,sensor_apnums):
        apnum_concat.append(np.ones(len(ys))*apnum)
        y_mean.append(np.nanmean(ys))
        y_median.append(np.nanmedian(ys))
        y_std.append(np.nanstd(ys))
        y_norm_mean.append(np.nanmean(ys_norm))
        y_norm_median.append(np.nanmedian(ys_norm))
        y_norm_std.append(np.nanstd(ys_norm))
        cellnum.append(len(ys))
    y_mean = np.asarray(y_mean)
    y_median = np.asarray(y_median)
    y_std = np.asarray(y_std)
    y_norm_mean = np.asarray(y_norm_mean)
    y_norm_median = np.asarray(y_norm_median)
    y_norm_std = np.asarray(y_norm_std)
    cellnum = np.asarray(cellnum)
    needed_points = cellnum >= plot_parameters['min_cell_num']
    ax_amplitude.errorbar(sensor_apnums[needed_points],
                          y_median[needed_points],
                          yerr = y_std[needed_points],
                          fmt='o-',
                          color = plot_parameters['sensor_colors'][sensor])
    #sns.swarmplot(x =np.concatenate(apnum_concat),y = np.concatenate(sensor_y_vals),color=plot_parameters['sensor_colors'][sensor],ax = ax_amplitude,alpha = .9,size = 3)     
    ax_amplitude_norm.errorbar(sensor_apnums[needed_points],
                               y_norm_median[needed_points],
                               yerr = y_norm_std[needed_points],
                               fmt='o-',
                               color = plot_parameters['sensor_colors'][sensor])
    #sns.swarmplot(x =np.concatenate(apnum_concat),y = np.concatenate(sensor_y_vals_norm),color=plot_parameters['sensor_colors'][sensor],ax = ax_amplitude_norm,alpha = .9,size = 3)   
    ax_cellnum.plot(sensor_apnums[needed_points],cellnum[needed_points],'o',color = plot_parameters['sensor_colors'][sensor],alpha = .8)
    #break
ax_amplitude_norm.plot([0,3],[0,3],'k--')
ax_cellnum.set_xlim([0,4])
ax_cellnum.set_xlabel('Number of spikes') 
ax_cellnum.set_ylabel('Number of cells in dataset') 
#ax_amplitude_norm.set_xlabel('Number of spikes') 
ax_amplitude_norm.set_ylabel('Amplitude normalized to single spike event')
#ax_amplitude.set_xlabel('Number of spikes') 
ax_amplitude.set_ylabel('Amplitude (dF/F)')

#%%#######################################################################INTERNEURONS - END################################################################################
#################################################################################%%############################################################################################


#%%#######################################################################SAMPLE TRACES - START################################################################################
#################################################################################%%############################################################################################
#%% Select events for putative pyramidal cells - sample traces
leave_out_putative_interneurons = True
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':.5,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':1,
                      'min_post_isi':.5,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':5,#0,5,10,20,50
                      'neuropil_subtraction':'0.8',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100,
                      }
if leave_out_putative_interneurons:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['min_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['max_ap_peak_to_trough_time_median']=-100
    ca_wave_parameters['min_ap_ahp_amplitude_median']=100
if leave_out_putative_pyramidal_cells:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=0.65
    ca_wave_parameters['max_ap_ahp_amplitude_median']=.1
else:
    ca_wave_parameters['min_ap_peak_to_trough_time_median']=100
    ca_wave_parameters['max_ap_ahp_amplitude_median']=-100    
cawave_properties_needed = ['cawave_baseline_dff_global',
                            'ap_group_ap_num',
                            'cawave_snr', 
                            'cawave_peak_amplitude_dff',
                            'cawave_peak_amplitude_f',
                            'cawave_baseline_f',
                            'cawave_baseline_f_std',
                            'cawave_rneu',
                            'cawave_rise_time',
                            'cawave_half_decay_time',
                            'movie_hmotors_sample_z',
                            'movie_power',
                            'session_sensor_expression_time',
                            'roi_pixel_num',
                            'roi_dwell_time_multiplier',
                            'roi_f0_percentile',
                            'roi_f0_percentile_value']
ephys_properties_needed = ['ap_ahp_amplitude_median',
                           'ap_halfwidth_median',
                           'ap_full_width_median',
                           'ap_rise_time_median',
                           'ap_peak_to_trough_time_median',
                           'recording_mode']
#%
ca_waves_dict_pyr = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)
#%% Sample traces#%% compare sensors with 1AP - scatter plots
import plot.imaging_gt_plot as imaging_plot
ap_num_to_plot = 1
min_ap_per_apnum = 2    
plot_parameters={'sensor_colors':sensor_colors,
                 'min_ap_per_apnum':min_ap_per_apnum,
                 'ap_num_to_plot':ap_num_to_plot,
                 'event_histogram_step' : 5,
                 'scatter_plot_alpha':.1,
                 'scatter_marker_size':3,
                 'order_cells_by':'cawave_snr_median_per_ap',#'roi_f0_percentile_median_per_ap',#
                 'correct_f0_with_power_depth':True}#cawave_snr_median_per_ap#cawave_peak_amplitude_dff_median_per_ap
#%
imaging_plot.scatter_plot_all_event_properties(ca_waves_dict_pyr,ca_wave_parameters,plot_parameters)


#%% sample traces
import utils.utils_ephys as utils_ephys
#% Putative pyramidal cells
movie_length = 40 #s
zoom_length = 1.5#2
sweeps = {'456':[{'subject_id':471993,'session':1,'cell_number':2,'sweep_number':2,'start_time':20,'zoom_start_time':44.5},
                 {'subject_id':478411,'session':1,'cell_number':1,'sweep_number':2,'start_time':102,'zoom_start_time':109}],
          '686':[{'subject_id':479117,'session':1,'cell_number':8,'sweep_number':4,'start_time':110,'zoom_start_time':140}],#132
          '688':[{'subject_id':472181,'session':1,'cell_number':1,'sweep_number':3,'start_time':16,'zoom_start_time':22.4},
                 {'subject_id':472181,'session':1,'cell_number':7,'sweep_number':3,'start_time':57,'zoom_start_time':94.25}]
          }
# =============================================================================
# # putative interneurons
# movie_length = 80 #s
# zoom_length = 10#2
# sweeps = {'456':[{'subject_id':478407,'session':1,'cell_number':7,'sweep_number':4,'start_time':35,'zoom_start_time':44}],
#           '686':[{'subject_id':472180,'session':1,'cell_number':1,'sweep_number':1,'start_time':50,'zoom_start_time':56}],#132
#           '688':[{'subject_id':479119,'session':1,'cell_number':9,'sweep_number':2,'start_time':25,'zoom_start_time':69}]
#           }
# =============================================================================

# =============================================================================
# sweeps = {'456':[{'subject_id':471993,'session':1,'cell_number':2,'sweep_number':2,'start_time':20}],
#           '686':[{'subject_id':479570,'session':1,'cell_number':1,'sweep_number':4,'start_time':31},
#                  {'subject_id':479117,'session':1,'cell_number':8,'sweep_number':4,'start_time':110}],
#           '688':[{'subject_id':472181,'session':1,'cell_number':3,'sweep_number':4,'start_time':80},
#                  {'subject_id':472181,'session':1,'cell_number':1,'sweep_number':0,'start_time':37},
#                  {'subject_id':472181,'session':1,'cell_number':1,'sweep_number':3,'start_time':16}]
#           }
# =============================================================================
plot_parameters = {'gaussian_filter_sigma':5,
                   }
cawave_rneu = .8
rownum = 4
cax_limits = [0,99]
alpha_roi = .3
fig_longtrace = plt.figure(figsize = [10,10])
fig_zoom = plt.figure(figsize = [2,10])
fig_meanimages = plt.figure(figsize = [40,20])
axs_longtrace =list()
axs_zoomtrace = list()
axs_meanimages = list()
#%
for sensor in sweeps.keys():
    sweeps_now =sweeps[sensor]
    for sweep in sweeps_now:
        start_time = sweep.pop('start_time',None)
        zoom_start_time = sweep.pop('zoom_start_time',None)
        
        mean_image,movie_x_size,movie_y_size,movie_pixel_size,roi_xpix,roi_ypix = (imaging.ROI()*imaging.Movie()*imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()&sweep&'channel_number = 1').fetch1('registered_movie_mean_image','movie_x_size','movie_y_size','movie_pixel_size','roi_xpix','roi_ypix')
        movie_pixel_size = float(movie_pixel_size)
        mean_image_red = (imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()&sweep&'channel_number = 2').fetch1('registered_movie_mean_image')
        axs_meanimages.append(fig_meanimages.add_subplot(rownum,2,len(axs_meanimages)+1))
        im_green = axs_meanimages[-1].imshow(mean_image,cmap = 'gray', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)])#, aspect='auto')
        clim = np.percentile(mean_image.flatten(),cax_limits)
        im_green.set_clim(clim)
        ROI = np.zeros_like(mean_image)*np.nan
        ROI[roi_ypix,roi_xpix] = 1
        axs_meanimages[-1].imshow(ROI,cmap = 'Greens', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)],alpha = alpha_roi,vmin = 0, vmax = 1)#, aspect='auto'
        utils_plot.add_scalebar_to_fig(axs_meanimages[-1],
                                       axis = 'x',
                                       size = 20,
                                       conversion = 1,
                                       unit = r'$\mu$m',
                                       color = 'white',
                                       location = 'top right',
                                       linewidth = 5) 
        axs_meanimages.append(fig_meanimages.add_subplot(rownum,2,len(axs_meanimages)+1))
        im_red = axs_meanimages[-1].imshow(mean_image_red,cmap = 'gray', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)])#, aspect='auto')
        clim = np.percentile(mean_image_red.flatten(),cax_limits)
        im_red.set_clim(clim)
        utils_plot.add_scalebar_to_fig(axs_meanimages[-1],
                                       axis = 'x',
                                       size = 20,
                                       conversion = 1,
                                       unit = r'$\mu$m',
                                       color = 'white',
                                       location = 'top right',
                                       linewidth = 5) 
        #continue
        if len(axs_longtrace)==0:
            axs_longtrace.append(fig_longtrace.add_subplot(rownum,1,len(axs_longtrace)+1))
        else:
            axs_longtrace.append(fig_longtrace.add_subplot(rownum,1,len(axs_longtrace)+1,sharey = axs_longtrace[0]))
        if len(axs_zoomtrace)==0:
            axs_zoomtrace.append(fig_zoom.add_subplot(rownum,1,len(axs_zoomtrace)+1))
        else:
            axs_zoomtrace.append(fig_zoom.add_subplot(rownum,1,len(axs_zoomtrace)+1,sharey = axs_zoomtrace[0]))
            
        
        table = ephys_cell_attached.Sweep()*imaging.ROI()*ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()*imaging_gt.ROISweepCorrespondance()*imaging.ROITrace()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()*ephys_cell_attached.Cell()*experiment.Session()&sweep&'channel_number = 1'&'neuropil_number = 1'
        apgroup_table = table*ephysanal_cell_attached.APGroup()
        ap_group_start_time,ap_group_end_time,ap_group_ap_num = apgroup_table.fetch('ap_group_start_time','ap_group_end_time','ap_group_ap_num')
        
        roi_f, neuropil_f,response_trace,sample_rate,frame_times,roi_time_offset,cell_recording_start, session_time,sweep_start_time= table.fetch1('roi_f','neuropil_f','response_trace','sample_rate','frame_times','roi_time_offset','cell_recording_start', 'session_time','sweep_start_time')
        try:
            squarepulse_indices = (table*ephysanal_cell_attached.SquarePulse()).fetch('square_pulse_start_idx')
            response_trace = utils_ephys.remove_stim_artefacts_with_stim(response_trace,sample_rate,squarepulse_indices)
        except:
            pass
        #response_trace = utils_ephys.remove_stim_artefacts_without_stim(response_trace,sample_rate)
        ap_group_start_time = np.asarray(ap_group_start_time,float)-float(sweep_start_time)
        ap_group_end_time = np.asarray(ap_group_end_time,float)-float(sweep_start_time)
        ap_group_mean_time = np.mean([ap_group_start_time,ap_group_end_time],0)
        frame_rate = 1/np.median(np.diff(frame_times))/1000
        if plot_parameters['gaussian_filter_sigma']>0:
            roi_f = utils_plot.gaussFilter(roi_f,frame_rate,plot_parameters['gaussian_filter_sigma'])
            neuropil_f = utils_plot.gaussFilter(neuropil_f,frame_rate,plot_parameters['gaussian_filter_sigma']) 
        roi_f_corr = roi_f-neuropil_f*cawave_rneu
        f0 = np.percentile(roi_f_corr,10)
        roi_dff = (roi_f_corr-f0)/f0
        
        ephys_time = np.arange(len(response_trace))/sample_rate
        frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))
        #%
        response_trace_filt = utils_ephys.hpFilter(response_trace, 50, 1, sample_rate, padding = True)
        response_trace_filt = utils_ephys.gaussFilter(response_trace_filt,sample_rate,sigma = .0001)
        response_trace_filt_scaled = (response_trace_filt-np.min(response_trace_filt))/(np.max(response_trace_filt)-np.min(response_trace_filt))
        response_trace_filt_scaled=response_trace_filt_scaled*1-1.5
        #plt.plot(ephys_time,response_trace)
        ephys_idx_needed = (ephys_time>=start_time) & (ephys_time<=start_time+movie_length)
        dff_idx_needed = (frame_times>=start_time) & (frame_times<=start_time+movie_length)
        
        axs_longtrace[-1].plot(ephys_time[ephys_idx_needed],response_trace_filt_scaled[ephys_idx_needed],color = 'gray')
        axs_longtrace[-1].plot(frame_times[dff_idx_needed],roi_dff[dff_idx_needed],color = sensor_colors[sensor])
        axs_longtrace[-1].set_xlim([start_time,start_time+movie_length])
        axs_longtrace[-1].axis('off')
        
        ephys_idx_needed = (ephys_time>=zoom_start_time) & (ephys_time<=zoom_start_time+zoom_length)
        dff_idx_needed = (frame_times>=zoom_start_time) & (frame_times<=zoom_start_time+zoom_length)
        
        axs_zoomtrace[-1].plot(ephys_time[ephys_idx_needed],response_trace_filt_scaled[ephys_idx_needed],color = 'gray')
        axs_zoomtrace[-1].plot(frame_times[dff_idx_needed],roi_dff[dff_idx_needed],color = sensor_colors[sensor])
        axs_zoomtrace[-1].set_xlim([zoom_start_time,zoom_start_time+zoom_length])
        axs_zoomtrace[-1].axis('off')
        
        rect_low = np.min(response_trace_filt_scaled[ephys_idx_needed])#-.9
        rect_high = np.max(roi_dff[dff_idx_needed])
        axs_longtrace[-1].plot([zoom_start_time,zoom_start_time+zoom_length,zoom_start_time+zoom_length,zoom_start_time,zoom_start_time],[rect_low,rect_low,rect_high,rect_high,rect_low],'k--')
        for ap_group_start_time_now,ap_group_ap_num_now in zip(ap_group_mean_time,ap_group_ap_num):
            if ap_group_start_time_now >start_time and ap_group_start_time_now<start_time+movie_length:
                if ap_group_ap_num_now==1:
                    textnow = '*'
                else:
                    textnow = str(ap_group_ap_num_now)
                axs_longtrace[-1].text(ap_group_start_time_now,-2,textnow,ha = 'center')
            
            if ap_group_start_time_now >zoom_start_time and ap_group_start_time_now<zoom_start_time+zoom_length:
                if ap_group_ap_num_now==1:
                    textnow = '*'
                else:
                    textnow = str(ap_group_ap_num_now)
                axs_zoomtrace[-1].text(ap_group_start_time_now,-2,textnow,ha = 'center')
# =============================================================================
#         break
#     break
# =============================================================================
axs_longtrace[-1].set_ylim([-2,5])
utils_plot.add_scalebar_to_fig(axs_longtrace[-1],
                               axis = 'x',
                               size = 5,
                               conversion = 1,
                               unit = r's',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5) 
utils_plot.add_scalebar_to_fig(axs_longtrace[-1],
                               axis = 'y',
                               size = 100,
                               conversion = 100,
                               unit = r'% dF/F',
                               color = 'black',
                               location = 'top right',
                               linewidth = 5) 

axs_longtrace[-1].set_ylim([-2,5])
#%
utils_plot.add_scalebar_to_fig(axs_zoomtrace[-1],
                               axis = 'x',
                               size = 500,
                               conversion = 1000,
                               unit = r'ms',
                               color = 'black',
                               location = 'bottom right',
                               linewidth = 5) 
utils_plot.add_scalebar_to_fig(axs_zoomtrace[-1],
                               axis = 'y',
                               size = 100,
                               conversion = 100,
                               unit = r'% dF/F',
                               color = 'black',
                               linewidth = 5) 
        #%
figfile = os.path.join(figures_dir,'fig_sample_traces_long.{}')
fig_longtrace.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )
        
figfile = os.path.join(figures_dir,'fig_sample_traces_zoom.{}')
fig_zoom.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd ) 


figfile = os.path.join(figures_dir,'fig_sample_traces_mean_images.{}')
fig_meanimages.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )
#%%#######################################################################SAMPLE TRACES - END################################################################################
#################################################################################%%############################################################################################