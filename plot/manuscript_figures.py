import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
import plot.cell_attached_ephys_plot as ephys_plot
import plot.imaging_gt_plot as imaging_plot
import utils.utils_plot as utils_plot
import utils.utils_ephys as utils_ephys
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
                      'sensors': ['GCaMP7F','456','686','688'], # this determines the order of the sensors,'XCaMPgf'
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
#%% Sample traces





#%% download sensor kinetics
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':1} # filters both grand average and cell-wise traces
ca_traces_dict_1ap_pyr,ca_traces_dict_by_cell_1ap_pyr  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,
                                                                                           ca_waves_dict_pyr,
                                                                                           wave_parameters)

#%%
#%%
from utils import utils_plot
superres_parameters={'sensor_colors':sensor_colors,
                     'normalize_raw_trace_offset':True,#set to 0
                     'min_ap_per_apnum':10, # filters only cell-wise trace
                     'bin_step_for_cells':.5,
                     'bin_size_for_cells':8,
                     'bin_step':.5,
                     'bin_size':8,
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
                     'bin_size':2,
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
                 'start_time':-20,#-50, # ms
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
                 'partial_risetime_target':.8}
ca_waves_dict = ca_waves_dict_pyr
superresolution_traces = superresolution_traces_1ap_pyr_low_resolution
amplitudes_all = list()
amplitudes_all_superresolution = list()
amplitudes_idx_all = list()
amplitudes_idx_all_superresolution = list()
risetimes_all = list()
risetimes_all_superresolution = list()
half_decay_times_all = list()
half_decay_times_all_superresolution = list()
partial_risetimes_all_superresolution = list()
fig_scatter_plots_1ap = plt.figure(figsize = [10,15])
ax_amplitude = fig_scatter_plots_1ap.add_subplot(4,2,1)
ax_risetime = fig_scatter_plots_1ap.add_subplot(4,2,3)
ax_half_decay_time= fig_scatter_plots_1ap.add_subplot(4,2,5)
ax_amplitude_superres = fig_scatter_plots_1ap.add_subplot(4,2,2,sharey = ax_amplitude)
ax_risetime_superres = fig_scatter_plots_1ap.add_subplot(4,2,4,sharey = ax_risetime)
ax_half_decay_time_superres = fig_scatter_plots_1ap.add_subplot(4,2,6,sharey = ax_half_decay_time)
ax_partial_risetime_superres = fig_scatter_plots_1ap.add_subplot(4,2,8)
for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
    sensor_amplitudes = list()
    sensor_rise_times = list()
    sensor_half_decay_times = list()
    sensor_amplitudes_superresolution = list()
    sensor_rise_times_superresolution = list()
    sensor_half_decay_times_superresolution = list()
    sensor_partial_rise_times_superresolution = list()
    
    # each transient separately
    for ca_waves_dict_cell in ca_waves_dict[sensor]:
        needed_apnum_idx = np.where((ca_waves_dict_cell['event_num_per_ap']>=wave_parameters['min_ap_per_apnum']) & (ca_waves_dict_cell['ap_group_ap_num_mean_per_ap']==wave_parameters['ap_num_to_plot']))[0]
        if len(needed_apnum_idx)==0:
            print('not enough events')
            continue
        else:
            needed_apnum_idx = needed_apnum_idx[0]
        amplitude = ca_waves_dict_cell['cawave_peak_amplitude_{}_{}_per_ap'.format(wave_parameters['trace_to_use'],wave_parameters['function_to_use'])][needed_apnum_idx]
        amplitudes_all.append(amplitude)   
        amplitudes_idx_all.append(sensor_i)
        sensor_amplitudes.append(amplitude)
        
        rise_time = ca_waves_dict_cell['cawave_rise_time_{}_per_ap'.format(wave_parameters['function_to_use'])][needed_apnum_idx]
        sensor_rise_times.append(rise_time)
        risetimes_all.append(rise_time)
        
        half_decay_time = ca_waves_dict_cell['cawave_half_decay_time_{}_per_ap'.format(wave_parameters['function_to_use'])][needed_apnum_idx]
        sensor_half_decay_times.append(half_decay_time)
        half_decay_times_all.append(half_decay_time)
        
    ax_amplitude.bar(sensor_i,np.median(sensor_amplitudes),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
    ax_risetime.bar(sensor_i,np.median(sensor_rise_times),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
    ax_half_decay_time.bar(sensor_i,np.median(sensor_half_decay_times),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
    
    # from superresolution traces
    xs = superresolution_traces[sensor]['{}_time_per_cell'.format(wave_parameters['trace_to_use'])]
    ys = superresolution_traces[sensor]['{}_{}_per_cell'.format(wave_parameters['trace_to_use'],wave_parameters['superresolution_function'])]
    ns = superresolution_traces[sensor]['{}_n_per_cell'.format(wave_parameters['trace_to_use'])]
    for x,y,n in zip(xs,ys,ns):
        if np.max(n)<wave_parameters['min_ap_per_apnum']:
            print('not enough events')
            continue
        baseline = np.nanmean(y[x<0])
        amplitudes_idx_all_superresolution.append(sensor_i)
        idx_to_use = x<wave_parameters['max_rise_time']
        amplitude = np.max(y[idx_to_use])-baseline
        peak_idx = np.argmax(y[idx_to_use])
        rise_time = x[idx_to_use][peak_idx]
        
        
        
        y_norm = (y-baseline)/amplitude
        half_decay_time_idx = np.argmax(y_norm[peak_idx:]<.5)+peak_idx
        half_decay_time = x[half_decay_time_idx]
        
        partial_rise_time_idx = np.argmax(y_norm>plot_parameters['partial_risetime_target'])
        partial_rise_time = x[partial_rise_time_idx]
        
        sensor_amplitudes_superresolution.append(amplitude)
        amplitudes_all_superresolution.append(amplitude)
        sensor_rise_times_superresolution.append(rise_time)
        risetimes_all_superresolution.append(rise_time)
        sensor_half_decay_times_superresolution.append(half_decay_time)
        half_decay_times_all_superresolution.append(half_decay_time)
        
        partial_risetimes_all_superresolution.append(partial_rise_time)
        sensor_partial_rise_times_superresolution.append(partial_rise_time)
    
    
        
    ax_amplitude_superres.bar(sensor_i,np.median(sensor_amplitudes_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
    ax_risetime_superres.bar(sensor_i,np.median(sensor_rise_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 
    ax_half_decay_time_superres.bar(sensor_i,np.median(sensor_half_decay_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 
    ax_partial_risetime_superres.bar(sensor_i,np.median(sensor_partial_rise_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 

sns.swarmplot(x =amplitudes_idx_all,y = amplitudes_all,color='black',ax = ax_amplitude,alpha = .6)       
sns.swarmplot(x =amplitudes_idx_all,y = risetimes_all,color='black',ax = ax_risetime,alpha = .6)    
sns.swarmplot(x =amplitudes_idx_all,y = half_decay_times_all,color='black',ax = ax_half_decay_time,alpha = .6)  




ax_amplitude.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_amplitude.set_xticklabels(ca_wave_parameters['sensors'])    
ax_amplitude.set_ylabel('Amplitude (dF/F)')
ax_amplitude.set_title('Calculated from single transients')
ax_risetime.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_risetime.set_xticklabels(ca_wave_parameters['sensors'])  
ax_risetime.set_ylabel('Time to peak (ms)')
#ax_risetime.set_title('Calculated from single transients')
ax_half_decay_time.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_half_decay_time.set_xticklabels(ca_wave_parameters['sensors'])  
ax_half_decay_time.set_ylabel('Half decay time (ms)')
#ax_half_decay_time.set_title('Calculated from single transients')


sns.swarmplot(x =amplitudes_idx_all_superresolution,y = amplitudes_all_superresolution,color='black',ax = ax_amplitude_superres,alpha = .6)       
sns.swarmplot(x =amplitudes_idx_all_superresolution,y = risetimes_all_superresolution,color='black',ax = ax_risetime_superres,alpha = .6)    
sns.swarmplot(x =amplitudes_idx_all_superresolution,y = half_decay_times_all_superresolution,color='black',ax = ax_half_decay_time_superres,alpha = .6)  
sns.swarmplot(x =amplitudes_idx_all_superresolution,y = partial_risetimes_all_superresolution,color='black',ax = ax_partial_risetime_superres,alpha = .6)  

for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
    # calculate from main superresolution trace
    x = superresolution_traces[sensor]['{}_time'.format(wave_parameters['trace_to_use'])]
    y = superresolution_traces[sensor]['{}_{}'.format(wave_parameters['trace_to_use'],wave_parameters['superresolution_function'])]
    baseline = np.nanmean(y[x<0])
    
    idx_to_use = x<wave_parameters['max_rise_time']
    amplitude = np.max(y[idx_to_use])-baseline
    peak_idx = np.argmax(y[idx_to_use])
    rise_time = x[idx_to_use][peak_idx]
    y_norm = (y-baseline)/amplitude
    half_decay_time_idx = np.argmax(y_norm[peak_idx:]<.5)+peak_idx
    half_decay_time = x[half_decay_time_idx]
    partial_rise_time_idx = np.argmax(y_norm>plot_parameters['partial_risetime_target'])
    partial_rise_time = x[partial_rise_time_idx]
    
    ax_amplitude_superres.plot(sensor_i,amplitude,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
    ax_risetime_superres.plot(sensor_i,rise_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
    ax_half_decay_time_superres.plot(sensor_i,half_decay_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
    ax_partial_risetime_superres.plot(sensor_i,partial_rise_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)

ax_amplitude_superres.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_amplitude_superres.set_xticklabels(ca_wave_parameters['sensors'])    
ax_amplitude_superres.set_ylabel('Amplitude (dF/F)')
ax_amplitude_superres.set_title('Calculated from superresolution traces')
ax_risetime_superres.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_risetime_superres.set_xticklabels(ca_wave_parameters['sensors'])  
ax_risetime_superres.set_ylabel('Time to peak (ms)')


ax_partial_risetime_superres.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_partial_risetime_superres.set_xticklabels(ca_wave_parameters['sensors'])  
ax_partial_risetime_superres.set_ylabel('0-{:.0f}% risetime (ms)'.format(plot_parameters['partial_risetime_target']*100))
#ax_risetime_superres.set_title('Calculated from superresolution traces')

ax_half_decay_time_superres.set_xticks(np.arange(len(ca_wave_parameters['sensors'])))
ax_half_decay_time_superres.set_xticklabels(ca_wave_parameters['sensors'])  
ax_half_decay_time_superres.set_ylabel('Half decay time (ms)')
#ax_half_decay_time_superres.set_title('Calculated from superresolution traces')
    #break
 #%     
figfile = os.path.join(figures_dir,'fig_amplitude_rise_time_scatter_pyr_1ap.{}')
fig_scatter_plots_1ap.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )    
    

#%%
# =============================================================================
# #%% calculate superresolution traces - high resolution for rise times and a low resolution for amplitudes
# superres_parameters={'normalize_raw_trace_offset':True,#set to 0
#                      'min_ap_per_apnum':5,
#                      'bin_step_for_cells':1,
#                      'bin_size_for_cells':4,
#                      'bin_step':.5,
#                      'bin_size':4,#8
#                      'start_time':-50, # ms
#                      'end_time':1000,#ms
#                      'traces_to_use':['f_corr','dff','f_corr_normalized'],
#                      'cell_wise_analysis':False,
#                      'dogaussian':True,
#                      'double_bin_size_for_old_sensors':False,
#                      }
# superresolution_traces_1ap_pyr_high_resolution = utils_plot.calculate_superresolution_traces_for_all_sensors(ca_traces_dict_1ap_pyr,
#                                                                                                              ca_traces_dict_by_cell_1ap_pyr,
#                                                                                                              ca_wave_parameters,
#                                                                                                              superres_parameters)
# 
# 
# #%% grand average - high resolution - rise time
# import utils.utils_plot as utils_plot
# 
# 
# fig_grand_average_traces_pyr_highres = plt.figure(figsize = [5,10])
# ax_grand_average = fig_grand_average_traces_pyr_highres.add_subplot(2,1,1)
# ax_grand_average_normalized = fig_grand_average_traces_pyr_highres.add_subplot(2,1,2)
# plot_parameters={'sensor_colors':sensor_colors,
#                  'low_res_sensors':[],#'GCaMP7F', 'XCaMPgf', '456', '686', '688'
#                  'start_time':-10, # ms
#                  'end_time':50,#ms
#                  'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
#                  'trace_to_use_2':'dff',
#                  'superresolution_function':'mean',#'mean',#'median'#
#                  'normalize_to_max':False,
#                  'normalize_to_max_2':True,
#                  'linewidth':3,
#                  'error_bar':'none',#se,sd
#                  }
# for sensor in ca_wave_parameters['sensors']:
#     if sensor in plot_parameters['low_res_sensors']:
#         superresolution_traces = superresolution_traces_1ap_pyr_low_resolution
#     else:
#         superresolution_traces = superresolution_traces_1ap_pyr_high_resolution
#     x = superresolution_traces[sensor]['{}_time'.format(plot_parameters['trace_to_use'])]
#     y = superresolution_traces[sensor]['{}_{}'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
#     if plot_parameters['error_bar'] == 'sd':
#         error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]
#     elif plot_parameters['error_bar'] == 'se':
#         error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use'])]/np.sqrt(superresolution_traces[sensor]['{}_n'.format(plot_parameters['trace_to_use'])])
#     else:
#         error =np.zeros(len(y))
#     if plot_parameters['normalize_to_max']:
#         error = error/np.nanmax(y)
#         y = y/np.nanmax(y)
#     color = plot_parameters['sensor_colors'][sensor]
#     ax_grand_average.plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.8)
#     if plot_parameters['error_bar'] != 'none':
#         ax_grand_average.fill_between(x, y-error, y+error,color =color,alpha = .3)
# ax_grand_average.set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
# ax_grand_average.plot(0,-.05,'r|',markersize = 18)
# utils_plot.add_scalebar_to_fig(ax_grand_average,
#                     axis = 'y',
#                     size = 10,
#                     conversion = 100,
#                     unit = r'dF/F%',
#                     color = 'black',
#                     location = 'top left',
#                     linewidth = 5)
# 
# utils_plot.add_scalebar_to_fig(ax_grand_average,
#                     axis = 'x',
#                     size = 10,
#                     conversion = 1,
#                     unit = r'ms',
#                     color = 'black',
#                     location = 'top left',
#                     linewidth = 5)
# 
# for sensor in ca_wave_parameters['sensors']:
#     if sensor in plot_parameters['low_res_sensors']:
#         superresolution_traces = superresolution_traces_1ap_pyr_low_resolution
#     else:
#         superresolution_traces = superresolution_traces_1ap_pyr_high_resolution
#     x = superresolution_traces[sensor]['{}_time'.format(plot_parameters['trace_to_use_2'])]
#     y = superresolution_traces[sensor]['{}_{}'.format(plot_parameters['trace_to_use_2'],plot_parameters['superresolution_function'])]
#     if plot_parameters['error_bar'] == 'sd':
#         error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use_2'])]
#     elif plot_parameters['error_bar'] == 'se':
#         error = superresolution_traces[sensor]['{}_sd'.format(plot_parameters['trace_to_use_2'])]/np.sqrt(superresolution_traces[sensor]['{}_n'.format(plot_parameters['trace_to_use_2'])])
#     else:
#         error =np.zeros(len(y))
#     if plot_parameters['normalize_to_max_2']:
#         error = error/np.nanmax(y)
#         y = y/np.nanmax(y)
#     color = plot_parameters['sensor_colors'][sensor]
#     ax_grand_average_normalized.plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.8)
#     if plot_parameters['error_bar'] != 'none':
#         ax_grand_average_normalized.fill_between(x, y-error, y+error,color =color,alpha = .3)
# ax_grand_average_normalized.set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
# ax_grand_average_normalized.plot(0,-.05,'r|',markersize = 18)
# utils_plot.add_scalebar_to_fig(ax_grand_average_normalized,
#                     axis = 'y',
#                     size = 10,
#                     conversion = 100,
#                     unit = r'%',
#                     color = 'black',
#                     location = 'top left',
#                     linewidth = 5)
# 
# utils_plot.add_scalebar_to_fig(ax_grand_average_normalized,
#                     axis = 'x',
#                     size = 10,
#                     conversion = 1,
#                     unit = r'ms',
#                     color = 'black',
#                     location = 'top left',
#                     linewidth = 5)
# figfile = os.path.join(figures_dir,'fig_average_onset_pyr.{}')
# fig_grand_average_traces_pyr_highres.savefig(figfile.format('svg'))
# inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
# subprocess.run(inkscape_cmd )
# =============================================================================
#%%
#%% download sensor kinetics for 2 AP
wave_parameters = {'ap_num_to_plot':2,#ap_num_to_plot
                   'min_ap_per_apnum':1} # filters both grand average and cell-wise traces
ca_traces_dict_2ap_pyr,ca_traces_dict_by_cell_2ap_pyr  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,
                                                                                           ca_waves_dict_pyr,
                                                                                           wave_parameters)

#%% 2 AP kinetics
plot_parameters={'trace_to_use':'dff',#'f_corr','dff','f_corr_normalized'
                 'trace_to_use_2':'dff',
                 'superresolution_function':'gauss',#'mean',#'median'#'gauss'
                 'normalize_to_max':True,
                 'linewidth':3
                 }

ca_traces_dict = ca_traces_dict_2ap_pyr
superresolution_traces_1ap = superresolution_traces_1ap_pyr_low_resolution
time_range_step = .005
time_range_window = .001
time_range_offset = .003
time_range_step_number = 9
bin_step = 1
bin_size = 8
fig_freq = plt.figure()

axs = list()

fig_merged = plt.figure(figsize = [5,10])
axs_merged=dict()
start_time = -25
end_time = 100
min_event_number = 20
exclude_outliers = False
sensors_to_plot = ['GCaMP7F','456','686','688']
slow_sensors = ['GCaMP7F']#'XCaMPgf'
axs_merged['slow_sensors']=fig_merged.add_subplot(2*len(sensors_to_plot)+1,1,1)

for sensor in slow_sensors:
    x = superresolution_traces_1ap[sensor]['{}_time'.format(plot_parameters['trace_to_use'])]
    y = superresolution_traces_1ap[sensor]['{}_{}'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
    if plot_parameters['normalize_to_max']:
        y = y/np.nanmax(y)
    color = sensor_colors[sensor]
    axs_merged['slow_sensors'].plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.9)
axs_merged['slow_sensors'].set_xlim([start_time,end_time])
axs_merged['slow_sensors'].plot(0,-.1,'r|',markersize = 18)
master_ax = axs_merged['slow_sensors']

for sensor_i,sensor in enumerate(sensors_to_plot):
    cawave_ap_group_lengts = np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])
    sensor_axs = list()
    axs_merged[sensor]=dict()
# =============================================================================
#     if sensor_i ==0:
#         #axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1)
#         axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(sensors_to_plot)+1,1,sensor_i*2+1+1)
#         
#     else:
#         #axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1,sharex = master_ax)
# =============================================================================
    axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(sensors_to_plot)+1,1,sensor_i*2+1+1,sharex = master_ax,sharey = master_ax)
    #axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1+len(sensors_to_plot),sharex = master_ax)
    axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2*len(sensors_to_plot)+1,1,sensor_i*2+2+1,sharex = master_ax)
    axs_merged[sensor]['ax_hist'].axis('off')
    axs_merged[sensor]['ax_wave'].axis('off')
    axs_merged[sensor]['ax_wave'].plot(0,-.1,'r|',markersize = 18)
    for step_i in np.arange(time_range_step_number)[::-1]:
        #try:
        if exclude_outliers:
            outlier_array = np.asarray(ca_traces_dict[sensor]['cawave_snr_list'])
            percentiles = np.percentile(outlier_array,[25,75])
            non_outliers = (outlier_array>=percentiles[0]) & (outlier_array<=percentiles[1]+np.abs(np.diff(percentiles)))#-np.abs(np.diff(percentiles))

        else:
            non_outliers = cawave_ap_group_lengts>-1
        cawave_idx = [False]
        increase_high_step = -1
        while sum(cawave_idx)<min_event_number and increase_high_step <= time_range_step*1000:
            increase_high_step+=1
            low = time_range_step*step_i + time_range_offset
            high = time_range_step*step_i + time_range_window+.001*increase_high_step + time_range_offset
            cawave_idx = (cawave_ap_group_lengts>=low)&(cawave_ap_group_lengts<high)&non_outliers
        if sum(cawave_idx)<min_event_number:
            continue
        x_conc = np.concatenate(np.asarray(ca_traces_dict[sensor]['cawave_time_list'])[cawave_idx])
        y_conc = np.concatenate(np.asarray(ca_traces_dict[sensor]['cawave_f_corr_normalized_list'])[cawave_idx])#cawave_dff_list#cawave_f_corr_list#cawave_f_corr_normalized_list
        needed_idx = (x_conc>=start_time-bin_size)&(x_conc<=end_time+bin_size)
        x_conc = x_conc[needed_idx]
        y_conc = y_conc[needed_idx]
        trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,bin_step,bin_size,function = 'all') 
        bin_centers = np.asarray(trace_dict['bin_centers'])
        trace = np.asarray(trace_dict['trace_{}'.format(plot_parameters['superresolution_function'])])
        trace_sd = np.asarray(trace_dict['trace_std'])
        if len(axs)==0:
        
            ax = fig_freq.add_subplot(time_range_step_number,len(sensors_to_plot),sensor_i+len(sensors_to_plot)*step_i+1)
        else:
            if len(sensor_axs) == 0:
                ax = fig_freq.add_subplot(time_range_step_number,len(sensors_to_plot),sensor_i+len(sensors_to_plot)*step_i+1,sharex = axs[0])
            else:
                ax = fig_freq.add_subplot(time_range_step_number,len(sensors_to_plot),sensor_i+len(sensors_to_plot)*step_i+1,sharex = axs[0],sharey = sensor_axs[0])
        ax_hist = ax.twinx()
        ax_hist.hist(np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])[cawave_idx]*1000,int(time_range_step*1000),alpha = .5)
        ax_hist.set_navigate(False)
        #break
        axs.append(ax)
        sensor_axs.append(ax)
        for x, y in zip(np.asarray(ca_traces_dict[sensor]['cawave_time_list'])[cawave_idx],np.asarray(ca_traces_dict[sensor]['cawave_f_corr_normalized_list'])[cawave_idx]):#cawave_dff_list#cawave_f_corr_list#cawave_f_corr_normalized_list
            needed_idx = (x>=start_time)&(x<=end_time)
            ax.plot(x[needed_idx],y[needed_idx],color =sensor_colors[sensor],alpha = .1 )
        ax.plot(bin_centers,trace,color =sensor_colors[sensor],alpha = 1,linewidth =2 )
        
        ax.fill_between(bin_centers, trace-trace_sd, trace+trace_sd,color =sensor_colors[sensor],alpha = .3)
        if sensor_i == len(sensors_to_plot)-1:
            ax_hist.set_ylabel('{} - {} ms'.format(int(low*1000),int(high*1000)))
            
        #alpha =  1-step_i*.1 #.1+step_i*.1
        color =utils_plot.lighten_color(sensor_colors[sensor], amount=1-step_i*.1)
        axs_merged[sensor]['ax_hist'].hist(np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])[cawave_idx]*1000,int(time_range_step*1000),color =color)
        axs_merged[sensor]['ax_wave'].plot(bin_centers,trace,color =color,linewidth =2 )#
ax.set_xlim([start_time,end_time])  
axs_merged['slow_sensors'].set_ylim([-.1,1.1])  
axs_merged[sensor]['ax_wave'].set_xlim([start_time,end_time])  
utils_plot.add_scalebar_to_fig(axs_merged['slow_sensors'],
                                axis = 'x',
                                size = 10,
                                conversion = 1,
                                unit = r'ms',
                                color = 'black',
                                location = 'bottom right',
                                linewidth = 5)

figfile = os.path.join(figures_dir,'fig_2ap_kinetics.{}')
fig_merged.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )



#%% Select events for putative pyramidal cells - multiple AP
leave_out_putative_interneurons = True
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':.5,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':1,
                      'min_post_isi':2,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':10,#0,5,10,20,50
                      'neuropil_subtraction':'0.8',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
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
#%%
plot_parameters = {'min_ap_per_apnum': 1,
                   'min_cell_num':1,
                   'sensor_colors':sensor_colors}
fig_dff_ap_dependence = plt.figure(figsize = [10,20])
ax_amplitude = fig_dff_ap_dependence.add_subplot(3,1,1)
#ax_amplitude_swarm = ax_amplitude.twinx()
ax_amplitude_norm = fig_dff_ap_dependence.add_subplot(3,1,2,sharex = ax_amplitude)
ax_cellnum = fig_dff_ap_dependence.add_subplot(3,1,3,sharex = ax_amplitude)
#ax_amplitude_norm_swarm = ax_amplitude_norm.twinx()
# =============================================================================
# apnum_concat = list()
# y_concat = lis()
# y_norm_concat = list()
# color = list()
# =============================================================================
for sensor in ca_wave_parameters['sensors']:
    cellwave_by_cell_list = ca_waves_dict_pyr_decay_time[sensor]
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
    #%
ax_amplitude_norm.plot([0,7],[0,7],'k--')
ax_cellnum.set_xlim([0.5,8.5])
ax_cellnum.set_xlabel('Number of spikes') 
ax_cellnum.set_ylabel('Number of cells in dataset') 
#ax_amplitude_norm.set_xlabel('Number of spikes') 
ax_amplitude_norm.set_ylabel('Amplitude normalized to single spike event')
#ax_amplitude.set_xlabel('Number of spikes') 
ax_amplitude.set_ylabel('Amplitude (dF/F)')
ax_amplitude.set_ylim([0,5])
ax_amplitude_norm.set_ylim([0,20])
#%
figfile = os.path.join(figures_dir,'fig_multiple_ap_scatter.{}')
fig_dff_ap_dependence.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )



#%% 
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
                      'minimum_f0_value':20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
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
                     'dogaussian':False
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
#%%#######################################################################################################

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
                      'minimum_f0_value':20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
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

#%% grand average
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
                      'minimum_f0_value':20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
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
movie_length = 40 #s
zoom_length = 1.5#2
sweeps = {'456':[{'subject_id':471993,'session':1,'cell_number':2,'sweep_number':2,'start_time':20,'zoom_start_time':44.5}],
          '686':[{'subject_id':479117,'session':1,'cell_number':8,'sweep_number':4,'start_time':110,'zoom_start_time':140}],#132
          '688':[{'subject_id':472181,'session':1,'cell_number':1,'sweep_number':3,'start_time':16,'zoom_start_time':22.4}]
          }

movie_length = 80 #s
zoom_length = 10#2
sweeps = {'456':[{'subject_id':478407,'session':1,'cell_number':7,'sweep_number':4,'start_time':35,'zoom_start_time':44}],
          '686':[{'subject_id':472180,'session':1,'cell_number':1,'sweep_number':1,'start_time':50,'zoom_start_time':56}],#132
          '688':[{'subject_id':479119,'session':1,'cell_number':9,'sweep_number':2,'start_time':25,'zoom_start_time':69}]
          }

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
rownum = 3
fig_longtrace = plt.figure(figsize = [10,10])
fig_zoom = plt.figure(figsize = [2,10])
axs_longtrace =list()
axs_zoomtrace = list()
#%
for sensor in sweeps.keys():
    sweeps_now =sweeps[sensor]
    for sweep in sweeps_now:
        if len(axs_longtrace)==0:
            axs_longtrace.append(fig_longtrace.add_subplot(rownum,1,len(axs_longtrace)+1))
        else:
            axs_longtrace.append(fig_longtrace.add_subplot(rownum,1,len(axs_longtrace)+1,sharey = axs_longtrace[0]))
        if len(axs_zoomtrace)==0:
            axs_zoomtrace.append(fig_zoom.add_subplot(rownum,1,len(axs_zoomtrace)+1))
        else:
            axs_zoomtrace.append(fig_zoom.add_subplot(rownum,1,len(axs_zoomtrace)+1,sharey = axs_zoomtrace[0]))
            
        start_time = sweep.pop('start_time',None)
        zoom_start_time = sweep.pop('zoom_start_time',None)
        table = ephys_cell_attached.Sweep()*imaging.ROI()*ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepStimulus()*ephys_cell_attached.SweepResponse()*imaging_gt.ROISweepCorrespondance()*imaging.ROITrace()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()*ephys_cell_attached.Cell()*experiment.Session()&sweep&'channel_number = 1'&'neuropil_number = 1'
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
                               location = 'bottom right',
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