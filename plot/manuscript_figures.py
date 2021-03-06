import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
#%%
#%%#######################################################################ALL RECORDING STATS - START################################################################################
#################################################################################%%############################################################################################
leave_out_putative_interneurons = False
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':2,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':.5,
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
ca_waves_dict_all = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,
                                                          cawave_properties_needed,
                                                          ephys_properties_needed)



#%% neuropil distribution
from plot import manuscript_figures_core
plot_properties = {'fig_name':'neuropil_distribution',
                   'figures_dir':figures_dir}
manuscript_figures_core.plot_r_neu_distribution(sensor_colors,ca_wave_parameters,plot_properties)
#%% neuropil correction example
from plot import manuscript_figures_core
from utils.utils_plot import gaussFilter
import scipy.ndimage as ndimage
from skimage.measure import label
# =============================================================================
# plot_parameters = {'alpha_roi':.5,
#                    'cax_limits' : [2,99.9],
#                    'mean_image_micron_size': [50,100],
#                    'xlim':[65,100],
#                    'xlim_zoom':[71,85],
#                    'figsize':[20,20]}
# key =  {'subject_id': 471991, 
#         'session': 1, 
#         'cell_number': 1, 
#         'sweep_number': 0, 
#         'movie_number': 0, 
#         'motion_correction_method': 'Suite2P', 
#         'roi_type': 'Suite2P', 
#         'roi_number': 3, 
#         'channel_number': 1, 
#         'neuropil_number': 1}
# =============================================================================


plot_parameters = {'alpha_roi':.5,
                   'cax_limits' : [1,98],
                   'mean_image_micron_size': [50,100],
                   'xlim':[0,120],
                   'xlim_zoom':[40,80],
                   'figsize':[14,14]}
key =  {'subject_id': 471991, 
        'session': 1, 
        'cell_number': 4, 
        'sweep_number': 2, 
        'movie_number': 14, 
        'motion_correction_method': 'Suite2P', 
        'roi_type': 'Suite2P', 
        'roi_number': 5, 
        'channel_number': 1, 
        'neuropil_number': 1}


#motion_corr_x_offset,motion_corr_y_offset=(imaging.MotionCorrection()&key&'motion_corr_description = "rigid registration"').fetch1('motion_corr_x_offset','motion_corr_y_offset')

neuropil_estimation_decay_time = 3 # - after each AP this interval is skipped for estimating the contribution of the neuropil
neuropil_estimation_filter_sigma = 0.05#0.05
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
if neuropil_estimation_filter_sigma>0:
    F_activitiy_conv= np.convolve(F_activitiy_conv[::-1],np.concatenate([np.zeros(int((neuropil_estimation_filter_sigma*framerate*5))),np.ones(int((neuropil_estimation_filter_sigma*framerate*5)))]),'same')[::-1]
F_activitiy_conv[:decay_step]=1
needed =F_activitiy_conv==0

if neuropil_estimation_filter_sigma>0:
    F_orig_filt = gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma)
else:
    F_orig_filt = F.copy()
F_orig_filt_ultra_low =  ndimage.minimum_filter(F_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))
F_orig_filt_highpass = F_orig_filt-F_orig_filt_ultra_low + F_orig_filt_ultra_low[0]
if neuropil_estimation_filter_sigma>0:
    Fneu_orig_filt = gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma)
else:
    Fneu_orig_filt  = Fneu.copy()
Fneu_orig_filt_ultra_low =  ndimage.minimum_filter(Fneu_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))
Fneu_orig_filt_highpass = Fneu_orig_filt-Fneu_orig_filt_ultra_low + Fneu_orig_filt_ultra_low[0]
p=np.polyfit(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],1)
R_neuopil_fit = np.corrcoef(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed])
r_neu_now = p[0]
key['r_neu'] = r_neu_now
key['r_neu_corrcoeff'] = R_neuopil_fit[0][1]
key['r_neu_pixelnum'] = sum(needed)

#print(key)
#%

fig = plt.figure(figsize = plot_parameters['figsize'])
gs = GridSpec(4, 2, figure=fig)
ax1 = fig.add_subplot(gs[0,:])
ax2 = fig.add_subplot(gs[2,0])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[3,:])
ax5 = fig.add_subplot(gs[1:-1,1])
apmaxtimes = apmaxtimes - frame_times[0]
frame_times = frame_times - frame_times[0]
needed_time = (frame_times>plot_parameters['xlim'][0]) & (frame_times<plot_parameters['xlim'][1])
needed_time_zoom = (frame_times>plot_parameters['xlim_zoom'][0]) & (frame_times<plot_parameters['xlim_zoom'][1])
ax1.plot(frame_times[needed_time],F_orig_filt_highpass[needed_time],'g-')
ax1.plot(frame_times[needed_time],Fneu_orig_filt_highpass[needed_time],'b-')

neededidxs = label(needed)
for needed_num in np.unique(neededidxs[neededidxs>0]):
    idx_now = (neededidxs==needed_num) & needed_time
    ax1.plot(frame_times[idx_now],F_orig_filt_highpass[idx_now],'r-',linewidth=8,alpha = .4)
    ax1.plot(frame_times[idx_now],Fneu_orig_filt_highpass[idx_now],'r-',linewidth = 8, alpha = .4)
#plt.tight_layout()
ylim = ax1.get_ylim()
ax1.plot(apmaxtimes,np.zeros(len(apmaxtimes))+ylim[0],'r|',ms = 22,linewidth = 3)
f_corrected = F_orig_filt_highpass-r_neu_now*Fneu_orig_filt_highpass
ax2.plot(frame_times[needed_time_zoom],F_orig_filt_highpass[needed_time_zoom]-np.median(F_orig_filt_highpass),'g-')
ax2.plot(frame_times[needed_time_zoom],f_corrected[needed_time_zoom]-np.median(f_corrected),'k-')
#plt.autoscale(enable=True, axis='xy', tight=True)
ylim = ax2.get_ylim()
ax2.plot(apmaxtimes,np.zeros(len(apmaxtimes))+ylim[0],'r|',ms=22,linewidth = 3)
ax3.plot(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],'ro',ms = 1, alpha = .1)
xvals = [np.min(Fneu_orig_filt_highpass[needed]),np.max(Fneu_orig_filt_highpass[needed])]
yvals = np.polyval(p,xvals)
ax3.plot(xvals,yvals,'k-',linewidth = 4)
ax3.set_title('F_roi = {:.2f} * F_neu + {:.2f}'.format(p[0],p[1]))
#ax3.plot(Fneu_orig_filt[needed],F_orig_filt[needed],'ro')
            #%



mean_image,movie_x_size,movie_y_size,movie_pixel_size,roi_xpix,roi_ypix,neuropil_xpix,neuropil_ypix= (imaging.ROINeuropil()*imaging.ROI()*imaging.Movie()*imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()&key&'channel_number = 1').fetch1('registered_movie_mean_image','movie_x_size','movie_y_size','movie_pixel_size','roi_xpix','roi_ypix','neuropil_xpix','neuropil_ypix')
movie_pixel_size = float(movie_pixel_size)
#mean_image_red = (imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()&key&'channel_number = 2').fetch1('registered_movie_mean_image')

im_green = ax4.imshow(mean_image,cmap = 'gray', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)])#, aspect='auto')
clim = np.percentile(mean_image.flatten(),plot_parameters['cax_limits'])
im_green.set_clim(clim)
ROI = np.zeros_like(mean_image)*np.nan
ROI[roi_ypix,roi_xpix] = 1
neuropil = np.zeros_like(mean_image)*np.nan
neuropil[neuropil_ypix,neuropil_xpix] = 1
ax4.imshow(ROI,cmap = 'Greens', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)],alpha = plot_parameters['alpha_roi'],vmin = 0, vmax = 1)#, aspect='auto'
ax4.imshow(neuropil,cmap = 'Blues', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)],alpha = plot_parameters['alpha_roi'],vmin = 0, vmax = 1)#, aspect='auto'
roi_center_x =np.mean(roi_xpix*float(movie_pixel_size))
roi_center_y =float(movie_pixel_size)*movie_y_size-np.mean(roi_ypix*float(movie_pixel_size))
image_edges_y = [np.max([0,roi_center_y-plot_parameters['mean_image_micron_size'][0]/2]),np.min([movie_y_size*float(movie_pixel_size),roi_center_y+plot_parameters['mean_image_micron_size'][0]/2])]
image_edges_x = [np.max([0,roi_center_x-plot_parameters['mean_image_micron_size'][1]/2]),np.min([movie_x_size*float(movie_pixel_size),roi_center_x+plot_parameters['mean_image_micron_size'][1]/2])]
ax4.set_xlim(image_edges_x)
ax4.set_ylim(image_edges_y)
ax1.set_xlim(plot_parameters['xlim'])
ax2.set_xlim(plot_parameters['xlim_zoom'])
ax1.autoscale(enable = True,axis = 'y',tight = True)
ax2.autoscale(enable = True,axis = 'y',tight = True)

ax1.set_xlabel('Time (s)')
ax2.set_xlabel('Time (s)')
ax1.set_ylabel('Pixel intensity')
ax2.set_ylabel('Pixel intensity')

ax3.set_xlabel('Neuropil pixel intensity')
ax3.set_ylabel('ROI pixel intensity')


plot_properties = {'fig_name':'neuropil_distribution',
                   'figures_dir':figures_dir,
                   'ax':ax5}
manuscript_figures_core.plot_r_neu_distribution(sensor_colors,ca_wave_parameters,plot_properties)

#%
filename = 'rneu_calculation_example'
figfile = os.path.join(figures_dir,filename+'.{}')
fig.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )

#%% number of ROIs in each movie
import plot.imaging_gt_plot as imaging_plot
imaging_plot.plot_number_of_rois(sensor_colors)
#%%
#%% stats for all experiments
sensor_names = {'GCaMP7F': 'jGCaMP7f',
                'XCaMPgf': 'XCaMPgf',
                '456': 'jGCaMP8f',
                '686': 'jGCaMP8m',
                '688': 'jGCaMP8s'}
plot_properties = {'sensors':['GCaMP7F','XCaMPgf','456','686','688'],
                   'sensor_colors':sensor_colors,
                   'max_expression_time':170,
                   'fig_name':'in_vivo_summary_stats',
                   'figures_dir':figures_dir,
                   'sensor_names':sensor_names}
inclusion_criteria = {'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'minimum_f0_value':20, # when the neuropil is bright, the F0 can go negative after neuropil subtraction
                      'min_channel_offset':100 # PMT error
                      }
#%
manuscript_figures_core.plot_stats(plot_properties,inclusion_criteria)
#%%ap parameters scatter plot, calcium sensor, median ap wave amplitude
from plot import manuscript_figures_core
plot_parameters = {'sensors':['GCaMP7F','XCaMPgf','456','686','688'],
                   'sensor_colors':sensor_colors,
                   'ap_num_to_plot':1,
                   'x_parameter':'ap_peak_to_trough_time_median',#'cell_mean_firing_rate',#'ap_peak_to_trough_time_median' # not selected by AP number
                   'y_parameter':'ap_ahp_amplitude_median',# not selected by AP number
                   'z_parameter':'cell_mean_firing_rate',#'cawave_peak_amplitude_dff',#'cawave_snr_median_per_ap',#'roi_f0_percentile_median_per_ap',#'cawave_snr_median_per_ap',#
                   'z_parameter_unit': 'Hz',
                   'invert y axis':True,
                   'alpha' : .5,
                   'markersize_edges' : [2,20],
                   'z_parameter_percentiles':[5,95],
                   'normalize z by sensor':False,
                   'xlim':[0.2,2.5],
                   'fig_name':'AP_parameters_interneuron_selection',
                   'figures_dir':figures_dir,
                   'xlabel':'Peak to trough time (ms)',
                   'ylabel':'Peak to trough ratio'}#'cawave_peak_amplitude_dff_median_per_ap'
manuscript_figures_core.plot_ap_scatter_with_third_parameter(ca_waves_dict,plot_parameters)

#%%
#%% figure explaining superresolution analysis



#key_first_ap = { 'session': 1, 'sweep_number': 2, 'ap_group_number': 112, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': 'calculated', 'neuropil_number': 1}        
# =============================================================================
# first_subject_id = 472181
# first_cell_number= 7
# first_ap_group_number = 112
# first_sweep_number = 2
# =============================================================================
key_first_ap = {}
key_all_ap = {'session': 1, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': '0.8', 'neuropil_number': 1,'ap_group_ap_num':1}        
snr_cond = 'ap_group_min_snr_dv>{}'.format(20)
preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
basline_dff_cond = 'cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
basline_dff_cond_2 = 'cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
cawave_snr_cond = 'cawave_snr > {}'.format(5)
sensor_cond = 'session_calcium_sensor = "{}"'.format('688')
big_table = ephysanal_cell_attached.APGroup()*imaging_gt.SessionCalciumSensor()*imaging_gt.CalciumWave.CalciumWaveProperties()*ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.APGroupTrace()*imaging_gt.CalciumWave.CalciumWaveNeuropil()*imaging_gt.CalciumWave()#imaging_gt.SessionCalciumSensor()*imaging_gt.CalciumWave.CalciumWaveProperties()*ephysanal_cell_attached.APGroup()
subject_ids,cell_numbers,ap_group_numbers,sweep_numbers,cawave_snr = (big_table&key_all_ap& snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 & cawave_snr_cond&sensor_cond).fetch('subject_id','cell_number','ap_group_number','sweep_number','cawave_snr')
# =============================================================================
# ap_group_numbers = np.concatenate([[first_ap_group_number],ap_group_numbers])
# sweep_numbers = np.concatenate([[first_sweep_number],sweep_numbers])
# subject_ids = np.concatenate([[first_subject_id],subject_ids])
# cell_numbers = np.concatenate([[first_cell_number],cell_numbers])
# =============================================================================
#%


alpha = .1
number_of_transients = 250
fig = plt.figure(figsize = [10,10])
ax_dff=fig.add_subplot(4,1,3)
ax_event_times = fig.add_subplot(4,1,2,sharex = ax_dff)
plt.axis('off')
ax_ephys = fig.add_subplot(4,1,1,sharex = ax_dff)
ax_superresolution = fig.add_subplot(4,1,4,sharex = ax_dff)
cawave_dff_list = list()
cawave_time_list = list()
x_limits = [-25,50]
for wave_idx,(sweep_number,ap_group_number,subject_id,cell_number) in enumerate(zip(sweep_numbers,ap_group_numbers,subject_ids,cell_numbers)):
    
    key_first_ap['sweep_number'] = sweep_number
    key_first_ap['ap_group_number'] = ap_group_number
    key_first_ap['subject_id'] = subject_id
    key_first_ap['cell_number'] = cell_number
    key_first_ap['neuropil_subtraction'] = '0.8'
    key_first_ap['gaussian_filter_sigma'] = 0
    key_first_ap['neuropil_number'] = 1
    #try:
    cawave_f,cawave_fneu,cawave_time,ap_group_trace,sample_rate,ap_group_trace_peak_idx,cawave_rneu,cawave_baseline_f = (big_table&key_first_ap).fetch1('cawave_f','cawave_fneu','cawave_time','ap_group_trace','sample_rate','ap_group_trace_peak_idx','cawave_rneu','cawave_baseline_f')
    #cawave_rneu,cawave_baseline_f = (imaging_gt.CalciumWave.CalciumWaveProperties()&key).fetch1('cawave_rneu','cawave_baseline_f')
    cawave_f_corr = cawave_f-cawave_fneu*cawave_rneu
    cawave_dff = (cawave_f_corr-cawave_baseline_f)/cawave_baseline_f 
    ap_group_trace_time = np.arange(-ap_group_trace_peak_idx,len(ap_group_trace)-ap_group_trace_peak_idx)/sample_rate*1000
    cawave_idx = (cawave_time>=x_limits[0]-10) & (cawave_time<=x_limits[1]+10)
    ephys_idx = (ap_group_trace_time>=x_limits[0]-10) & (ap_group_trace_time<=x_limits[1]+10)
    ax_dff.plot(cawave_time[cawave_idx],cawave_dff[cawave_idx],'go-',alpha = alpha)
    ax_ephys.plot(ap_group_trace_time[ephys_idx][::20],ap_group_trace[ephys_idx][::20],'k-',alpha = alpha)
    ax_event_times.plot(cawave_time[cawave_idx],np.zeros(len(cawave_time[cawave_idx]))+wave_idx,'go',alpha = alpha)
    cawave_dff_list.append(cawave_dff)
    cawave_time_list.append(cawave_time)
    if wave_idx == number_of_transients-1:
        break
# =============================================================================
#     except:
#         pass
# =============================================================================
ax_event_times.set_ylim([-10,510])

#ax_ephys.set_xlim([-25,50])
ax_ephys.set_ylim([-.5,1.5])

ax_ephys.set_ylabel('Ephys normalized')
ax_dff.set_ylabel('dF/F')
ax_dff.set_ylim([-.25,1])
#%
#if plot_superresolution_instead_of_ephys:
x_conc = np.concatenate(cawave_time_list)
y_conc = np.concatenate(cawave_dff_list)
needed_idx = (x_conc>x_limits[0]-10)& (x_conc<x_limits[1]+10)
x_conc = x_conc[needed_idx]
y_conc = y_conc[needed_idx]
trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,bin_step=.5,bin_size=5,function = 'all',dogaussian = True)

ax_superresolution.plot(trace_dict['bin_centers'],trace_dict['trace_gauss'],'go-')#trace_mean
ax_superresolution.set_ylabel('dF/F')
#ax_ephys.set_ylim([-.5,2])
ax_superresolution.set_xlim(x_limits)
ax_superresolution.set_xlabel('Time from AP peak (ms)')
filename = 'superresolution_{}_events'.format(number_of_transients)
figfile = os.path.join(figures_dir,filename+'.{}')
fig.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )


# =============================================================================
# #%%
# import scipy.stats as stats
# from scipy import signal
# bin_size = 5
# bin_step=1
# threshold = .1
# from utils import utils_plot
# trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,bin_step=bin_step,bin_size=bin_size,function = 'all',dogaussian = True)
# 
# idxes_gauss = np.arange(-bin_size,bin_size,bin_step)
# gauss_f = stats.norm(0,bin_size/4)
# gauss_w = gauss_f.pdf(idxes_gauss)
# gauss_w = gauss_w/np.max(gauss_w)
# gauss_w = gauss_w[gauss_w >threshold]
# #%
# out = signal.deconvolve(trace_dict['trace_gauss'], gauss_w)
# 
# fig  = plt.figure()
# ax = fig.add_subplot(212)
# ax.plot(out[0]/np.max(out[0]))
# ax.plot(trace_dict['trace_gauss']/np.max(trace_dict['trace_gauss']))
# ax1 = fig.add_subplot(211)
# ax1.plot(gauss_w)
# 
# 
# #%%
# bin_size = 10
# bin_step=.1
# slope = 0.015
# from utils import utils_plot
# trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,bin_step=bin_step,bin_size=bin_size,function = 'all',dogaussian = True)
# 
# idxes_gauss = np.arange(-bin_size,bin_size,bin_step)
# gauss_f = stats.norm(0,bin_size/4)
# gauss_w = gauss_f.pdf(idxes_gauss)
# gauss_w = gauss_w/np.max(gauss_w)
# 
# ap_idx = np.argmax(trace_dict['bin_centers']>0)
# realtransient = np.zeros(len(trace_dict['trace_gauss']))
# realtransient[ap_idx:] = np.arange(len(trace_dict['trace_gauss'])-ap_idx)*slope
# realtransient[realtransient>1]=1
# transient_convolved = np.convolve(realtransient,gauss_w,mode = 'same')
# 
# fig  = plt.figure()
# ax = fig.add_subplot(212)
# ax.plot(trace_dict['bin_centers'],transient_convolved/np.max(transient_convolved))
# ax.plot(trace_dict['bin_centers'],realtransient)
# ax.plot(trace_dict['bin_centers'],trace_dict['trace_gauss']/np.max(trace_dict['trace_gauss'])/.9)
# ax1 = fig.add_subplot(211)
# ax1.plot(gauss_w)
# =============================================================================
#%%#######################################################################ALL RECORDING STATS - END################################################################################
#################################################################################%%############################################################################################

#%%#######################################################################EPHYS PLOTS - START################################################################################
#################################################################################%%############################################################################################
inclusion_criteria = {'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      }
ap_data_to_fetch = ['subject_id',
                    'session',
                    'cell_number',
                    'sweep_number',
                    'ap_halfwidth',
                    'recording_mode',
                    'ap_amplitude',
                    'ap_max_index',
                    'ap_full_width',
                    'ap_rise_time',
                    'ap_integral',
                    'ap_snr_v',
                    'ap_snr_dv']#%subject_id,session,cell_number,sweep_number,ap_halfwidth,recording_mode,ap_amplitude,ap_max_index
ephys_snr_cond = 'sweep_ap_snr_dv_median>={}'.format(inclusion_criteria['min_sweep_ap_snr_dv_median'])
ephys_hwstd_cond = 'sweep_ap_hw_std_per_median<={}'.format(inclusion_criteria['max_sweep_ap_hw_std_per_median'])
apdatalist = (ephysanal_cell_attached.SweepAPQC()*ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.ActionPotential()&ephys_snr_cond&ephys_hwstd_cond).fetch(*ap_data_to_fetch)
#%%
plot_parameters = {'alpha':.1,
                   'binnum':100,
                   'ap_baseline_length':1}#ms


example_ap_group_key_cc  = {'subject_id':471991,
                         'session':1,
                         'cell_number':6,
                         'sweep_number':1,
                         'ap_group_number':1}
example_ap_group_key_vc  = {'subject_id':471991,
                         'session':1,
                         'cell_number':1,
                         'sweep_number':0,
                         'ap_group_number':15}

trace_cc,ap_group_trace_peak_idx_cc,sample_rate_cc,multiplier_cc = (ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.APGroupTrace()&example_ap_group_key_cc).fetch1('ap_group_trace','ap_group_trace_peak_idx','sample_rate','ap_group_trace_multiplier')
trace_time_cc = (np.arange(len(trace_cc))-ap_group_trace_peak_idx_cc)/sample_rate_cc *1000
#trace_filt_cc = utils_ephys.gaussFilter(trace_cc,sample_rate_cc,sigma = .00005)  
needed_idx = (trace_time_cc>=-plot_parameters['ap_baseline_length']) & (trace_time_cc<=2*plot_parameters['ap_baseline_length'])
trace_time_cc = trace_time_cc[needed_idx]
trace_cc = trace_cc[needed_idx]*multiplier_cc

trace_vc,ap_group_trace_peak_idx_vc,sample_rate_vc,multiplier_vc = (ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.APGroupTrace()&example_ap_group_key_vc).fetch1('ap_group_trace','ap_group_trace_peak_idx','sample_rate','ap_group_trace_multiplier')
trace_time_vc = (np.arange(len(trace_vc))-ap_group_trace_peak_idx_vc)/sample_rate_vc *1000
#trace_filt_vc = utils_ephys.gaussFilter(trace_vc,sample_rate_vc,sigma = .00005)  
needed_idx = (trace_time_vc>=-plot_parameters['ap_baseline_length']) & (trace_time_vc<=2*plot_parameters['ap_baseline_length'])
trace_time_vc = trace_time_vc[needed_idx]
trace_vc = trace_vc[needed_idx]*multiplier_vc

ap_data = dict()
for data,data_name in zip(apdatalist,ap_data_to_fetch):
    ap_data[data_name] = data 
apnum = len(ap_data['subject_id'])
cell_ids = np.asarray(np.column_stack([ap_data['subject_id'],np.repeat(['_session_'], apnum)[:,None],ap_data['session'],np.repeat(['_cell_'], apnum)[:,None],ap_data['cell_number']]),str)
cell_IDs = list()
for cell_id in cell_ids:
    string = ''
    for _string in cell_id:
        string += _string
    cell_IDs.append(string)      
cell_IDs = np.asarray(cell_IDs)
ccidx = ap_data['recording_mode'] == 'current clamp'
vcidx = ap_data['recording_mode'] == 'voltage clamp'

ccidx_where = np.where(ccidx)[0]
vcidx_where = np.where(vcidx)[0]
#% cell by cell stats
uniquecells = np.unique(cell_IDs)
    #%
fig = plt.figure(figsize = [15,15])
ax_apwave_cc = fig.add_subplot(321)
ax_apwave_cc.plot(trace_time_cc,trace_cc,'k-')
ax_apwave_cc.set_xlim([-plot_parameters['ap_baseline_length'],plot_parameters['ap_baseline_length']*2])
ax_apwave_cc.set_title('Current clamp')
ax_apwave_cc.set_yticks([])
ax_apwave_cc.set_xlabel('ms')

ax_apwave_vc = fig.add_subplot(322)
ax_apwave_vc.plot(trace_time_vc,trace_vc,'k-')
ax_apwave_vc.set_xlim([-plot_parameters['ap_baseline_length'],plot_parameters['ap_baseline_length']*2])
ax_apwave_vc.set_title('Voltage clamp')
ax_apwave_vc.set_yticks([])
ax_apwave_cc.set_xlabel('ms')

ax_hw_amplitude_cc = fig.add_subplot(323)
ax_hw_amplitude_cc.set_xlabel('Halfwidth (ms)')
ax_hw_amplitude_cc.set_ylabel('Amplitude (mV)')
ax_hw_amplitude_vc = fig.add_subplot(324)
ax_hw_amplitude_vc.set_xlabel('Halfwidth (ms)')
ax_hw_amplitude_vc.set_ylabel('Amplitude (pA)')

logbins = np.logspace(np.log10(1),np.log10(1000),plot_parameters['binnum'])

ax_snr_distribution_cc = fig.add_subplot(325)
ax_snr_distribution_cc_cumsum = ax_snr_distribution_cc.twinx()
ax_snr_distribution_cc.hist(ap_data['ap_snr_v'][ccidx],logbins,color='black')
snrvals, binedges = np.histogram(ap_data['ap_snr_v'][ccidx],logbins)
bincenters = np.mean([binedges[:-1],binedges[1:]],0)
ax_snr_distribution_cc.set_xscale('log')
ax_snr_distribution_cc_cumsum.plot(bincenters,np.cumsum(snrvals)/sum(snrvals),'r-')
ax_snr_distribution_cc.set_xlim([4,1000])
ax_snr_distribution_cc.set_xlabel('SNR of APs')
ax_snr_distribution_cc.set_ylabel('AP num')
ax_snr_distribution_cc_cumsum.set_ylim([0,1])

logbins = np.logspace(np.log10(1),np.log10(1000),int(plot_parameters['binnum']))
ax_snr_distribution_vc = fig.add_subplot(326)
ax_snr_distribution_vc_cumsum = ax_snr_distribution_vc.twinx()
ax_snr_distribution_vc.hist(ap_data['ap_snr_v'][vcidx],logbins,color='black')
snrvals, binedges = np.histogram(ap_data['ap_snr_v'][vcidx],logbins)
bincenters = np.mean([binedges[:-1],binedges[1:]],0)
ax_snr_distribution_vc.set_xscale('log')
ax_snr_distribution_vc_cumsum.plot(bincenters,np.cumsum(snrvals)/sum(snrvals),'r-')
ax_snr_distribution_vc.set_xlim([4,1000])
ax_snr_distribution_vc.set_xlabel('SNR of APs')
ax_snr_distribution_vc.set_ylabel('AP num')
ax_snr_distribution_vc_cumsum.set_ylim([0,1])



for cellidx,uniquecell in enumerate(uniquecells):
    idxs = np.where((cell_IDs==uniquecell)&ccidx)[0]
    ax_hw_amplitude_cc.plot(ap_data['ap_halfwidth'][idxs],ap_data['ap_amplitude'][idxs]*1000,'o',ms=1, picker=0,alpha = plot_parameters['alpha'])
    idxs = np.where((cell_IDs==uniquecell)&vcidx)[0]
    ax_hw_amplitude_vc.plot(ap_data['ap_halfwidth'][idxs],ap_data['ap_amplitude'][idxs]*1e12,'o',ms=1, picker=0,alpha = plot_parameters['alpha'])
    
ax_hw_amplitude_cc.set_xlim([0,2])
ax_hw_amplitude_cc.set_ylim([0,20])
ax_hw_amplitude_vc.set_xlim([0,1])
ax_hw_amplitude_vc.set_ylim([0,900])
#%
filename = 'ephys_summary'
figfile = os.path.join(figures_dir,filename+'.{}')
fig.savefig(figfile.format('png'),dpi=600)
#%
ax_hw_amplitude_cc.cla()
ax_hw_amplitude_vc.cla()
fig.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd )
#%%#######################################################################EPHYS PLOTS - END################################################################################
#################################################################################%%############################################################################################

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
fig = plt.figure(figsize = [7,21])
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
    
    
    
    ax_dict[sensor].plot(x,y,'o',markersize = 3,alpha = .2,color = sensor_colors[sensor])
    ax_dict[sensor].plot(x_range+fit_parameters['rise_times'][sensor],yrange,'-',color = 'white',linewidth =6)#sensor_colors[sensor]
    ax_dict[sensor].axis('off')
    #
    
    #break
ax_dict[sensor].set_xlim([-50,600])    
ax_dict[sensor].set_ylim([-.5,1])    

# =============================================================================
# utils_plot.add_scalebar_to_fig(ax_dict[sensor],
#                                axis = 'x',
#                                size = 100,
#                                conversion = 1,
#                                unit = r'ms',
#                                color = 'black',
#                                location = 'top right',
#                                linewidth = 5)
# =============================================================================
    

# =============================================================================
# figfile = os.path.join(figures_dir,'fig_decay_fit.{}')
# fig.savefig(figfile.format('svg'))
# inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
# subprocess.run(inkscape_cmd )
# =============================================================================

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

#%
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
                     'dogaussian':False,
                     'double_bin_size_for_old_sensors':False
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
                 'start_time':-50, # ms
                 'end_time':300,#ms
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
#%%
import utils.utils_ephys as utils_ephys
#% Putative pyramidal cells

# putative interneurons
movie_length = 80 #s
zoom_length = 10#2
sweeps = {'456':[{'subject_id':478407,'session':1,'cell_number':7,'sweep_number':4,'start_time':35,'zoom_start_time':44}],
          '686':[{'subject_id':472180,'session':1,'cell_number':1,'sweep_number':1,'start_time':50,'zoom_start_time':56}],#132
          '688':[{'subject_id':479119,'session':1,'cell_number':9,'sweep_number':2,'start_time':25,'zoom_start_time':69}]
          }

plot_parameters = {'gaussian_filter_sigma':5,
                   'cawave_rneu' : .8,
                   'rownum' : 4,
                   'cax_limits' : [0,99],
                   'alpha_roi':.3,
                   'filename':'int',
                   'figures_dir':figures_dir,
                   'movie_length':movie_length,
                   'zoom_length':zoom_length,
                   'sweeps':sweeps,
                   'sensor_colors':sensor_colors
                   }

manuscript_figures_core.plot_sample_traces(plot_parameters)

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
                   'cawave_rneu' : .8,
                   'rownum' : 4,
                   'cax_limits' : [0,99],
                   'alpha_roi':.3,
                   'filename':'pyr',
                   'figures_dir':figures_dir,
                   'movie_length':movie_length,
                   'zoom_length':zoom_length,
                   'sweeps':sweeps,
                   'sensor_colors':sensor_colors
                   }

manuscript_figures_core.plot_sample_traces(plot_parameters)



#%%#######################################################################SAMPLE TRACES - END################################################################################
#################################################################################%%############################################################################################

#%%
#%%#######################################################################PROTEIN EXPRESSION - START################################################################################
#################################################################################%%############################################################################################

from utils import utils_anatomy

injection_site_dict = utils_anatomy.digest_anatomy_json_file()
sensor_f_corrs = utils_anatomy.export_in_vivo_brightnesses()
#%%
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
sensor_names = {'GCaMP7F': 'jGCaMP7f',
                'XCaMPgf': 'XCaMPgf',
                '456': 'jGCaMP8f',
                '686': 'jGCaMP8m',
                '688': 'jGCaMP8s'}
fig_expression = plt.figure(figsize = [15,10])
ax_rfp = fig_expression.add_subplot(2,2,1)
ax_in_vivo = fig_expression.add_subplot(2,2,2)
ax_in_vivo_gt = fig_expression.add_subplot(2,2,4)
ax_fitc = fig_expression.add_subplot(2,2,3)
sensor_num_list = list()
rfp_intensity_list = list()
fitc_intensity_list = list()
in_vivo_f0_list = list()
sensor_num_list_invivo = list()
closest_to_median_files = list()
sensor_num_list_gt = list()
in_vivo_gt_f0_list = list()
for sensor_i,sensor in enumerate(sensor_names.keys()):

    baselines_ground_truth = (imaging_gt.CellBaselineFluorescence()*imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)).fetch('cell_baseline_roi_f0_median')
    baselines_ground_truth = baselines_ground_truth[np.isnan(baselines_ground_truth) == False]
    idx = np.asarray(injection_site_dict['sensor'])==sensor
    rfp_values = np.asarray(injection_site_dict['RFP'])[idx]
    closest_to_median_files.append(np.asarray(injection_site_dict['filename'])[idx][np.argmin(np.abs(rfp_values-np.median(rfp_values)))])
    fitc_values = np.asarray(injection_site_dict['FITC'])[idx]
    sensor_nums = np.ones(sum(idx))*sensor_i
    sensor_num_list.append(sensor_nums)
    
    invivo_data = np.concatenate(sensor_f_corrs[sensor])
    invivo_data = invivo_data[invivo_data>0]
    
    if sensor_i == 0:
        rfp_baselne = np.median(rfp_values)
        fitc_baselne = np.median(fitc_values)
        invivo_baseline = np.median(invivo_data)
        invivo_gt_baseline = np.median(baselines_ground_truth)
    
    rfp_values = rfp_values/rfp_baselne
    fitc_values = fitc_values/fitc_baselne
    invivo_data = invivo_data/invivo_baseline
    baselines_ground_truth = baselines_ground_truth/invivo_gt_baseline
    
    rfp_intensity_list.append(rfp_values)
    fitc_intensity_list.append(fitc_values)
    in_vivo_f0_list.append(invivo_data)
    sensor_num_list_invivo.append(np.ones(len(invivo_data))*sensor_i)
    in_vivo_gt_f0_list.append(baselines_ground_truth)
    sensor_num_list_gt.append(np.ones(len(baselines_ground_truth))*sensor_i)
    
    ax_rfp.bar(sensor_i,np.median(rfp_values),edgecolor = sensor_colors[sensor],fill=False,linewidth = 4)
    ax_fitc.bar(sensor_i,np.median(fitc_values),edgecolor = sensor_colors[sensor],fill=False,linewidth = 4)
    ax_in_vivo_gt.bar(sensor_i,np.median(baselines_ground_truth),edgecolor = sensor_colors[sensor],fill=False,linewidth = 4)
    #ax_in_vivo.bar(sensor_i,np.mean(invivo_data),edgecolor = sensor_colors[sensor],fill=False,linewidth = 4)
    violin_parts  = ax_in_vivo.violinplot(invivo_data,positions = [sensor_i],widths = [.9],showmedians = True,showextrema = False)
    violin_parts['bodies'][0].set_facecolor(sensor_colors[sensor])
    violin_parts['bodies'][0].set_edgecolor(sensor_colors[sensor])
    violin_parts['cmedians'].set_color(sensor_colors[sensor])
ax_in_vivo.set_yscale('log')
ax_in_vivo_gt.set_yscale('log')

# =============================================================================
#     break
# =============================================================================
sns.swarmplot(x =np.concatenate(sensor_num_list),y = np.concatenate(rfp_intensity_list),color='black',ax = ax_rfp,alpha = .9,size = 8)     
sns.swarmplot(x =np.concatenate(sensor_num_list),y = np.concatenate(fitc_intensity_list),color='black',ax = ax_fitc,alpha = .9,size = 8)   
sns.swarmplot(x =np.concatenate(sensor_num_list_gt),y = np.concatenate(in_vivo_gt_f0_list),color='black',ax = ax_in_vivo_gt,alpha = .9,size = 4)   


sensor_names_list = []
for sensor in sensor_names: sensor_names_list.append(sensor_names[sensor])
ax_rfp.set_xticks(np.arange(len(sensors)))
ax_rfp.set_xticklabels(sensor_names_list, rotation = 20)  
ax_rfp.set_ylabel('Anti-GFP fluorescence')
ax_in_vivo.set_xticks(np.arange(len(sensors)))
ax_in_vivo.set_xticklabels(sensor_names_list, rotation = 20) 
ax_in_vivo.set_ylabel('In vivo baseline fluorescence')
ax_fitc.set_xticks(np.arange(len(sensors)))
ax_fitc.set_xticklabels(sensor_names_list, rotation = 20) 
ax_fitc.set_ylabel('Fixed native fluorescence')

ax_in_vivo_gt.set_xticks(np.arange(len(sensors)))
ax_in_vivo_gt.set_xticklabels(sensor_names_list, rotation = 20) 
ax_in_vivo_gt.set_ylabel('In vivo ground truth baseline fluorescence')

figfile = os.path.join(figures_dir,'fig_anatomy_expression_brightness.{}')
fig_expression.savefig(figfile.format('svg'))
inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
subprocess.run(inkscape_cmd)

#%%#######################################################################PROTEIN EXPRESSION - END################################################################################
#################################################################################%%############################################################################################