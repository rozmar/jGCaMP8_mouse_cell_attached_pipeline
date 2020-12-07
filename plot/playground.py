import numpy as np

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
        'size'   : 23}

matplotlib.rc('font', **font)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Tahoma']
%matplotlib qt

# =============================================================================
# #%% dff vs noise
# f0,dff,std = (imaging_gt.CalciumWave().CalciumWaveProperties()).fetch('cawave_baseline_f','cawave_peak_amplitude_dff','cawave_baseline_f_std')
# #%%
# plt.plot(f0,std/f0,'ko',markersize = 1,alpha = .1)
# =============================================================================

#%% injection number and titers
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
fig_injections = plt.figure(figsize = [15,15])
ax_inj = fig_injections.add_subplot(111)
for sensor in sensors:
    subjects = imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)
    for subject in subjects:
        titers,volumes,dilutions = (lab.Virus()*lab.Surgery.VirusInjection()&subject).fetch('titer','volume','dilution')
        titer = np.median(titers)/np.median(dilutions)
        total_volume = np.sum(volumes)
        ax_inj.semilogy(total_volume,titer,'o',color = sensor_colors[sensor],alpha = .5,markersize = 12)
        
ax_inj.set_xlabel('Total volume injected (nl)')      
ax_inj.set_ylabel('Titer')      


#%% ROI F0 percentiles for all sensors
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
channel = 1
fig_percentiles = plt.figure()
ax_perc = fig_percentiles.add_subplot(2,1,1)
ax_f0 = fig_percentiles.add_subplot(2,1,2)
binvals_perc = np.arange(0,100,1)
binvals_f = np.arange(-.05,1.5,.02)
for sensor in sensors:
    rois = imaging_gt.ROISweepCorrespondance()*imaging_gt.SessionROIFPercentile()*imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor) &'channel_number = {}'.format(channel)
    roi_f0_percentiles,roi_f0_values = rois.fetch('roi_f0_percentile','roi_f0_percentile_value')
    
    vals, binedges = np.histogram(roi_f0_percentiles,binvals_perc)
    bincenters = np.mean([binedges[:-1],binedges[1:]],0)
    ax_perc.plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor])
    
    vals, binedges = np.histogram(roi_f0_values,binvals_f)
    bincenters = np.mean([binedges[:-1],binedges[1:]],0)
    ax_f0.plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor])
   # perc
    #break
    
    



#%% ROI fluorescence for ALL ROIs
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
channel = 1
neuropil_number = 1
roi_name = 'roi_f_min'#'neuropil_f_min'#
correct_depth = False
f_corrs = list()
f_corrs_norm = list()
depths = list()
sensor_f_corrs = dict()
sensor_f_corrs_norm = dict()
sensor_neuropil_to_roi = dict()
sensor_expression_time = dict()
subject_median_roi_values = dict()
subject_expression_times = dict()
sensr_depths = dict()
for sensor in sensors:
    #sensor = '686'
    sensr_depths[sensor]=list()
    sensor_f_corrs[sensor]=list()
    sensor_f_corrs_norm[sensor]=list()
    subject_median_roi_values[sensor] = list()
    subject_expression_times[sensor] = list()
    sensor_neuropil_to_roi[sensor]=list()
    sensor_expression_time[sensor]=list()
    sessions = experiment.Session()*imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)
    for i,session in enumerate(sessions):
        rois = imaging.ROINeuropilTrace()*imaging_gt.SessionCalciumSensor*imaging_gt.MovieDepth()*imaging.MovieMetaData*imaging.MoviePowerPostObjective()*imaging.ROITrace()&session&'channel_number = {}'.format(channel) & 'neuropil_number = {}'.format(neuropil_number)
        f,power,depth,session_sensor_expression_time,roi_f_min,neuropil_f_min = rois.fetch(roi_name,'movie_power','movie_depth','session_sensor_expression_time','roi_f_min','neuropil_f_min')#
        power = np.asarray(power,float)
        f = np.asarray(f,float)
        depth = np.asarray(depth,float)
        depth[depth<0]=100
        roi_f_min = np.asarray(roi_f_min,float)
        neuropil_f_min = np.asarray(neuropil_f_min,float)
        #f = roi_f_min-0.8*neuropil_f_min#roi_f_min/neuropil_f_min# 
        if correct_depth:
            f_corr = f/(power**2)*40**2+.004*f*depth#*np.exp(depth/200)
            neuropil_f_min_corr = neuropil_f_min/(power**2)*40**2+.004*neuropil_f_min*depth
        else:
            f_corr = f/(power**2)*40**2
            neuropil_f_min_corr = neuropil_f_min/(power**2)*40**2#+.004*neuropil_f_min_corr*depth
        f_corr_norm = (f_corr-np.nanmedian(f_corr))/np.diff(np.percentile(f_corr,[25,75]))#np.nanstd(f_corr)#
        if min(depth)>0:
            sensr_depths[sensor].append(depth)
            depths.append(depth)
            f_corrs.append(f_corr)
            sensor_f_corrs[sensor].append(f_corr)
            f_corrs_norm.append(f_corr_norm)
            sensor_f_corrs_norm[sensor].append(f_corr_norm)
            subject_median_roi_values[sensor].append(np.nanmedian(f_corr))
            subject_expression_times[sensor].append(np.nanmedian(session_sensor_expression_time))
            sensor_neuropil_to_roi[sensor].append(neuropil_f_min_corr/f_corr)      
            sensor_expression_time[sensor].append(session_sensor_expression_time)
# =============================================================================
#         break
#     break
# =============================================================================
#%% see depth dependence
fig_depth = plt.figure()
ax_depth = fig_depth.add_subplot(2,1,1)
ax_depth_norm = fig_depth.add_subplot(2,1,2)
for sensor_i,sensor in enumerate(sensors):
    ax_depth.plot(np.concatenate(sensr_depths[sensor]),np.concatenate(sensor_f_corrs[sensor]),'o',color=sensor_colors[sensor],alpha = .1)
    ax_depth_norm.plot(np.concatenate(sensr_depths[sensor]),np.concatenate(sensor_f_corrs_norm[sensor]),'o',color=sensor_colors[sensor],alpha = .1)


d = np.concatenate(depths)
f = np.concatenate(f_corrs_norm)
needed = (np.isnan(d) | np.isnan(f)) == False
p_norm = np.polyfit(d[needed],f[needed],1)
x = np.arange(0,200,1)
ax_depth_norm.plot(x,np.polyval(p_norm,x),'r-')
ax_depth_norm.set_title(p_norm)

d = np.concatenate(depths)
f = np.concatenate(f_corrs)
needed = (np.isnan(d) | np.isnan(f) | (f<=0)) == False
d = d[needed]
f = f[needed]
p = np.polyfit(d,f,1)
x = np.arange(0,200,1)
ax_depth.plot(x,np.polyval(p,x),'r-')
ax_depth.set_title(p)
#%% relative neuropil to ROI fluorescence
colnum = 3
binnum = np.arange(0,2,.001)#2000
alpha = .5
fig_neuropil = plt.figure()
axs_neuropil = dict()
axs_neuropil['ax_neuropil_per_cell_all'] = fig_neuropil.add_subplot(len(sensors)+1,colnum,len(sensors)*colnum+1)
axs_neuropil['ax_neuropil_per_cell_all_concat'] = fig_neuropil.add_subplot(len(sensors)+1,colnum,len(sensors)*colnum+2)
axs_neuropil['exptime__per_cell_all'] = fig_neuropil.add_subplot(len(sensors)+1,colnum,len(sensors)*colnum+3)
for sensor_i,sensor in enumerate(sensors):
    axs_neuropil[sensor] = fig_neuropil.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+1)
    axs_neuropil['{}_concatenated'.format(sensor)] = fig_neuropil.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+2)
    axs_neuropil['exptime_{}'.format(sensor)] = fig_neuropil.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+3)
    for origvals in sensor_neuropil_to_roi[sensor]:
        origvals = origvals[np.isnan(origvals)==False]
        origvals = origvals[np.isinf(origvals)==False]
        vals, binedges = np.histogram(origvals,binnum)
        bincenters = np.mean([binedges[:-1],binedges[1:]],0)
        axs_neuropil[sensor].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
        axs_neuropil['ax_neuropil_per_cell_all'].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
    origvals = np.concatenate(sensor_neuropil_to_roi[sensor])
    origvals = origvals[np.isnan(origvals)==False]
    origvals = origvals[np.isinf(origvals)==False]
    vals, binedges = np.histogram(origvals,binnum)
    bincenters = np.mean([binedges[:-1],binedges[1:]],0)
    axs_neuropil['{}_concatenated'.format(sensor)].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
    axs_neuropil['ax_neuropil_per_cell_all_concat'].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
    
    axs_neuropil['exptime_{}'.format(sensor)].loglog(np.concatenate(sensor_expression_time[sensor]),np.concatenate(sensor_neuropil_to_roi[sensor]),'o',color = sensor_colors[sensor], alpha = alpha,markersize = 1)
    axs_neuropil['exptime__per_cell_all'].loglog(np.concatenate(sensor_expression_time[sensor]),np.concatenate(sensor_neuropil_to_roi[sensor]),'o',color = sensor_colors[sensor], alpha = alpha,markersize = 1)
    #ax_neuropil.semilogy(np.concatenate(sensor_expression_time[sensor]),np.concatenate(sensor_neuropil_to_roi[sensor]),'o',color=sensor_colors[sensor],alpha = .1)
    
    axs_neuropil['ax_neuropil_per_cell_all'].set_xlabel('neuropil/ROI fluorescence per cell')
    axs_neuropil['ax_neuropil_per_cell_all_concat'].set_xlabel('neuropil/ROI fluorescence per sensor')
    axs_neuropil['exptime__per_cell_all'].set_xlabel('expression time vs neuropil/ROI fluorescence')
    
#%% expression time
fig_f0_vs_exptime = plt.figure()
ax_f0_exptime = fig_f0_vs_exptime.add_subplot(1,1,1)
for sensor_i,sensor in enumerate(sensors):
    ax_f0_exptime.semilogx(subject_expression_times[sensor],subject_median_roi_values[sensor],'o',color = sensor_colors[sensor], alpha = .5, label = sensor)
    ax_f0_exptime.semilogx(np.concatenate(sensor_expression_time[sensor]),np.concatenate(sensor_f_corrs[sensor]),'o',color = sensor_colors[sensor], alpha = .1,markersize = 1)
    ax_f0_exptime.legend()
ax_f0_exptime.set_xlabel('Expression time (days)')
ax_f0_exptime.set_ylabel('power corrected median f0')
#%% plot F0 distributions
fig_roi_fluorescence = plt.figure()
axs_sensors = dict()
axs_sensors_norm = dict()
axs_sensors_norm_conc = dict()
binnum = 100
alpha = .5
# =============================================================================
# if depth_correct_squared:
#     f_corr_bins = np.arange(0,15000,1000)
# else:
#     f_corr_bins = np.arange(0,200,10)
# =============================================================================
f_corr_norm_bins = np.arange(-3,5,.25)
f_corr_norm_bins_concatenated = np.arange(-3,5,.1)
f_corr_norm_bins_concatenated_cumsum = np.arange(-3,5,.01)
colnum = 3

axs_sensors['all']=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,len(sensors)*colnum+1)
axs_sensors_norm['all']=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,len(sensors)*colnum+2)
for sensor_i,sensor in enumerate(sensors):
    axs_sensors[sensor]=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+1)
    #axs_sensors[sensor].hist(sensor_f_corrs[sensor],f_corr_bins)
    for origvals in sensor_f_corrs[sensor]:
        origvals = origvals[np.isnan(origvals)==False]
        vals, binedges = np.histogram(origvals,binnum)
        bincenters = np.mean([binedges[:-1],binedges[1:]],0)
        axs_sensors[sensor].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
        axs_sensors['all'].semilogx(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
    
    
    #axs_sensors[sensor].set_title(sensor)
    axs_sensors[sensor].set_ylabel(sensor)
    if sensor_i ==0:
        axs_sensors[sensor].set_title(roi_name)
    
    axs_sensors_norm[sensor]=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+2)
    #axs_sensors_norm[sensor].hist(sensor_f_corrs_norm[sensor],f_corr_norm_bins)
    
    for origvals in sensor_f_corrs_norm[sensor]:
        origvals = origvals[np.isnan(origvals)==False]
        vals, binedges = np.histogram(origvals,binnum)
        bincenters = np.mean([binedges[:-1],binedges[1:]],0)
        axs_sensors_norm[sensor].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
        axs_sensors_norm['all'].plot(bincenters,np.cumsum(vals)/sum(vals),'-',color = sensor_colors[sensor],alpha = alpha)
    axs_sensors_norm[sensor].set_xlim([-3,5])
    axs_sensors_norm[sensor].set_ylim([0,1])
    axs_sensors_norm['all'].set_xlim([-3,5])
    axs_sensors_norm['all'].set_ylim([0,1])
    #axs_sensors_norm[sensor].set_title(sensor)
    
    axs_sensors_norm_conc[sensor]=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+3)
    axs_sensors_norm_conc['{}_cumsum'.format(sensor)] = axs_sensors_norm_conc[sensor].twinx()
    axs_sensors_norm_conc[sensor].hist(np.concatenate(sensor_f_corrs_norm[sensor]),f_corr_norm_bins_concatenated,color = 'black')
    vals, binedges = np.histogram(np.concatenate(sensor_f_corrs_norm[sensor]),f_corr_norm_bins_concatenated_cumsum)
    bincenters = np.mean([binedges[:-1],binedges[1:]],0)
    axs_sensors_norm_conc['{}_cumsum'.format(sensor)].plot(bincenters,np.cumsum(vals)/sum(vals),'r-')
    axs_sensors_norm_conc['{}_cumsum'.format(sensor)].set_ylim([0,1])
    #axs_sensors_norm_conc[sensor].set_title(sensor)
sensor = 'all sensors'
sensor_i += 1

#axs_sensors[sensor].set_title(sensor)
axs_sensors['all'].set_ylabel(sensor)
axs_sensors['all'].set_xlabel('power corrected f0')


#axs_sensors_norm[sensor].hist(f_corrs_norm,f_corr_norm_bins)
#axs_sensors_norm[sensor].set_title(sensor)
axs_sensors_norm['all'].set_xlabel('corrected f0 - normalized per subject')    

axs_sensors_norm_conc[sensor]=fig_roi_fluorescence.add_subplot(len(sensors)+1,colnum,sensor_i*colnum+3)
axs_sensors_norm_conc['{}_cumsum'.format(sensor)] = axs_sensors_norm_conc[sensor].twinx()
axs_sensors_norm_conc[sensor].hist(np.concatenate(f_corrs_norm),f_corr_norm_bins_concatenated,color = 'black')
vals, binedges = np.histogram(np.concatenate(f_corrs_norm),f_corr_norm_bins_concatenated_cumsum)
bincenters = np.mean([binedges[:-1],binedges[1:]],0)
axs_sensors_norm_conc['{}_cumsum'.format(sensor)].plot(bincenters,np.cumsum(vals)/sum(vals),'r-')
axs_sensors_norm_conc['{}_cumsum'.format(sensor)].set_ylim([0,1])
#axs_sensors_norm_conc[sensor].set_title(sensor)
axs_sensors_norm_conc[sensor].set_xlabel('corrected f0 - normalized per subject, concatenated')  
# =============================================================================
#     if i ==13:
#         break
# =============================================================================


#%% stats for all experiments
plot_properties = {'sensors':['GCaMP7F','XCaMPgf','456','686','688'],
                   'sensor_colors':sensor_colors,
                   'max_expression_time':170}
imaging_plot.plot_stats(plot_properties)
#%% AP wise SNR and hw distribution and sample AP
filters= {'hw_filter_sweep':'sweep_ap_hw_std_per_median<=.2',#'sweep_ap_hw_std_per_median<330.3',#'sweep_ap_snr_dv_median>=0',
          'snr_filter_sweep':'sweep_ap_snr_dv_median>10'}
ephys_plot.ap_hw_snr_distribution_plot(filters)
#%% hw std distribution and cutoff
hw_std_cut_value = .2
snr_cut_value = 10
sweep_ap_snr_dv_median,sweep_ap_snr_dv_min,sweep_ap_hw_max,sweep_ap_hw_median, sweep_ap_hw_std,sweep_ap_fw_std,sweep_ap_fw_median,sweep_ap_hw_std_per_median,sweep_ap_fw_std_per_median= (ephysanal_cell_attached.SweepAPQC()).fetch('sweep_ap_snr_dv_median','sweep_ap_snr_dv_min','sweep_ap_hw_max','sweep_ap_hw_median','sweep_ap_hw_std','sweep_ap_fw_std','sweep_ap_fw_median','sweep_ap_hw_std_per_median','sweep_ap_fw_std_per_median')
#%
sweep_ap_hw_std_per_median = sweep_ap_hw_std_per_median[np.isnan(sweep_ap_hw_std_per_median)==False]
sweep_ap_snr_dv_median = sweep_ap_snr_dv_median[np.isnan(sweep_ap_snr_dv_median)==False]
#sweep_ap_hw_std_per_median = sweep_ap_hw_std[np.isnan(sweep_ap_hw_std)==False]
fig_hw_hist = plt.figure()
ax_hw_hist = fig_hw_hist.add_subplot(2,1,1)
ax_hw_hist.hist(sweep_ap_hw_std_per_median,np.arange(0,1,.005),color = 'black')
ax_hw_cumsum = ax_hw_hist.twinx()
hwvals, binedges = np.histogram(sweep_ap_hw_std_per_median,200)
bincenters = np.mean([binedges[:-1],binedges[1:]],0)
ax_hw_cumsum.plot(bincenters,np.cumsum(hwvals)/np.sum(hwvals),'r-')
ax_hw_cumsum.plot([hw_std_cut_value,hw_std_cut_value],[0,1],'r--')
ax_hw_cumsum.set_ylim([0,1])
ax_hw_hist.set_title('std/median of halfwidth in a sweep')
ax_hw_hist.set_ylabel('# of sweeps')
ax_hw_hist.set_xlim([0,1])

ax_snr_hist = fig_hw_hist.add_subplot(2,1,2)
ax_snr_hist.hist(sweep_ap_snr_dv_median,np.arange(0,200,5),color = 'black')
#%
ax_snr_cumsum = ax_snr_hist.twinx()
hwvals, binedges = np.histogram(sweep_ap_snr_dv_median,200)
bincenters = np.mean([binedges[:-1],binedges[1:]],0)
ax_snr_cumsum.plot(bincenters,np.cumsum(hwvals)/np.sum(hwvals),'r-')
ax_snr_cumsum.plot([snr_cut_value,snr_cut_value],[0,1],'r--')
ax_snr_cumsum.set_ylim([0,1])
ax_snr_hist.set_title('median SNR in a sweep')
ax_snr_hist.set_ylabel('# of sweeps')
#ax_snr_hist.set_xlim([0,1])


#%%
snrs = (ephysanal_cell_attached.ActionPotential()*ephysanal_cell_attached.SweepAPQC()&'sweep_ap_snr_dv_median>20').fetch('ap_snr_dv')
#%% fetch data for AP plots
ap_data_to_fetch = ['subject_id',
                    'session',
                    'cell_number',
                    'sweep_number',
                    'ap_halfwidth',
                    'recording_mode',
                    'ap_amplitude',
                    'ap_ahp_amplitude',
                    'ap_max_index',
                    'ap_full_width',
                    'ap_rise_time',
                    'ap_peak_to_trough_time',
                    'ap_isi',
                    'ap_integral',
                    'ap_snr_v',
                    'ap_snr_dv']#%subject_id,session,cell_number,sweep_number,ap_halfwidth,recording_mode,ap_amplitude,ap_max_index
hw_filter_sweep = 'sweep_ap_hw_std_per_median<=0.2'#'sweep_ap_fw_std_per_median>0.65'#'sweep_ap_hw_median>.75'
snr_filter_sweep = 'sweep_ap_snr_dv_median>10'
apdatalist = (ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.ActionPotential()*ephysanal_cell_attached.SweepAPQC()&hw_filter_sweep&snr_filter_sweep).fetch(*ap_data_to_fetch)
#%
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
uniquecells = np.unique(cell_IDs)
ccidx = ap_data['recording_mode'] == 'current clamp'
vcidx = ap_data['recording_mode'] == 'voltage clamp'
#%% fetch and plot single APs and respective sweep on click
plot_parameters = {'alpha':.1}
data = ephys_plot.plot_ap_features_scatter(plot_parameters,ap_data)

#%% collect AP features per cell
recording_mode = 'voltage clamp'
minapnum = 5
minsnr = 10
isi_percentile_needed = 1
features_needed = [#'ap_halfwidth_mean',
                   'ap_halfwidth_median',
                   #'ap_full_width_mean',
                   #'ap_full_width_median',
                   #'ap_rise_time_mean',
                   #'ap_rise_time_median',
                   #'ap_peak_to_trough_time_mean',
                   'ap_peak_to_trough_time_median',
                   #'ap_isi_mean',
                   #'ap_isi_median',
                   #'ap_integral_mean',
                   #'ap_integral_median',
                   'ap_isi_percentiles',
                   'ahp_freq_dependence',
                   #'ap_ahp_amplitude_mean',
                   'ap_ahp_amplitude_median',
                   #'ap_snr_mean',
                   'ap_amplitude_median']
                   #]
# ----------------------------- PARAMETERS -----------------------------
TSNE_perplexity = 50#4
TSNE_learning_rate = 15#10#5#14
UMAP_learning_rate = .5#70#
UMAP_n_neigbors =50#4#14
UMAP_min_dist = 0.01 # between 0 and 1
UMAP_n_epochs=1000
# ----------------------------- PARAMETERS -----------------------------
pickertolerance = 5    
ephys_plot.cell_clustering_plots(recording_mode=recording_mode,
                                 sensor_colors = sensor_colors,
                                 minapnum=minapnum,
                                 minsnr=minsnr,
                                 isi_percentile_needed = isi_percentile_needed,
                                 features_needed = features_needed,
                                 TSNE_perplexity = TSNE_perplexity,
                                 TSNE_learning_rate = TSNE_learning_rate,
                                 UMAP_learning_rate = UMAP_learning_rate,
                                 UMAP_n_neigbors =UMAP_n_neigbors,
                                 UMAP_min_dist = UMAP_min_dist,
                                 UMAP_n_epochs = UMAP_n_epochs,
                                 pickertolerance = pickertolerance,
                                 invert_y_axis = False)

#%% #collect calcium wave properties for all sensors
import utils.utils_plot as utils_plot
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
                      'neuropil_subtraction':'none',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20,
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
ca_waves_dict = utils_plot.collect_ca_wave_parameters(ca_wave_parameters,cawave_properties_needed,ephys_properties_needed)
#%% compare sensors with 1AP - scatter plots
import plot.imaging_gt_plot as imaging_plot
ap_num_to_plot = 1
min_ap_per_apnum = 5    
plot_parameters={'sensor_colors':sensor_colors,
                 'min_ap_per_apnum':min_ap_per_apnum,
                 'ap_num_to_plot':ap_num_to_plot,
                 'event_histogram_step' : 5,
                 'scatter_plot_alpha':.1,
                 'scatter_marker_size':3,
                 'order_cells_by':'cawave_snr_median_per_ap',#'roi_f0_percentile_median_per_ap',#
                 'correct_f0_with_power_depth':True}#cawave_snr_median_per_ap#cawave_peak_amplitude_dff_median_per_ap
#%
imaging_plot.scatter_plot_all_event_properties(ca_waves_dict,ca_wave_parameters,plot_parameters)
#%% neuropil distribution
imaging_plot.plot_r_neu_distribution(sensor_colors,ca_wave_parameters)
#%% ap parameters scatter plot, calcium sensor, median ap wave amplitude
plot_parameters = {'sensors':['GCaMP7F','XCaMPgf','456','686','688'],
                   'sensor_colors':sensor_colors,
                   'ap_num_to_plot':1,
                   'x_parameter':'ap_peak_to_trough_time_median',#'ap_peak_to_trough_time_median' # not selected by AP number
                   'y_parameter':'ap_ahp_amplitude_median',# not selected by AP number
                   'z_parameter':'cawave_snr_median_per_ap',#'roi_f0_percentile_median_per_ap',#'cawave_snr_median_per_ap',#
                   'invert y axis':True,
                   'alpha' : .5,
                   'markersize_edges' : [2,20],
                   'z_parameter_percentiles':[5,95],
                   'normalize z by sensor':True}#'cawave_peak_amplitude_dff_median_per_ap'
imaging_plot.plot_ap_scatter_with_ophys_prop(ca_waves_dict,plot_parameters)
#%% download sensor kinetics
wave_parameters = {'ap_num_to_plot':1,#ap_num_to_plot
                   'min_ap_per_apnum':min_ap_per_apnum}
ca_traces_dict,ca_traces_dict_by_cell  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,ca_waves_dict,wave_parameters   )
#%% plot all traces        
bin_step_for_cells = 1
bin_size_for_cells = 8   
bin_step = .5
bin_size = 8 #2 
normalize_raw_trace_offset = True
plot_parameters={'sensor_colors':sensor_colors,
                 'normalize_raw_trace_offset':normalize_raw_trace_offset,
                 'min_ap_per_apnum':min_ap_per_apnum,
                 'bin_step_for_cells':bin_step_for_cells,
                 'bin_size_for_cells':bin_size_for_cells,
                 'bin_step':bin_step,
                 'bin_size':bin_size,
                 'shaded_error_bar':'se',#'se','sd','none'
                 'start_time':-50, # ms
                 'end_time':500,#ms
                 }
imaging_plot.plot_all_traces(ca_traces_dict,ca_traces_dict_by_cell,ca_wave_parameters,plot_parameters)    

#%%
wave_parameters = {'ap_num_to_plot':2,#ap_num_to_plot
                   'min_ap_per_apnum':min_ap_per_apnum}
ca_traces_dict,ca_traces_dict_by_cell  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,ca_waves_dict,wave_parameters   )
#%% plot traces by ISI 
time_range_step = .005
time_range_window = .001
time_range_offset = .003
time_range_step_number = 9
bin_step = 1
bin_size = 8
fig_freq = plt.figure()

axs = list()

fig_merged = plt.figure()
axs_merged=dict()
start_time = -50
end_time = 200
min_event_number = 20
exclude_outliers = False
sensors_to_plot = ['456','686','688']
for sensor_i,sensor in enumerate(sensors_to_plot):
    cawave_ap_group_lengts = np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])
    sensor_axs = list()
    axs_merged[sensor]=dict()
    if sensor_i ==0:
        #axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1)
        axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(sensors_to_plot),1,sensor_i*2+1)
        master_ax = axs_merged[sensor]['ax_wave']
    else:
        #axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1,sharex = master_ax)
        axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(sensors_to_plot),1,sensor_i*2+1,sharex = master_ax)
    #axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1+len(sensors_to_plot),sharex = master_ax)
    axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2*len(sensors_to_plot),1,sensor_i*2+2,sharex = master_ax)
    for step_i in np.arange(time_range_step_number):
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
        trace = np.asarray(trace_dict['trace_mean'])
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
            
        alpha =  1-step_i*.1 #.1+step_i*.1
        axs_merged[sensor]['ax_hist'].hist(np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])[cawave_idx]*1000,int(time_range_step*1000),color =sensor_colors[sensor],alpha = alpha)
        axs_merged[sensor]['ax_wave'].plot(bin_centers,trace,color =sensor_colors[sensor],alpha = alpha,linewidth =2 )#
ax.set_xlim([start_time,end_time])  
axs_merged[sensor]['ax_wave'].set_xlim([start_time,end_time])     
# =============================================================================
#         except:
#             pass
# =============================================================================
        #break
    #break

#%% properties for a single sensor
min_ap_per_apnum = 5         
for sensor in ca_wave_parameters['sensors']:
    #key['session_calcium_sensor'] = sensor
    cellwave_by_cell_list = ca_waves_dict[sensor]
    fig = plt.figure()
    ax_all_ap_dff = fig.add_subplot(3,3,1)
    ax_cell_ap_dff = fig.add_subplot(3,3,4)
    ax_all_ap_f = fig.add_subplot(3,3,2)
    ax_cell_ap_f = fig.add_subplot(3,3,5)
    
    ax_all_ap_snr = fig.add_subplot(3,3,3)
    ax_cell_ap_snr = fig.add_subplot(3,3,6)
    
    ax_amplitude_vs_baseline_dff = fig.add_subplot(3,3,7)
    
    ax_cell_ap_num = fig.add_subplot(3,3,8)
    
    ax_cell_dff_std_vs_f0 =  fig.add_subplot(3,3,9)
    #ax_all_ap_f.set_title(key)
    ax_all_ap_f.set_xlabel('AP num')
    ax_all_ap_f.set_ylabel('raw event amplitude (pixel intensity)')
    ax_all_ap_dff.set_xlabel('AP num')
    ax_all_ap_dff.set_ylabel('event amplitude (df/f)')
    
    ax_all_ap_snr.set_xlabel('AP num')
    ax_all_ap_snr.set_ylabel('event SNR')
    ax_cell_ap_snr.set_xlabel('AP num')
    ax_cell_ap_snr.set_ylabel('event SNR')
    
    ax_cell_ap_num.set_xlabel('AP num')
    ax_cell_ap_num.set_ylabel('# of events per cell')
    ax_amplitude_vs_baseline_dff.set_xlabel('baseline dff')
    ax_amplitude_vs_baseline_dff.set_ylabel('peak amplitude dff for 1 AP')
    
    ax_cell_dff_std_vs_f0.set_xlabel('baseline f0')
    ax_cell_dff_std_vs_f0.set_ylabel('std of event amplitude (df/f)')
    
    cell_data_list = list()
    ax_cell_ap_num.plot([0,10],[min_ap_per_apnum,min_ap_per_apnum],'k-')
    for cell_data in cellwave_by_cell_list:
        needed_apnum_idxes = cell_data['event_num_per_ap']>=min_ap_per_apnum
        random_x_offset = np.random.uniform(-.4,.4,len(cell_data['ap_group_ap_num']))
        ax_all_ap_f.plot(cell_data['ap_group_ap_num']+random_x_offset,
                         cell_data['cawave_peak_amplitude_f'],
                         'o',markersize = 1)
        ax_cell_ap_f.errorbar(cell_data['ap_group_ap_num_mean_per_ap'][needed_apnum_idxes],
                              cell_data['cawave_peak_amplitude_f_mean_per_ap'][needed_apnum_idxes],
                              yerr = cell_data['cawave_peak_amplitude_f_std_per_ap'][needed_apnum_idxes],
                              fmt='o-')
        ax_cell_ap_f.set_xlabel('AP num')
        ax_cell_ap_f.set_ylabel('raw event amplitude (pixel intensity)')
        
        
        ax_all_ap_dff.plot(cell_data['ap_group_ap_num']+random_x_offset,
                         cell_data['cawave_peak_amplitude_dff'],
                         'o',markersize = 1)
        ax_cell_ap_dff.errorbar(cell_data['ap_group_ap_num_mean_per_ap'][needed_apnum_idxes],
                                cell_data['cawave_peak_amplitude_dff_mean_per_ap'][needed_apnum_idxes],
                                yerr = cell_data['cawave_peak_amplitude_dff_std_per_ap'][needed_apnum_idxes],
                                fmt='o-')
        ax_cell_ap_dff.set_xlabel('AP num')
        ax_cell_ap_dff.set_ylabel('event amplitude (df/f)')
        
        
        ax_all_ap_snr.plot(cell_data['ap_group_ap_num']+random_x_offset,
                           cell_data['cawave_snr'],
                           'o',markersize = 1)
        ax_cell_ap_snr.errorbar(cell_data['ap_group_ap_num_mean_per_ap'][needed_apnum_idxes],
                                cell_data['cawave_snr_mean_per_ap'][needed_apnum_idxes],
                                yerr = cell_data['cawave_snr_std_per_ap'][needed_apnum_idxes],
                                fmt='o-')
        
        
        ax_cell_ap_num.plot(cell_data['ap_group_ap_num_mean_per_ap'],
                         cell_data['event_num_per_ap'],
                         'o',markersize = 1)
        aps_needed = cell_data['ap_group_ap_num']==1
        ax_amplitude_vs_baseline_dff.plot(cell_data['cawave_baseline_dff_global'][aps_needed],
                                     cell_data['cawave_peak_amplitude_dff'][aps_needed],
                                     'o',markersize = 1)
        
        ax_cell_dff_std_vs_f0.plot(cell_data['cawave_baseline_f_mean_per_ap'][needed_apnum_idxes],#cawave_baseline_f_mean_per_ap#ap_group_ap_num_mean_per_ap
                                   cell_data['cawave_peak_amplitude_dff_std_per_ap'][needed_apnum_idxes],'o')



#%% figure explaining superresolution analysis
key_first_ap = { 'session': 1, 'sweep_number': 2, 'ap_group_number': 112, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': 'calculated', 'neuropil_number': 1}        
first_subject_id = 472181
first_cell_number= 7
first_ap_group_number = 112
first_sweep_number = 2
key_all_ap = {'session': 1, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': 'calculated', 'neuropil_number': 1,'ap_group_ap_num':1}        
snr_cond = 'ap_group_min_snr_dv>{}'.format(20)
preisi_cond = 'ap_group_pre_isi>{}'.format(ca_wave_parameters['min_pre_isi'])
postisi_cond = 'ap_group_post_isi>{}'.format(ca_wave_parameters['min_post_isi'])
basline_dff_cond = 'cawave_baseline_dff_global<{}'.format(ca_wave_parameters['max_baseline_dff'])
basline_dff_cond_2 = 'cawave_baseline_dff_global>{}'.format(ca_wave_parameters['min_baseline_dff'])
cawave_snr_cond = 'cawave_snr > {}'.format(5)
subject_ids,cell_numbers,ap_group_numbers,sweep_numbers = (imaging_gt.CalciumWave.CalciumWaveProperties()*ephysanal_cell_attached.APGroup()&key_all_ap& snr_cond & preisi_cond & postisi_cond & basline_dff_cond & basline_dff_cond_2 & cawave_snr_cond).fetch('subject_id','cell_number','ap_group_number','sweep_number')
ap_group_numbers = np.concatenate([[first_ap_group_number],ap_group_numbers])
sweep_numbers = np.concatenate([[first_sweep_number],sweep_numbers])
subject_ids = np.concatenate([[first_subject_id],subject_ids])
cell_numbers = np.concatenate([[first_cell_number],cell_numbers])
#%

plot_superresolution_instead_of_ephys = True
fig = plt.figure(figsize = [20,10])
ax_dff=fig.add_subplot(3,1,2)
ax_event_times = fig.add_subplot(3,1,1,sharex = ax_dff)
plt.axis('off')
ax_ephys = fig.add_subplot(3,1,3,sharex = ax_dff)
cawave_dff_list = list()
cawave_time_list = list()

for wave_idx,(sweep_number,ap_group_number,subject_id,cell_number) in enumerate(zip(sweep_numbers,ap_group_numbers,subject_ids,cell_numbers)):
    key_first_ap['sweep_number'] = sweep_number
    key_first_ap['ap_group_number'] = ap_group_number
    key_first_ap['subject_id'] = subject_id
    key_first_ap['cell_number'] = cell_number
    cawave_f,cawave_fneu,cawave_time,ap_group_trace,sample_rate,ap_group_trace_peak_idx,cawave_rneu,cawave_baseline_f = (imaging_gt.CalciumWave.CalciumWaveProperties()*ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.APGroupTrace()*imaging_gt.CalciumWave.CalciumWaveNeuropil()*imaging_gt.CalciumWave()&key_first_ap).fetch1('cawave_f','cawave_fneu','cawave_time','ap_group_trace','sample_rate','ap_group_trace_peak_idx','cawave_rneu','cawave_baseline_f')
    #cawave_rneu,cawave_baseline_f = (imaging_gt.CalciumWave.CalciumWaveProperties()&key).fetch1('cawave_rneu','cawave_baseline_f')
    cawave_f_corr = cawave_f-cawave_fneu*cawave_rneu
    cawave_dff = (cawave_f_corr-cawave_baseline_f)/cawave_baseline_f 
    ap_group_trace_time = np.arange(-ap_group_trace_peak_idx,len(ap_group_trace)-ap_group_trace_peak_idx)/sample_rate*1000
    ax_dff.plot(cawave_time,cawave_dff,'go-',alpha = .5)
    ax_ephys.plot(ap_group_trace_time,ap_group_trace,'k-',alpha = .5)
    ax_event_times.plot(cawave_time,np.zeros(len(cawave_time))+wave_idx,'go',alpha = .5)
    cawave_dff_list.append(cawave_dff)
    cawave_time_list.append(cawave_time)
    if wave_idx == 500:
        break
ax_event_times.set_ylim([-10,510])

ax_ephys.set_xlim([-25,50])
ax_ephys.set_ylim([-.5,1.5])
ax_ephys.set_xlabel('Time from AP peak (ms)')
ax_ephys.set_ylabel('Ephys normalized')
ax_dff.set_ylabel('dF/F')
ax_dff.set_ylim([-.5,2])
#%
if plot_superresolution_instead_of_ephys:
    x_conc = np.concatenate(cawave_time_list)
    y_conc = np.concatenate(cawave_dff_list)
    needed_idx = (x_conc>-30)& (x_conc<60)
    x_conc = x_conc[needed_idx]
    y_conc = y_conc[needed_idx]
    trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,bin_step=1,bin_size=5,function = 'all',dogaussian = True)
    ax_ephys.cla()
    ax_ephys.plot(trace_dict['bin_centers'],trace_dict['trace_gauss'],'go-')#trace_mean
    ax_ephys.set_ylabel('dF/F')
    #ax_ephys.set_ylim([-.5,2])
    ax_ephys.set_xlim([-25,50])
#%%
        
rneu_16pix = (imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>.1'&'channel_number=1'&'neuropil_number = 3').fetch('r_neu')
rneu_8pix = (imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>.1'&'channel_number=1'&'neuropil_number = 2').fetch('r_neu')
rneu_4pix = (imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>.7'&'channel_number=1'&'neuropil_number = 1').fetch('r_neu')
rneu_2pix = (imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>.7'&'channel_number=1'&'neuropil_number = 0').fetch('r_neu')

pixnum_16pix = (imaging.ROINeuropil()*imaging_gt.ROISweepCorrespondance()&'neuropil_number = 3').fetch('neuropil_pixel_num')
pixnum_8pix = (imaging.ROINeuropil()*imaging_gt.ROISweepCorrespondance()&'neuropil_number = 2').fetch('neuropil_pixel_num')
pixnum_4pix = (imaging.ROINeuropil()*imaging_gt.ROISweepCorrespondance()&'neuropil_number = 1').fetch('neuropil_pixel_num')
pixnum_2pix = (imaging.ROINeuropil()*imaging_gt.ROISweepCorrespondance()&'neuropil_number = 0').fetch('neuropil_pixel_num')


fig = plt.figure()
ax_2 = fig.add_subplot(4,2,1)
ax_2.hist(rneu_2pix,50)
ax_4 = fig.add_subplot(4,2,3)
ax_4.hist(rneu_4pix,50)
ax_8 = fig.add_subplot(4,2,5)
ax_8.hist(rneu_8pix,50)
ax_16 = fig.add_subplot(4,2,7)
ax_16.hist(rneu_16pix,50)


ax_2_pix = fig.add_subplot(4,2,2)
ax_2_pix.hist(pixnum_2pix,50)
ax_4_pix = fig.add_subplot(4,2,4)
ax_4_pix.hist(pixnum_4pix,50)
ax_8_pix = fig.add_subplot(4,2,6)
ax_8_pix.hist(pixnum_8pix,50)
ax_16_pix = fig.add_subplot(4,2,8)
ax_16_pix.hist(pixnum_16pix,50)



#%% select high SNR movies for all sensors

