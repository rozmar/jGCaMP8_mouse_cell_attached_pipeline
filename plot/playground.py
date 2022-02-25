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
#%%
min_event_number = 10
sensors = ['456','686','688','GCaMP7F','XCaMPgf']
fig_peak_amplitudes = plt.figure()
ax_df=fig_peak_amplitudes.add_subplot(4,2,1)
ax_df.set_ylabel('Peak amplitude (F)')
ax_df_scatter=fig_peak_amplitudes.add_subplot(4,2,2)
ax_df_scatter.set_ylabel('std(amplitude) (F)')
ax_df_scatter.set_xlabel('mean(amplitude) (F)')
ax_dff=fig_peak_amplitudes.add_subplot(4,2,3)
ax_dff.set_ylabel('Peak amplitude (DF/F)')
ax_dff_scatter=fig_peak_amplitudes.add_subplot(4,2,4)
ax_dff_scatter.set_ylabel('std(amplitude) (DF/F)')
ax_dff_scatter.set_xlabel('mean(amplitude) (DF/F)')
ax_f0 = fig_peak_amplitudes.add_subplot(4,2,5)
ax_f0.set_ylabel('baseline fluorescence (F)')
# =============================================================================
# ax_f0_scatter = fig_peak_amplitudes.add_subplot(3,2,6)
# ax_f0_scatter.set_xlabel('baseline fluorescence (F)')
# ax_f0_scatter.set_ylabel('std(amplitude (DF/F)')
# =============================================================================
ax_dprime  = fig_peak_amplitudes.add_subplot(4,2,7)
ax_dprime.set_ylabel("d' (mean(DF)/std(DF))")
ax_dprome_box = fig_peak_amplitudes.add_subplot(4,2,8)
ax_dprome_box.set_ylabel("d' (mean(DF)/std(DF))")
x_offset = 0
amplitudes_idx_all = []
dprime_all = []
palettes = []
for sensor_i,sensor in enumerate(sensors):
    df_amplitude_mean_list = []
    df_amplitude_std_list = []
    dfperf_amplitude_mean_list = []
    dfperf_amplitude_std_list = []
    f0_list = []
    for cell_i, cell in enumerate(ca_waves_dict_pyr[sensor]):
        
        needed_events = cell['ap_group_ap_num'] == 1
        if np.sum(needed_events)<min_event_number:
            continue
        df_amplitude_mean_list.append(np.mean(cell['cawave_peak_amplitude_f'][needed_events]))
        df_amplitude_std_list.append(np.std(cell['cawave_peak_amplitude_f'][needed_events]))
        dfperf_amplitude_mean_list.append(np.mean(cell['cawave_peak_amplitude_dff'][needed_events]))
        dfperf_amplitude_std_list.append(np.std(cell['cawave_peak_amplitude_dff'][needed_events]))
        f0_list.append(np.mean(cell['cawave_baseline_f'][needed_events]))
        amplitudes_idx_all.append(sensor_i)
    palettes.append(sensor_colors['boxplot'][sensor])
    dprime = np.asarray(df_amplitude_mean_list)/np.asarray(df_amplitude_std_list)
    dprime_all.append(dprime)
    order = np.argsort(dprime)
    #order = np.argsort(np.asarray(dfperf_amplitude_mean_list)-np.asarray(dfperf_amplitude_std_list))
    ax_df.errorbar(np.arange(len(order))+x_offset,np.asarray(df_amplitude_mean_list)[order],np.asarray(df_amplitude_std_list)[order],fmt = 'o',color = sensor_colors[sensor],markerfacecolor = sensor_colors[sensor])
    ax_df_scatter.plot(df_amplitude_mean_list,df_amplitude_std_list,'o',color = sensor_colors[sensor])
    
    ax_dff.errorbar(np.arange(len(order))+x_offset,np.asarray(dfperf_amplitude_mean_list)[order],np.asarray(dfperf_amplitude_std_list)[order],fmt = 'o',color = sensor_colors[sensor],markerfacecolor = sensor_colors[sensor])
    ax_dff_scatter.plot(dfperf_amplitude_mean_list,dfperf_amplitude_std_list,'o',color = sensor_colors[sensor])
    
    ax_f0.plot(np.arange(len(order))+x_offset,np.asarray(f0_list)[order],'o',color = sensor_colors[sensor])
    #ax_f0_scatter.plot(f0_list,np.asarray(dfperf_amplitude_mean_list)/np.mean(dfperf_amplitude_mean_list),'o',color = sensor_colors[sensor])
    #ax_f0_scatter.plot(f0_list,dfperf_amplitude_std_list,'o',color = sensor_colors[sensor])
   #ax_f0_scatter.errorbar(f0_list,np.asarray(dfperf_amplitude_mean_list)[order],np.asarray(dfperf_amplitude_std_list)[order],fmt = 'o',color = sensor_colors[sensor],markerfacecolor = sensor_colors[sensor])
   
    
    ax_dprime.plot(np.arange(len(order))+x_offset,np.asarray(dprime)[order],'o',color = sensor_colors[sensor])
    x_offset+= len(order)
    
dprime_all = np.concatenate(dprime_all)   
sns.stripplot(x =amplitudes_idx_all,y = dprime_all,color='black',ax = ax_dprome_box,alpha = .6,size = 5)  #swarmplot   

sns.boxplot(ax=ax_dprome_box,    
            x=amplitudes_idx_all, 
            y=dprime_all, 
            showfliers=False, 
            linewidth=2,
            width=.5,
            palette = palettes)
ax_dprome_box.set_xticks(np.arange(len(sensors)))
ax_dprome_box.set_xticklabels(sensors)  
#%%
def estimate_statistics(values,function = 'std',bootstrapping = False):
    """
    function to estimate standard deviation

    Parameters
    ----------
    values : array of float
    function: str - std or mean
    bootstrapping: set true to do bootstrapping to estimate the stats
        .

    Returns
    -------
    output : estimate
    ci : 2x1, confidence interval

    """
    if function == 'std':
        if bootstrapping:
            bs = np.random.choice(values, (len(values), 100), replace=True)
            bs_stds = bs.std(axis=0)
            output = bs_stds.mean()
            ci = [np.quantile(bs_stds, 0.05),np.quantile(bs_stds, 0.95)]
        else:
            output = np.std(values)
            ci = None
    elif function == 'mean':
        if bootstrapping:
            bs = np.random.choice(values, (len(values), 100), replace=True)
            bs_means = bs.mean(axis=0)
            output = bs_means.mean()
            ci = [np.quantile(bs_means, 0.05),np.quantile(bs_means, 0.95)]
        else:
            output = np.mean(values)
            ci = None
    return output, ci

def test_stat_estimate(sample_num = 5, runs = 100,function = 'std',bootstrapping = False):
    """
    Parameters
    ----------
    sample_num : int, optional
        DESCRIPTION. Number of random samples to generate. The default is 5.
    runs : int, optional
        Number of runs to generate statistics of the estimate. The default is 100.
    function : str, either std or mean
        Function that should be tested
    bootstrapping:
        if the function should be bootstrapped or not

    Returns
    -------
    error_mean : float
        mean normalized error  0 = no error, 1 = error is the same magnitude as value
    error_std : float 
        std of normalized error

    """
    if function == 'std':
        expected_value = 10
    elif function == 'mean':
        expected_value = 100

    error_list = []
    ci_list_lower = []
    ci_list_upper = []
    for i in range(runs):
        rnd_vals = np.random.poisson(100,sample_num)
        #rnd_vals = np.random.normal(1000,5000,sample_num)
        std_est,ci = estimate_statistics(rnd_vals,function,bootstrapping)
        error_list.append(np.abs(std_est-expected_value))
        if bootstrapping:
            ci_list_lower.append(ci[0]/expected_value)
            ci_list_upper.append(ci[1]/expected_value)
    error_percentage = np.asarray(error_list)/expected_value
    error_mean = np.mean(error_percentage)
    error_std = np.std(error_percentage)
    if bootstrapping:
        ci_mean = [np.mean(ci_list_lower),np.mean(ci_list_upper)]
        ci_std = [np.std(ci_list_lower),np.std(ci_list_upper)]
    else:
        ci_mean = ci_std = None
    return error_mean, error_std, ci_mean, ci_std



def plot_std_mean_estimate_errors(sample_nums = np.arange(5,55,5)):
    std_error_mean_list = []
    std_error_std_list = []
    std_error_mean_boot_list = []
    std_error_std_boot_list = []
    
    mean_error_mean_list = []
    mean_error_std_list = []
    mean_error_mean_boot_list = []
    mean_error_std_boot_list = []
    
    
    mean_ci_mean_boot_list = []
    mean_ci_std_boot_list = []
    
    std_ci_mean_boot_list = []
    std_ci_std_boot_list = []
    
    for sample_num in sample_nums:
        error_mean, error_std, ci_mean, ci_std = test_stat_estimate(sample_num=sample_num,runs = 1000,function = 'std',bootstrapping =False)
        std_error_mean_list.append(error_mean)
        std_error_std_list.append(error_std)
        error_mean, error_std, ci_mean, ci_std = test_stat_estimate(sample_num=sample_num,runs = 1000 ,function = 'std',bootstrapping =True)
        std_error_mean_boot_list.append(error_mean)
        std_error_std_boot_list.append(error_std)
        std_ci_mean_boot_list.append(ci_mean)
        std_ci_std_boot_list.append(ci_std)
        
        error_mean, error_std, ci_mean, ci_std = test_stat_estimate(sample_num=sample_num,runs = 1000,function = 'mean',bootstrapping =False)
        mean_error_mean_list.append(error_mean)
        mean_error_std_list.append(error_std)
        error_mean, error_std, ci_mean, ci_std = test_stat_estimate(sample_num=sample_num,runs = 1000 ,function = 'mean',bootstrapping =True)
        mean_error_mean_boot_list.append(error_mean)
        mean_error_std_boot_list.append(error_std)
        mean_ci_mean_boot_list.append(ci_mean)
        mean_ci_std_boot_list.append(ci_std)
        
    #%
    fig_std_estimation = plt.figure()
    ax_original = fig_std_estimation.add_subplot(2,2,1)
    ax_original.errorbar(sample_nums,std_error_mean_list,std_error_std_list,fmt='o-',label = 'numpy.std()', alpha = .8)
    ax_original.errorbar(sample_nums,std_error_mean_boot_list,std_error_std_boot_list,fmt='o-',label = 'bootstrapped', alpha = .8)
    ax_original.set_title('Standard deviation')
    ax_original.set_ylabel('normalized error in estimating std')
    ax_original.set_xlabel('sample size')
    ax_original.legend()
    
    ax_mean = fig_std_estimation.add_subplot(2,2,2)
    ax_mean.errorbar(sample_nums,mean_error_mean_list,mean_error_std_list,fmt='o-',label = 'numpy.mean()')
    ax_mean.errorbar(sample_nums,mean_error_mean_boot_list,mean_error_std_boot_list,fmt='o-',label = 'bootstrapped')
    ax_mean.set_title('Mean')
    ax_mean.set_ylabel('normalized error in estimating mean')
    ax_mean.set_xlabel('sample size')
    ax_mean.legend()
    
    ax_std_ci = fig_std_estimation.add_subplot(2,2,3)
    ax_std_ci.errorbar(sample_nums,np.asarray(std_ci_mean_boot_list)[:,0],np.asarray(std_ci_std_boot_list)[:,0],fmt='o-')
    ax_std_ci.errorbar(sample_nums,np.asarray(std_ci_mean_boot_list)[:,1],np.asarray(std_ci_std_boot_list)[:,1],fmt='o-')
    ax_std_ci.set_ylabel('normalized confidence interval')
    ax_std_ci.set_xlabel('sample size')
    
    ax_std_ci = fig_std_estimation.add_subplot(2,2,4)
    ax_std_ci.errorbar(sample_nums,np.asarray(mean_ci_mean_boot_list)[:,0],np.asarray(mean_ci_std_boot_list)[:,0],fmt='o-')
    ax_std_ci.errorbar(sample_nums,np.asarray(mean_ci_mean_boot_list)[:,1],np.asarray(mean_ci_std_boot_list)[:,1],fmt='o-')
    ax_std_ci.set_ylabel('normalized confidence interval')
    ax_std_ci.set_xlabel('sample size')

plot_std_mean_estimate_errors(sample_nums = np.arange(5,55,5))
#%% how does firing frequency change during a visual stim - not at all
from scipy.stats import wilcoxon
quality_key= {'max_sweep_ap_hw_std_per_median': .2,
              'min_sweep_ap_snr_dv_median' : 10}
ephys_snr_cond = 'sweep_ap_snr_dv_median>={}'.format(quality_key['min_sweep_ap_snr_dv_median'])
ephys_hwstd_cond = 'sweep_ap_hw_std_per_median<={}'.format(quality_key['max_sweep_ap_hw_std_per_median'])
i=0
sensor_dict= dict()
for cell in ephys_cell_attached.Cell():
    print(cell)
    sensor = (imaging_gt.SessionCalciumSensor()&cell).fetch1('session_calcium_sensor')
    celltype = (ephysanal_cell_attached.EphysCellType()&cell).fetch1('ephys_cell_type')
    session_time = (experiment.Session()&cell).fetch1('session_time')
    cell_start_time = cell['cell_recording_start'].total_seconds() - session_time.total_seconds()
    cell_end_time = cell_start_time+ float(np.max((ephys_cell_attached.Sweep()&cell).fetch('sweep_end_time')))
    ap_max_time,sweep_start_time = (ephysanal_cell_attached.SweepAPQC()*ephys_cell_attached.Sweep()*ephysanal_cell_attached.ActionPotential()&cell&ephys_snr_cond&ephys_hwstd_cond).fetch('ap_max_time','sweep_start_time')
    ap_max_time = np.asarray(ap_max_time,float)+np.asarray(sweep_start_time,float) +cell['cell_recording_start'].total_seconds() - session_time.total_seconds()
    apgroups = ephys_cell_attached.Sweep()*ephysanal_cell_attached.SweepAPQC()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.ROISweepCorrespondance()&'channel_number = 1' & 'neuropil_number = 1'&'neuropil_subtraction = "0.8"'&'gaussian_filter_sigma = 10'&cell&ephys_snr_cond&ephys_hwstd_cond
    if len(apgroups) == 0:
        continue
    ap_group_start_time,ap_group_ap_num,cawave_peak_amplitude_dff,sweep_start_time = apgroups.fetch('ap_group_start_time','ap_group_ap_num','cawave_peak_amplitude_dff','sweep_start_time')
    ap_group_start_time = np.asarray(ap_group_start_time-sweep_start_time,float)#np.asarray(ap_group_start_time,float) +cell['cell_recording_start'].total_seconds()-session_time.total_seconds()
    x = ap_group_start_time
    y = cawave_peak_amplitude_dff/ap_group_ap_num
    y = y/np.mean(y)
    superres_dict = utils_plot.generate_superresolution_trace(x,y,2,10,function = 'all',dogaussian = False)

    visualstim_trials = experiment.VisualStimTrial()&cell&'visual_stim_trial_start >= {}'.format(cell_start_time-10)&'visual_stim_trial_end <= {}'.format(cell_end_time+10)
    angles = np.unique(visualstim_trials.fetch('visual_stim_grating_angle'))
    cell_dict = dict()
    cell_dict['angle'] = list()
    cell_dict['sweep_time'] = list()
    cell_dict['ctrl_freq'] = list()
    cell_dict['gratings_freq'] = list()
    cell_dict['diff_freq'] = list()
    cell_dict['cawave_superres'] = superres_dict
    
    for visualstim_trial in visualstim_trials:
        ctrl_ap = sum((ap_max_time>float(visualstim_trial['visual_stim_trial_start'])) & (ap_max_time<float(visualstim_trial['visual_stim_grating_start'])))/float(visualstim_trial['visual_stim_grating_start']-visualstim_trial['visual_stim_trial_start'])
        gratings_ap = sum((ap_max_time>float(visualstim_trial['visual_stim_grating_start'])) & (ap_max_time<float(visualstim_trial['visual_stim_trial_end'])))/float(visualstim_trial['visual_stim_trial_end']-visualstim_trial['visual_stim_grating_start'])
        cell_dict['angle'].append(visualstim_trial['visual_stim_grating_angle'])
        cell_dict['sweep_time'].append(visualstim_trial['visual_stim_sweep_time'])
        cell_dict['ctrl_freq'].append(ctrl_ap)
        cell_dict['gratings_freq'].append(gratings_ap)
        cell_dict['diff_freq'].append(gratings_ap-ctrl_ap)
    #%
    cell_dict['p_values']=np.ones(len(cell_dict['diff_freq']))
    cell_dict['diff_freq_mean']=np.zeros(len(cell_dict['diff_freq']))
    for key in cell_dict.keys():
        cell_dict[key]=np.asarray(cell_dict[key])
    for angle in np.unique(cell_dict['angle']):
        idx = cell_dict['angle'] == angle
        stat,p = wilcoxon(cell_dict['diff_freq'][idx])
        cell_dict['p_values'][idx] = p
        cell_dict['diff_freq_mean'][idx] = np.nanmean(cell_dict['diff_freq'][idx])
    cell_dict['cell_key'] = cell
    
    if sensor not in sensor_dict.keys():
        sensor_dict[sensor] = dict()
    if celltype not in sensor_dict[sensor].keys():
        sensor_dict[sensor][celltype] = list()
    sensor_dict[sensor][celltype].append(cell_dict)
    i+=1
    if len(apgroups)>0:
        break
    
#%% how AP num and calcium transient amplitudes change over a sweep.. they are stable..
fig = plt.figure(figsize = [10,10]) 
fig_img = plt.figure(figsize = [10,10])
ax_dict = dict()
ax_dict_im = dict()
sensornum = len(sensor_dict.keys())
i=0
pvalue = .05
firstax = None
for x,sensor in enumerate(sensor_dict.keys()):
    for y,celltype in enumerate(['pyr','int']):
        i+=1
        if firstax == None:
            ax_dict['{}_{}'.format(sensor,celltype)] = fig.add_subplot(sensornum,2,i)
            firstax  = ax_dict['{}_{}'.format(sensor,celltype)]
            ax_dict_im['{}_{}'.format(sensor,celltype)] = fig_img.add_subplot(sensornum,2,i)
            firstax_im  = ax_dict_im['{}_{}'.format(sensor,celltype)]
        else:
            ax_dict['{}_{}'.format(sensor,celltype)] = fig.add_subplot(sensornum,2,i,sharex = firstax,sharey = firstax)
            ax_dict_im['{}_{}'.format(sensor,celltype)] = fig_img.add_subplot(sensornum,2,i,sharex = firstax_im,sharey = firstax_im)
        ax_dict['{}_{}'.format(sensor,celltype)].set_title('{} - {}'.format(sensor,celltype))
        ax_dict_im['{}_{}'.format(sensor,celltype)].set_title('{} - {}'.format(sensor,celltype))
        

for sensor in sensor_dict.keys():
    for celltype in ['pyr','int']:
        cell_dict_list = sensor_dict[sensor][celltype]
        xs = list()
        ys = list()
        x_img_list = list()
        y_img_list = list()
        for cell_dict in cell_dict_list:
            idx = cell_dict['p_values']<pvalue
            x = cell_dict['sweep_time'][idx]
            y = cell_dict['diff_freq'][idx]#cell_dict['diff_freq'][idx]/cell_dict['diff_freq_mean'][idx]
            ax_dict['{}_{}'.format(sensor,celltype)].plot(x,y,'ko',ms = 4,alpha = .2)
            xs.append(x)
            ys.append(y)
            x_img = np.asarray(cell_dict['cawave_superres'].tolist()['bin_centers'])
            y_img = np.asarray(cell_dict['cawave_superres'].tolist()['trace_mean'])
            x_img = np.concatenate([x_img,np.zeros(150)*np.nan])
            y_img = np.concatenate([y_img,np.zeros(150)*np.nan])
            
            x_img_list.append(x_img[:150])
            y_img_list.append(y_img[:150])
            ax_dict_im['{}_{}'.format(sensor,celltype)].plot(x_img,y_img)
        superres_dict = utils_plot.generate_superresolution_trace(np.concatenate(xs),np.concatenate(ys),2,10,function = 'all',dogaussian = False)
        ax_dict['{}_{}'.format(sensor,celltype)].plot(superres_dict['bin_centers'],superres_dict['trace_mean'],'r-',linewidth = 4)
        ax_dict_im['{}_{}'.format(sensor,celltype)].plot(np.nanmean(x_img_list,0),np.nanmean(y_img_list,0),'r-',linewidth = 4)
        #break
        #ax_dict['{}_{}'.format(sensor,celltype)].errorbar(superres_dict['bin_centers'],superres_dict['trace_mean'],superres_dict['trace_std'])
# =============================================================================
#         break
#     break
# =============================================================================
    #break
            
    
    




#%%
#even_keys = 
for cell in ephys_cell_attached.Cell():
    print(cell)
    sensor = (imaging_gt.SessionCalciumSensor()&cell).fetch1('session_calcium_sensor')
    celltype = (ephysanal_cell_attached.EphysCellType()&cell).fetch1('ephys_cell_type')
    apgroups = ephys_cell_attached.Sweep()*ephysanal_cell_attached.SweepAPQC()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.ROISweepCorrespondance()&'channel_number = 1' & 'neuropil_number = 1'&'neuropil_subtraction = "0.8"'&'gaussian_filter_sigma = 10'&cell&ephys_snr_cond&ephys_hwstd_cond
    apgroups = ephys_cell_attached.Sweep()*ephysanal_cell_attached.SweepAPQC()*ephysanal_cell_attached.APGroup()*imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.ROISweepCorrespondance()&'channel_number = 1' & 'neuropil_number = 1'&'neuropil_subtraction = "0.8"'&'gaussian_filter_sigma = 10'&cell&ephys_snr_cond&ephys_hwstd_cond
    if len(apgroups) == 0:
        continue
    ap_group_start_time,ap_group_ap_num,cawave_peak_amplitude_dff,sweep_start_time = apgroups.fetch('ap_group_start_time','ap_group_ap_num','cawave_peak_amplitude_dff','sweep_start_time')
    ap_group_start_time = np.asarray(ap_group_start_time-sweep_start_time,float)#np.asarray(ap_group_start_time,float) +cell['cell_recording_start'].total_seconds()-session_time.total_seconds()
    x = ap_group_start_time
    y = cawave_peak_amplitude_dff/ap_group_ap_num
    y = y/np.mean(y)
    superres_dict = utils_plot.generate_superresolution_trace(x,y,2,5,function = 'all',dogaussian = False)
    
    break
#%% ROI weights are not consistent
weights,sensors = (imaging_gt.SessionCalciumSensor()*imaging.ROI()).fetch('roi_weights','session_calcium_sensor')
weights_sum = list()
for weight in weights: weights_sum.append(np.sum(weight))
#unique_sensors
plt.hist(weights_sum)
plt.hist(weights_sum,1000)
plt.yscale('log')
plt.ylabel('# of ROIs')
plt.xlabel('sum of pixel weights')


#%% neuropil vs LFP -  for Jerome at Allen
import scipy
import scaleogram as scg
sweeps=ephys_cell_attached.Sweep()
for i, sweep in enumerate(sweeps):
    if i<395:#395
        continue
    sweep_now = ephys_cell_attached.Sweep()*ephys_cell_attached.SweepResponse()*ephys_cell_attached.SweepMetadata()&'recording_mode = "current clamp"'&sweep
    if len(sweep_now )>0:
        neuropil_key = {'motion_correction_method': 'Suite2P',
                        'roi_type':'Suite2P',
                        'neuropil_number':1,
                        'channel_number':1}
        sweep_start_time =  sweep_now.fetch1('sweep_start_time')
        neuropil_f,frame_times  = (sweep_now*imaging_gt.ROISweepCorrespondance()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()&neuropil_key).fetch1('neuropil_f','frame_times')
        
        print(i)
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
        ephys_time = np.arange(len(trace))/sample_rate
        frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))
        frame_rate = 1/np.diff(frame_times)[0]
        trace_highpass = utils_ephys.hpFilter(trace_no_ap, .3, 3, sample_rate, padding = True)
        trace_bandpass = utils_ephys.lpFilter(trace_highpass, 200, 3, sample_rate, padding = True)
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
        ephys_std = utils_ephys.rollingfun(trace_bandpass_downsampled, window = std_window, func = 'std')
        xcorr = scipy.signal.correlate(scipy.stats.zscore(neuropil_f),scipy.stats.zscore(ephys_std), mode='full', method='direct')
        xcorr_t = np.arange(len(xcorr))/frame_rate-len(frame_times)/frame_rate
        #%
        fig = plt.figure(figsize = [10,10])
        ax_o = fig.add_subplot(511)
        ax_e = fig.add_subplot(512,sharex = ax_o)
        ax_e_std = fig.add_subplot(513,sharex = ax_o)
        ax_fft = fig.add_subplot(515)
        ax_xcorr = fig.add_subplot(514)
        ax_o.plot(frame_times,scipy.stats.zscore(neuropil_f))
        ax_e.plot(ephys_time,scipy.stats.zscore(trace_bandpass))
        #ax_e.plot(frame_times,trace_bandpass_downsampled)
        ax_fft.plot(out[0][needed],out[1][needed]*out[0][needed])
        #ax_e.set_title(i)
        ax_xcorr.plot(xcorr_t,xcorr)
        ax_e_std.plot(frame_times,scipy.stats.zscore(ephys_std))
        ax_o.set_ylabel('neuropil signal')
        ax_e.set_ylabel('LFP')
        
        ax_e_std.set_ylabel('std filtered LFP')
        ax_e_std.set_xlabel('time (s)')
        ax_xcorr.set_ylabel('Xcorr')
        ax_xcorr.set_xlabel('time (s)')
        #%
        break
        
        #%%
        
# =============================================================================
#         ephys_std = utils_ephys.rollingfun(trace_bandpass, window = 10, func = 'std')
#         
#         #%%
# sample_rate_psd = 500        
# downsample_factor_psd = int(sample_rate/sample_rate_psd)
# trace_to_psd = trace_bandpass[::downsample_factor_psd]
# time_trace_to_psd = ephys_time[::downsample_factor_psd]
# scg.set_default_wavelet('morl')
# signal_length = len(trace_to_psd)
# 
# freqs = np.logspace(1,np.log10(300),50)
# intervals = sample_rate_psd/freqs
# #%
# # choose default wavelet function 
# 
# 
# 
# # range of scales to perform the transform
# scales = scg.periods2scales( intervals )
# 
# 
# # plot the signal 
# fig1, ax1 = plt.subplots(1, 1, figsize=(9, 3.5));  
# ax1.plot(time_trace_to_psd, trace_to_psd, linewidth=3, color='blue')
# 
# # the scaleogram
# out = scg.cws(trace_to_psd[:signal_length], scales=scales, figsize=(10, 4.0), coi = True, ylabel="Period", xlabel="Time",yaxis = 'frequency')
#         
# =============================================================================



#%% model an image
framenum = 100
photon_per_dwelltime = 3
pixel_intensity_per_photon = 5000
pix_x = 512
pix_y = 100
linemask = [13, 12, 13, 12, 12, 12, 12, 11, 12, 11, 11, 11, 11, 11, 11, 10, 11,
       10, 10, 11, 10, 10, 10,  9, 10, 10,  9, 10,  9,  9, 10,  9,  9,  9,
        9,  9,  9,  9,  8,  9,  9,  8,  9,  8,  9,  8,  9,  8,  8,  8,  9,
        8,  8,  8,  8,  8,  8,  8,  7,  8,  8,  8,  7,  8,  8,  7,  8,  7,
        8,  7,  8,  7,  8,  7,  7,  8,  7,  7,  7,  8,  7,  7,  7,  7,  7,
        7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  7,  7,  7,  7,  6,
        7,  7,  6,  7,  7,  6,  7,  6,  7,  7,  6,  7,  6,  7,  6,  7,  6,
        6,  7,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  6,  7,
        6,  6,  6,  6,  7,  6,  6,  6,  6,  6,  6,  7,  6,  6,  6,  6,  6,
        6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
        6,  6,  5,  6,  6,  6,  6,  6,  6,  6,  6,  5,  6,  6,  6,  6,  6,
        5,  6,  6,  6,  6,  5,  6,  6,  6,  5,  6,  6,  6,  6,  5,  6,  6,
        6,  5,  6,  6,  5,  6,  6,  6,  5,  6,  6,  5,  6,  6,  6,  5,  6,
        6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,
        6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  5,  6,  6,  5,  6,  6,  5,
        6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  5,  6,  6,  5,  6,  6,  5,
        6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,  5,  6,  6,
        5,  6,  6,  5,  6,  6,  6,  5,  6,  6,  5,  6,  6,  6,  5,  6,  6,
        5,  6,  6,  6,  5,  6,  6,  6,  6,  5,  6,  6,  6,  5,  6,  6,  6,
        6,  5,  6,  6,  6,  6,  6,  5,  6,  6,  6,  6,  6,  6,  6,  6,  5,
        6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
        6,  6,  6,  6,  6,  6,  6,  7,  6,  6,  6,  6,  6,  6,  7,  6,  6,
        6,  6,  7,  6,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,  6,  7,  6,
        7,  6,  6,  7,  6,  7,  6,  7,  6,  7,  7,  6,  7,  6,  7,  7,  6,
        7,  7,  6,  7,  7,  7,  7,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,
        7,  7,  7,  7,  7,  7,  7,  8,  7,  7,  7,  8,  7,  7,  8,  7,  8,
        7,  8,  7,  8,  7,  8,  8,  7,  8,  8,  8,  7,  8,  8,  8,  8,  8,
        8,  8,  9,  8,  8,  8,  9,  8,  9,  8,  9,  8,  9,  9,  8,  9,  9,
        9,  9,  9,  9,  9, 10,  9,  9, 10,  9, 10, 10,  9, 10, 10, 10, 11,
       10, 10, 11, 10, 11, 11, 11, 11, 11, 11, 12, 11, 12, 12, 12, 12, 13,
       12, 13]
movie = list()

for i_frame in np.arange(framenum):
    frame = np.zeros([pix_y,pix_x])
    for i_y in np.arange(pix_y):
        for i_x,mask_now in enumerate(linemask):
            frame[i_y,i_x]=np.mean(np.random.poisson(photon_per_dwelltime,mask_now)*pixel_intensity_per_photon)
    movie.append(frame)
    print(i_frame)
#%% Noise analysis
import os
import scipy.ndimage as ndimage
noise_upper_limit = -30000
font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 9}
alpha = .005
matplotlib.rc('font', **font)
from ScanImageTiffReader import ScanImageTiffReader
from utils.utils_pipeline import extract_scanimage_metadata
import matplotlib.colors as colors
#noisedir = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/2021-02-14-fluorescent_slide'   
#noisedir = '/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/20210214-fluorescent_slide'  
#noisedir = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/2021-02-22-noise'   
noisedir = '/home/rozmar/Network/Gsuite/mr_share_1/Data/Calcium_imaging/raw/DOM3-MMIMS/2021-02-26-fluorescent_slide'   
#noisedir = '/home/rozmar/Data/Calcium_imaging/raw/DOM3-MMIMS/2021-02-26-fluorescent_slide/no_binning'
files = os.listdir(noisedir)
files_real = list()
for file in files:
    if '.tif' in file:
        files_real.append(file)
files = files_real[::-1] 
axs = list()
means_all = list()
vars_all = list()
files = list(np.sort(files))
files.append('sum')
#%
for file in files:# = 'green_PMT_off_00002_00001.tif'
    #%
    if file != 'sum':
        #%
        image = ScanImageTiffReader(os.path.join(noisedir,file))
        metadata = extract_scanimage_metadata(os.path.join(noisedir,file))
        linemask = np.asarray(metadata['metadata']['hScan2D']['mask'][1:-1].split(';'),int)
        offset_subtracted = metadata['metadata']['hChannels']['channelSubtractOffset'][1:-1].split(' ')[0]=='true'
        
       # movie = np.asarray(image.data(),float)
        movie = image.data()
        #linemask_altered = linemask/np.min(linemask)
        #movie = movie*(linemask_altered[np.newaxis,np.newaxis,:])
        if not offset_subtracted:
            movie -= np.asarray(metadata['metadata']['hChannels']['channelOffset'][1:-1].split(' '),int)[0]
        movie[movie<0] = 0
        #%
        means = np.nanmean(movie,0)
# =============================================================================
#         if np.mean(means)>10000:
#             continue
# =============================================================================
# =============================================================================
#         if np.min(means)<0:
#             means = means -np.min(means)
# =============================================================================
        variances = np.nanvar(movie,0)*(np.asarray(linemask)[np.newaxis,:])
        
# =============================================================================
#         means = ndimage.filters.gaussian_filter(means,1)
#         variances = ndimage.filters.gaussian_filter(variances,1)
# =============================================================================
        #%
        ratio = variances/means
        ratio_flat = ratio.flatten()
        edges = np.percentile(ratio_flat, [0.5, 99.5])
        edges_mean = np.percentile(means.flatten(), [0.5, 99.5])
        edges_var = np.percentile(variances.flatten(), [0.5, 99.5])
        means_all.append(means.flatten())
        vars_all.append(variances.flatten()  )
        movie_flat  =movie.flatten()
        movie_flat = movie_flat[movie_flat  > noise_upper_limit]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(movie_flat,1000)
        ax.set_xlabel('pixel intensity')
        ax.set_ylabel('pixel count')
        ax.set_title('mean intensity = {}'.format(int(np.mean(means))))
        #ax.set_yscale('log')
        #%
    else:
        means = np.concatenate(means_all)
        variances = np.concatenate(vars_all)
        ratio = variances/means
        ratio_flat = ratio.flatten()
        edges = np.percentile(ratio_flat, [0.5, 99.5])
        edges_mean = np.percentile(means.flatten(), [0.5, 99.5])
        edges_var = np.percentile(variances.flatten(), [0.5, 99.5])
        means_all.append(means.flatten())
        vars_all.append(variances.flatten()  )
  #  
    
#%
    fig = plt.figure(figsize = [7,15])
    
    axs.append(fig.add_subplot(3,2,1) )
    #axs[-1].plot(means.flatten(),ratio.flatten(),'ko',ms=2,alpha = alpha)
    axs[-1].hist2d(means.flatten(),ratio.flatten(),150,[edges_mean,edges],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
    axs[-1].set_title('{} - powers: {}'.format(file,metadata['metadata']['hBeams']['powers']))      
    axs[-1].set_xlabel('mean pixel intensity')
    axs[-1].set_ylabel('gain of pixel')
    #axs[-1].set_yscale('log')
    #axs[-1].set_xscale('log')
    
    
    axs.append(fig.add_subplot(3,2,5) )
    #axs[-1].plot(means.flatten(),variances.flatten(),'ko',ms=2,alpha = alpha)
    axs[-1].hist2d(means.flatten(),variances.flatten(),150,[edges_mean,edges_var],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
    #axs[-1].set_title('{} - powers: {}'.format(file,metadata['metadata']['hBeams']['powers']))      
    axs[-1].set_xlabel('mean pixel intensity')
    axs[-1].set_ylabel('variance of pixel intensity')
    #axs[-1].set_yscale('log')
    #axs[-1].set_xscale('log')
    
    if file != 'sum':
        axs.append(fig.add_subplot(3,2,2) )
        axs[-1].set_xlabel('x pixels')
        axs[-1].set_ylabel('y pixels')
        axs[-1].set_title('gain of each pixel (var/mean)')
        im = axs[-1].imshow(ratio)
        im.set_clim(edges[0], edges[1])
        fig.colorbar(im, ax=axs[-1])
        
    axs.append(fig.add_subplot(3,2,3) )
    
    axs[-1].hist(ratio_flat[ratio_flat<np.inf],np.arange(edges[0],edges[1],np.diff(edges)/100))
    axs[-1].set_xlabel('gain')
    axs[-1].set_ylabel('pixel count')
    
    if file != 'sum':
        axs.append(fig.add_subplot(3,2,4) )
        axs[-1].set_xlabel('x pixels')
        axs[-1].set_ylabel('y pixels')
        axs[-1].set_title('mean pixel intensity')
        im = axs[-1].imshow(means)
        fig.colorbar(im, ax=axs[-1])
    
    
        axs.append(fig.add_subplot(3,2,6) )
        axs[-1].set_xlabel('x pixels')
        axs[-1].set_ylabel('y pixels')
        axs[-1].set_title('variance of pixel intensity')
        im = axs[-1].imshow(variances)
        fig.colorbar(im, ax=axs[-1])
    #plt.show()
    #break
        
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.concatenate(means_all),np.concatenate(vars_all),'ko')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('mean pixel intensity')
ax.set_ylabel('variance of pixel intensity')
ax.set_title('multiple laser intensities and no laser')
     #%%       
     




#%% does snr corelate with overall firing rate?
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
fields_to_get = ['movie_mean_cawave_snr_per_ap',
                 'movie_total_ap_num',
                 'movie_frame_num',
                 'movie_frame_rate',
                 'ephys_cell_type',
                 'subject_id',
                 'cell_number',
                 'sweep_start_time',
                 'sweep_end_time',
                 'roi_f_min',
                 'movie_power']
fig = plt.figure()
for i,sensor in enumerate(sensors):
    if i ==0:
        ax_now = fig.add_subplot(len(sensors),3,3*i+1)
        ax_start = ax_now
        ax_now_cell = fig.add_subplot(len(sensors),3,3*i+2)
        ax_start_cell = ax_now_cell
    else:
        ax_now = fig.add_subplot(len(sensors),3,3*i+1,sharex = ax_start, sharey = ax_start)
        ax_now_cell = fig.add_subplot(len(sensors),3,3*i+2,sharex = ax_start_cell, sharey = ax_start_cell)
    data  = (imaging.ROITrace()*imaging.MoviePowerPostObjective()*ephys_cell_attached.Sweep()*imaging_gt.ROISweepCorrespondance()*ephysanal_cell_attached.EphysCellType()*imaging_gt.MovieCalciumWaveSNR()*imaging_gt.SessionCalciumSensor()*imaging.Movie() &'session_calcium_sensor = "{}"'.format(sensor)&'channel_number = 1').fetch(*fields_to_get)
    data_dict = dict()
    for fieldname,data_now in zip(fields_to_get,data):
        data_dict[fieldname] = data_now
    unique_cell_id = list()
    for subject_id,cell_number in zip(data_dict['subject_id'],data_dict['cell_number']):
        unique_cell_id.append('{}_{}'.format(subject_id,cell_number))
    unique_cell_id=np.asarray(unique_cell_id)
    movie_len = np.asarray(data_dict['sweep_end_time']-data_dict['sweep_start_time'],float)#data_dict['movie_frame_num']/data_dict['movie_frame_rate']
    firing_rate = data_dict['movie_total_ap_num']/movie_len
    
    pyr_idx = np.where(data_dict['ephys_cell_type']=='pyr')[0]
    int_idx = np.where(data_dict['ephys_cell_type']=='int')[0]
    ax_now.plot(firing_rate[pyr_idx],data_dict['movie_mean_cawave_snr_per_ap'][pyr_idx],'ko',alpha = .3)
    ax_now.plot(firing_rate[int_idx],data_dict['movie_mean_cawave_snr_per_ap'][int_idx],'ro',alpha = .3)
    if i == len(sensors)-1:
        ax_now.set_xlabel('Average firing rate in movie (Hz)')
    if i == 0:
        ax_now.set_title('SNR/AP estimate per movie')
    ax_now.set_ylabel(sensor)
    
    for cell_id_now in np.unique(unique_cell_id):
        idx = cell_id_now==unique_cell_id
        firing_rate_cell = np.mean(firing_rate[idx])
        firing_rate_cell_std = np.std(firing_rate[idx])/np.sqrt(sum(idx))
        snr_per_cell = np.mean(data_dict['movie_mean_cawave_snr_per_ap'][idx])
        snr_per_cell_std = np.std(data_dict['movie_mean_cawave_snr_per_ap'][idx])/np.sqrt(sum(idx))
        if data_dict['ephys_cell_type'][idx][0]=='pyr':
            ax_now_cell.plot(firing_rate_cell,snr_per_cell,'ko',alpha = .3)
            ax_now_cell.errorbar(firing_rate_cell,snr_per_cell,firing_rate_cell_std,snr_per_cell_std,color = 'black',alpha = .3)
        elif data_dict['ephys_cell_type'][idx][0]=='int':
            ax_now_cell.plot(firing_rate_cell,snr_per_cell,'ro',alpha = .3)
            ax_now_cell.errorbar(firing_rate_cell,snr_per_cell,firing_rate_cell_std,snr_per_cell_std,color = 'red',alpha = .3)
    if i == len(sensors)-1:
        ax_now_cell.set_xlabel('Average firing rate for a cell (Hz)')
    if i == 0:
        ax_now_cell.set_title('SNR/AP estimate per cell')
    #ax_now_cell.set_ylabel('SNR/AP estimate')
    
    
    #break

#%% injection number and titers
sensors = ['GCaMP7F','XCaMPgf','456','686','688']
fig_injections = plt.figure(figsize = [15,15])
ax_inj = fig_injections.add_subplot(111)
for sensor in sensors:
    subjects = imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)
    for subject in subjects:
        titers,volumes,dilutions = (lab.Virus()*lab.Surgery.VirusInjection()&subject).fetch('titer','volume','dilution')
# =============================================================================
#         titer = np.median(titers)/np.median(dilutions)
#         total_volume = np.sum(volumes)
# =============================================================================
        titer = titers/dilutions
        
        ax_inj.semilogy(volumes,titer,'o',color = sensor_colors[sensor],alpha = .05,markersize = 12)
        
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
recording_mode = 'current clamp'
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
leave_out_putative_interneurons = False
leave_out_putative_pyramidal_cells = False
ca_wave_parameters = {'only_manually_labelled_movies':False,
                      'max_baseline_dff':2,
                      'min_baseline_dff':-.5,# TODO check why baseline dff so low
                      'min_ap_snr':10,
                      'min_pre_isi':1,
                      'min_post_isi':1,
                      'max_sweep_ap_hw_std_per_median':.2,
                      'min_sweep_ap_snr_dv_median':10,
                      'gaussian_filter_sigma':10,#0,5,10,20,50
                      'neuropil_subtraction':'calculated',#'none','0.7','calculated','0.8'
                      'neuropil_number':1,#0,1 # 0 is 2 pixel , 1 is 4 pixels left out around the cell
                      'sensors': ['GCaMP7F','XCaMPgf','456','686','688'], # this determines the order of the sensors
                      'ephys_recording_mode':'current clamp',
                      'minimum_f0_value':20,
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
                           'recording_mode',
                           'cell_mean_firing_rate']
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
import plot.imaging_gt_plot as imaging_plot
imaging_plot.plot_r_neu_distribution(sensor_colors,ca_wave_parameters)
#%% number of ROIs in each movie
import plot.imaging_gt_plot as imaging_plot
imaging_plot.plot_number_of_rois(sensor_colors)
#%% ap parameters scatter plot, calcium sensor, median ap wave amplitude
import plot.imaging_gt_plot as imaging_plot
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
                   'normalize z by sensor':False}#'cawave_peak_amplitude_dff_median_per_ap'
imaging_plot.plot_ap_scatter_with_ophys_prop(ca_waves_dict,plot_parameters)
#%%

# =============================================================================
# firing_rate,cell_type,sensors = (imaging_gt.SessionCalciumSensor()*ephysanal_cell_attached.EphysCellType()*ephysanal_cell_attached.CellMeanFiringRate()).fetch('cell_mean_firing_rate','ephys_cell_type','session_calcium_sensor')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for i,sensor in enumerate(np.unique(sensors)):
#     idx = sensors == sensor
#     ax.plot(np.ones(sum(idx))*i,firing_rate[idx],'ko',alpha = .1)
# 
# =============================================================================


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
from utils import utils_plot
wave_parameters = {'ap_num_to_plot':2,#ap_num_to_plot
                   'min_ap_per_apnum':1}
ca_wave_parameters['min_pre_isi']=.1
ca_wave_parameters['min_post_isi']=.1
ca_traces_dict,ca_traces_dict_by_cell  = utils_plot.collect_ca_wave_traces( ca_wave_parameters,ca_waves_dict,wave_parameters,use_doublets = True   )
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
#%% anatomy
#% filtering images

       
        
# =============================================================================
#         break
#     break
# =============================================================================
#%% analyze dictionaries
    
# load json file
    
    
   
#%%
    
    

#%%   
