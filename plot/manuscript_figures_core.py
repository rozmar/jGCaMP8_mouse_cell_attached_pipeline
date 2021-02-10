import matplotlib.pyplot as plt
from utils import utils_plot
import numpy as np
import os 
import subprocess
import seaborn as sns

def plot_superresolution_grand_averages(superresolution_traces,plot_parameters):
    fig_grand_average_traces_pyr = plt.figure()
    ax_grand_average = fig_grand_average_traces_pyr.add_subplot(1,1,1)
    for sensor in plot_parameters['sensors']:
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
                        size = plot_parameters['yscalesize'],
                        conversion = 100,
                        unit = r'dF/F%',
                        color = 'black',
                        location = 'top right',
                        linewidth = 5)
    utils_plot.add_scalebar_to_fig(ax_grand_average,
                        axis = 'x',
                        size = plot_parameters['xscalesize'],
                        conversion = 1,
                        unit = r'ms',
                        color = 'black',
                        location = 'top right',
                        linewidth = 5)
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['fig_name']+'.{}')
    fig_grand_average_traces_pyr.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )


def plot_2ap_superresolution_traces(ca_traces_dict,superresolution_traces_1ap,plot_parameters):
    
    
    fig_freq = plt.figure()
    
    axs = list()
    fig_merged = plt.figure(figsize = [5,15])
    axs_merged=dict()
    
    
    axs_merged['slow_sensors']=fig_merged.add_subplot(2*len(plot_parameters['sensors_to_plot'])+1,1,1)
    
    for sensor in plot_parameters['slow_sensors']:
        x = superresolution_traces_1ap[sensor]['{}_time'.format(plot_parameters['trace_to_use'])]
        y = superresolution_traces_1ap[sensor]['{}_{}'.format(plot_parameters['trace_to_use'],plot_parameters['superresolution_function'])]
        if plot_parameters['normalize_to_max']:
            y = y/np.nanmax(y)
        color = plot_parameters['sensor_colors'][sensor]
        axs_merged['slow_sensors'].plot(x,y,'-',color = color,linewidth = plot_parameters['linewidth'],alpha = 0.9)
    axs_merged['slow_sensors'].set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])
    axs_merged['slow_sensors'].plot(0,-.1,'r|',markersize = 18)
    master_ax = axs_merged['slow_sensors']
    
    for sensor_i,sensor in enumerate(plot_parameters['sensors_to_plot']):
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
        if plot_parameters['normalize_to_max'] :
            axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(plot_parameters['sensors_to_plot'])+1,1,sensor_i*2+1+1,sharex = master_ax,sharey = master_ax)
        else:
            axs_merged[sensor]['ax_wave']=fig_merged.add_subplot(2*len(plot_parameters['sensors_to_plot'])+1,1,sensor_i*2+1+1,sharex = master_ax)
        #axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2,len(sensors_to_plot),sensor_i+1+len(sensors_to_plot),sharex = master_ax)
        axs_merged[sensor]['ax_hist']=fig_merged.add_subplot(2*len(plot_parameters['sensors_to_plot'])+1,1,sensor_i*2+2+1,sharex = master_ax)
        axs_merged[sensor]['ax_hist'].axis('off')
        axs_merged[sensor]['ax_wave'].axis('off')
        axs_merged[sensor]['ax_wave'].plot(0,-.1,'r|',markersize = 18)
        for step_i in np.arange(plot_parameters['time_range_step_number'])[::-1]:
            #try:
            if plot_parameters['exclude_outliers']:
                outlier_array = np.asarray(ca_traces_dict[sensor]['cawave_snr_list'])
    # =============================================================================
    #             percentiles = np.percentile(outlier_array,[25,75])
    #             non_outliers = (outlier_array>=percentiles[0]-np.abs(np.diff(percentiles))) & (outlier_array<=percentiles[1]+np.abs(np.diff(percentiles)))#-np.abs(np.diff(percentiles))
    # =============================================================================
                percentiles = np.percentile(outlier_array,[5,95])
                non_outliers = (outlier_array>=percentiles[0]) & (outlier_array<=percentiles[1])#-np.abs(np.diff(percentiles))
            else:
                non_outliers = cawave_ap_group_lengts>-1
            cawave_idx = [False]
            increase_high_step = -1
            while sum(cawave_idx)<plot_parameters['min_event_number'] and increase_high_step <= plot_parameters['time_range_step']*1000:
                increase_high_step+=1
                low = plot_parameters['time_range_step']*step_i + plot_parameters['time_range_offset']
                high = plot_parameters['time_range_step']*step_i + plot_parameters['time_range_window']+.001*increase_high_step + plot_parameters['time_range_offset']
                cawave_idx = (cawave_ap_group_lengts>=low)&(cawave_ap_group_lengts<high)&non_outliers
            if sum(cawave_idx)<plot_parameters['min_event_number']:
                continue
            x_conc = np.concatenate(np.asarray(ca_traces_dict[sensor]['cawave_time_list'])[cawave_idx])
            y_conc = np.concatenate(np.asarray(ca_traces_dict[sensor]['cawave_{}_list'.format(plot_parameters['trace_to_use'])])[cawave_idx])#cawave_dff_list#cawave_f_corr_list#cawave_f_corr_normalized_list
            needed_idx = (x_conc>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc<=plot_parameters['end_time']+plot_parameters['bin_size'])
            x_conc = x_conc[needed_idx]
            y_conc = y_conc[needed_idx]
            trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'all',dogaussian =True) 
            bin_centers = np.asarray(trace_dict['bin_centers'])
            trace = np.asarray(trace_dict['trace_{}'.format(plot_parameters['superresolution_function'])])
            trace_sd = np.asarray(trace_dict['trace_std'])
            if len(axs)==0:
            
                ax = fig_freq.add_subplot(plot_parameters['time_range_step_number'],len(plot_parameters['sensors_to_plot']),sensor_i+len(plot_parameters['sensors_to_plot'])*step_i+1)
            else:
                if len(sensor_axs) == 0:
                    ax = fig_freq.add_subplot(plot_parameters['time_range_step_number'],len(plot_parameters['sensors_to_plot']),sensor_i+len(plot_parameters['sensors_to_plot'])*step_i+1,sharex = axs[0])
                else:
                    ax = fig_freq.add_subplot(plot_parameters['time_range_step_number'],len(plot_parameters['sensors_to_plot']),sensor_i+len(plot_parameters['sensors_to_plot'])*step_i+1,sharex = axs[0],sharey = sensor_axs[0])
            ax_hist = ax.twinx()
            ax_hist.hist(np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])[cawave_idx]*1000,int(plot_parameters['time_range_step']*1000),alpha = .5)
            ax_hist.set_navigate(False)
            #break
            axs.append(ax)
            sensor_axs.append(ax)
            for x, y in zip(np.asarray(ca_traces_dict[sensor]['cawave_time_list'])[cawave_idx],np.asarray(ca_traces_dict[sensor]['cawave_{}_list'.format(plot_parameters['trace_to_use'])])[cawave_idx]):#cawave_dff_list#cawave_f_corr_list#cawave_f_corr_normalized_list
                needed_idx = (x>=plot_parameters['start_time'])&(x<=plot_parameters['end_time'])
                ax.plot(x[needed_idx],y[needed_idx],color =plot_parameters['sensor_colors'][sensor],alpha = .1 )
            ax.plot(bin_centers,trace,color =plot_parameters['sensor_colors'][sensor],alpha = 1,linewidth =2 )
            
            ax.fill_between(bin_centers, trace-trace_sd, trace+trace_sd,color =plot_parameters['sensor_colors'][sensor],alpha = .3)
            if sensor_i == len(plot_parameters['sensors_to_plot'])-1:
                ax_hist.set_ylabel('{} - {} ms'.format(int(low*1000),int(high*1000)))
                
            #alpha =  1-step_i*.1 #.1+step_i*.1
            color =utils_plot.lighten_color(plot_parameters['sensor_colors'][sensor], amount=1-step_i*.1)
            axs_merged[sensor]['ax_hist'].hist(np.asarray(ca_traces_dict[sensor]['cawave_ap_group_length_list'])[cawave_idx]*1000,int(plot_parameters['time_range_step']*1000),color =color)
            axs_merged[sensor]['ax_wave'].plot(bin_centers,trace,color =color,linewidth =2 )#
    ax.set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])  
    if plot_parameters['normalize_to_max']:
        axs_merged['slow_sensors'].set_ylim([-.1,1.1])  
    axs_merged[sensor]['ax_wave'].set_xlim([plot_parameters['start_time'],plot_parameters['end_time']])  
    utils_plot.add_scalebar_to_fig(axs_merged['slow_sensors'],
                                    axis = 'x',
                                    size = 10,
                                    conversion = 1,
                                    unit = r'ms',
                                    color = 'black',
                                    location = 'bottom right',
                                    linewidth = 5)
    
    figfile = os.path.join(plot_parameters['figures_dir'],'fig_2ap_kinetics.{}')
    fig_merged.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )
    

def plot_cell_wise_scatter_plots(ca_wave_parameters,ca_waves_dict,superresolution_traces,wave_parameters,plot_parameters):
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
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['plot_name']+'.{}')
    fig_scatter_plots_1ap.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )    
        