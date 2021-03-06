from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils.utils_plot as utils_plot
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
#%%



    #%%
def plot_number_of_rois(sensor_colors):
    #%%
    sensors = list(sensor_colors.keys())
    session_roi_nums_dict = dict()
    movie_roi_nums_dict = dict()
    movie_depths_dict = dict()
    for sensor_i,sensor in enumerate(sensors):
        session_roi_nums_dict[sensor]=list()
        movie_roi_nums_dict[sensor]=list()
        movie_depths_dict[sensor] = list()
        sessions = imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)
        for session in sessions:
            session_roi_nums_dict[sensor].append(len(imaging.ROI()*imaging_gt.SessionCalciumSensor()&session))
            movies = imaging.Movie()&session
            for movie in movies:
                movie_roi_nums_dict[sensor].append(len(imaging.ROI()*imaging_gt.SessionCalciumSensor()&movie))
                movie_depths_dict[sensor].append((imaging_gt.MovieDepth()&movie).fetch1('movie_depth'))
    #%  ROI number in movies, etc
    
    depth_bin_step = 1 
    depth_bin_size = 10
    depth_bin_range = np.arange(0,300,depth_bin_step)
    fig_roi_num = plt.figure(figsize = [25,20])
    ax_roi_num = fig_roi_num.add_subplot(2,2,1)
    ax_roi_num_per_subject = ax_roi_num.twinx() 
    x_roinum_swarm_list = list()
    y_roinum_swarm_list = list()
    
    ax_movie_num = fig_roi_num.add_subplot(2,2,2)
    ax_roi_num_per_movie = ax_movie_num.twinx() 
    x_roinum_swarm__movie_list = list()
    y_roinum_swarm__movie_list = list()
    
    ax_depth_dependence = fig_roi_num.add_subplot(2,2,3)
    
    ax_depth_hist = fig_roi_num.add_subplot(2,2,4)
    ax_depth_hist_cum = ax_depth_hist.twinx()
    depth_hist_range = np.arange(0,300+depth_bin_size,depth_bin_size)
    depth_cum_hist_range = np.arange(0,300+1,1)
    depths_all_cum = list()
    depths_all = list()
    bar_width = depth_bin_size*.9#/len(ca_wave_parameters['sensors'])*.9
    for sensor_i,sensor in enumerate(sensors):
        ax_roi_num.bar(sensor_i,sum(session_roi_nums_dict[sensor]),edgecolor = sensor_colors[sensor],fill=False,linewidth = 2)
        x_roinum_swarm_list.append(np.ones(len(session_roi_nums_dict[sensor]))*sensor_i)
        y_roinum_swarm_list.append(session_roi_nums_dict[sensor])
        
        ax_movie_num.bar(sensor_i,sum(movie_roi_nums_dict[sensor]),edgecolor = sensor_colors[sensor],fill=False,linewidth = 2)
        x_roinum_swarm__movie_list.append(np.ones(len(movie_roi_nums_dict[sensor]))*sensor_i)
        y_roinum_swarm__movie_list.append(movie_roi_nums_dict[sensor])
        
        ax_depth_dependence.plot(movie_depths_dict[sensor],movie_roi_nums_dict[sensor],'o',color = sensor_colors[sensor],alpha = .3)
        mean_roi_nums = list()
        sd_roi_nums = list()
        for bin_start in depth_bin_range:
            idx = (movie_depths_dict[sensor]>=bin_start)&(movie_depths_dict[sensor]<bin_start+depth_bin_size)
            if sum(idx)>0:
                mean_roi_nums.append(np.mean(np.asarray(movie_roi_nums_dict[sensor])[idx]))
                sd_roi_nums.append(np.std(np.asarray(movie_roi_nums_dict[sensor])[idx]))
            else:
                mean_roi_nums.append(np.nan)
                sd_roi_nums.append(np.nan)
        mean_roi_nums=np.asarray(mean_roi_nums)
        sd_roi_nums = np.asarray(mean_roi_nums)
        bin_needed = np.isnan(mean_roi_nums)==False
        ax_depth_dependence.plot(depth_bin_range[bin_needed]+depth_bin_size/2,mean_roi_nums[bin_needed],'-',color = sensor_colors[sensor],linewidth = 4)
        #ax_depth_dependence.errorbar(depth_bin_range[bin_needed]+depth_bin_size/2,mean_roi_nums[bin_needed],sd_roi_nums[bin_needed],color = sensor_colors[sensor])
        
        y_depth,x = np.histogram(movie_depths_dict[sensor],depth_hist_range)
        y_cum,x_cum = np.histogram(movie_depths_dict[sensor],depth_cum_hist_range)
        depths_all_cum.append(y_cum)
        depths_all.append(y_depth)
        
        
        #
    depth_hist_bins = np.mean([depth_hist_range[:-1],depth_hist_range[1:]],0)
    depth_cum_hist_bins = np.mean([depth_cum_hist_range[:-1],depth_cum_hist_range[1:]],0)
    bottoms = np.zeros(len(depth_hist_bins))
    for i,sensor in enumerate(sensor_colors.keys()):
        ax_depth_hist.bar(depth_hist_bins, depths_all[i], bar_width,bottom=bottoms, color= sensor_colors[sensor],alpha = .3)#-plot_parameters['event_histogram_step']/2+bar_width*i
        ax_depth_hist_cum.plot(depth_cum_hist_bins, np.cumsum(depths_all_cum[i])/sum(depths_all_cum[i]), color= sensor_colors[sensor],linewidth = 2)#-plot_parameters['event_histogram_step']/2+bar_width*i
        bottoms = depths_all[i]+bottoms 
    
    
    sns.swarmplot(x =np.concatenate(x_roinum_swarm_list),y = np.concatenate(y_roinum_swarm_list),color='black',ax = ax_roi_num_per_subject)     
    sns.swarmplot(x =np.concatenate(x_roinum_swarm__movie_list),y = np.concatenate(y_roinum_swarm__movie_list),color='black',ax = ax_roi_num_per_movie,size = 2)     
    
    ax_roi_num.set_xticks(np.arange(len(sensors)))
    ax_roi_num.set_xticklabels(sensors)  
    ax_roi_num.set_ylabel('Total number of ROIs')
    ax_roi_num_per_subject.set_ylabel('Number of ROIs per subject')
    ax_roi_num_per_subject.set_ylim([0,ax_roi_num_per_subject.get_ylim()[1]])
    
    ax_movie_num.set_xticks(np.arange(len(sensors)))
    ax_movie_num.set_xticklabels(sensors)  
    ax_movie_num.set_ylabel('Total number of ROIs')
    ax_roi_num_per_movie.set_ylabel('Number of ROIs per movie')
    ax_roi_num_per_movie.set_ylim([0,ax_roi_num_per_movie.get_ylim()[1]])
    
    ax_depth_dependence.set_xlabel('depth from pia (microns)')
    ax_depth_dependence.set_ylabel('number of active ROIs per movie')
    #%%



def ca_wave_properties_scatter_onpick(event):
    cax_limits = [0,99] # mean image grayscale
    axesdata = event.canvas.axes_dict
    data = axesdata['data_dict']
    event_idx = event.ind[0]
    key = axesdata['key_dict']['apgroup_key'][event_idx]
    
    
    if event.mouseevent.button == 3:
        plottrace = False
        for event_idx_now in event.ind:
            if event_idx_now in event.canvas.idxsofar:
                idxsofar = event.canvas.idxsofar[event.canvas.idxsofar != event_idx_now]
                break
    else:
        plottrace = True
        if event_idx not in event.canvas.idxsofar:
            idxsofar = np.concatenate([event.canvas.idxsofar,[event_idx]])
    
    
# =============================================================================
#     if event_idx in event.canvas.idxsofar and event.mouseevent.button == 3:
#         if event.mouseevent.button ==1:
#             plottrace = True
#         else:
#             plottrace = False
#             idxsofar = event.canvas.idxsofar[event.canvas.idxsofar != event_idx]
#     else:
#         if event.mouseevent.button ==1:
#             plottrace = True
#             idxsofar = np.concatenate([event.canvas.idxsofar,[event_idx]])
#         else:
#             plottrace = False
# =============================================================================
    
    for pointsvar in axesdata['points_dict'].keys(): # going throught the plots
        try:
            for sel in axesdata['points_dict'][pointsvar].selected: # going through the already selected points
                sel.remove()         # remove points from plots
        except:
            pass
     
    for pointsvar in axesdata['points_dict'].keys(): # going throught the plots
        pointslist = list()
        axesdata[pointsvar].set_prop_cycle(None)
        for cell_idx in idxsofar:
            sel, = axesdata[pointsvar].plot(np.concatenate(data['{}_x'.format(pointsvar)])[cell_idx],np.concatenate(data['{}_y'.format(pointsvar)])[cell_idx],'x',ms=14,markeredgewidth=4)
            pointslist.append(sel)
        axesdata['points_dict'][pointsvar].selected = pointslist
    #%
    redo_figure = False
    try:
        cawavefig_dict = event.canvas.cawavefig_dict
        if plt.fignum_exists(cawavefig_dict['fig_cawaves'].number):
           redo_figure = False
        else:
           redo_figure = True
    except:
        redo_figure = True
    if redo_figure:
        cawavefig_dict = dict()
        cawavefig_dict['fig_cawaves'] = plt.figure()
        grid = plt.GridSpec(4, 1, wspace=0.4, hspace=0.3)
        cawavefig_dict['ax_ophys_raw'] = cawavefig_dict['fig_cawaves'].add_subplot(grid[0, 0])
        cawavefig_dict['ax_ophys_raw_fneu'] = cawavefig_dict['ax_ophys_raw'].twinx()
        cawavefig_dict['ax_ophys_raw'].get_shared_y_axes().join(cawavefig_dict['ax_ophys_raw'] , cawavefig_dict['ax_ophys_raw_fneu'])
        cawavefig_dict['ax_ophys_corrected'] = cawavefig_dict['fig_cawaves'].add_subplot(grid[1, 0],sharex = cawavefig_dict['ax_ophys_raw'])
        cawavefig_dict['ax_ophys_dff'] = cawavefig_dict['fig_cawaves'].add_subplot(grid[2, 0],sharex = cawavefig_dict['ax_ophys_raw'])
        cawavefig_dict['ax_ephys'] = cawavefig_dict['fig_cawaves'].add_subplot(grid[3, 0],sharex = cawavefig_dict['ax_ophys_raw'])
        
        cawavefig_dict['ax_ophys_raw'].set_ylabel('raw pixel values')
        cawavefig_dict['ax_ophys_corrected'].set_ylabel('neuropil corrected pixel values')
        cawavefig_dict['ax_ophys_dff'].set_ylabel('dF/F')
        cawavefig_dict['ax_ephys'].set_ylabel('normalized voltage/current')
        cawavefig_dict['ax_ephys'].set_xlabel('Time from AP (ms)')
    
    cawavefig_dict['ax_ophys_raw'].cla()
    cawavefig_dict['ax_ophys_raw'].set_prop_cycle(None)
    cawavefig_dict['ax_ophys_raw_fneu'].cla()
    cawavefig_dict['ax_ophys_raw_fneu'].set_prop_cycle(None)
    cawavefig_dict['ax_ophys_corrected'].cla()
    cawavefig_dict['ax_ophys_corrected'].set_prop_cycle(None)
    cawavefig_dict['ax_ophys_dff'].cla()
    cawavefig_dict['ax_ophys_dff'].set_prop_cycle(None)
    cawavefig_dict['ax_ephys'].cla()
    cawavefig_dict['ax_ephys'].set_prop_cycle(None)
    
    for cell_idx in idxsofar:
        key = axesdata['key_dict']['apgroup_key'][cell_idx]
        cawave_f,cawave_time,cawave_fneu,cawave_rneu,cawave_baseline_f= (imaging_gt.CalciumWave.CalciumWaveProperties()*imaging_gt.CalciumWave.CalciumWaveNeuropil()*imaging_gt.CalciumWave()&key).fetch1('cawave_f','cawave_time','cawave_fneu','cawave_rneu','cawave_baseline_f')
        frame_rate = 1/np.median(np.diff(cawave_time))
        if key['gaussian_filter_sigma']>0:
            cawave_f = utils_plot.gaussFilter(cawave_f,frame_rate,key['gaussian_filter_sigma'])
            cawave_fneu = utils_plot.gaussFilter(cawave_fneu,frame_rate,key['gaussian_filter_sigma'])
        cawave_f_corr = cawave_f-cawave_fneu*cawave_rneu
        cawave_dff = (cawave_f_corr-cawave_baseline_f)/cawave_baseline_f
        ap_group_trace, ap_group_trace_peak_idx,sample_rate,sweep_start_time =  (ephys_cell_attached.Sweep()*ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.APGroupTrace()&key).fetch1('ap_group_trace','ap_group_trace_peak_idx','sample_rate','sweep_start_time')
        ap_group_trace_time = np.arange(-ap_group_trace_peak_idx,len(ap_group_trace)-ap_group_trace_peak_idx)/sample_rate*1000
        
        
        cawavefig_dict['ax_ophys_raw'].plot(cawave_time,cawave_f)
        cawavefig_dict['ax_ophys_raw_fneu'].plot(cawave_time,cawave_fneu,alpha = .4)
        cawavefig_dict['ax_ophys_corrected'].plot(cawave_time,cawave_f_corr)
        cawavefig_dict['ax_ophys_dff'].plot(cawave_time,cawave_dff)
        cawavefig_dict['ax_ephys'].plot(ap_group_trace_time,ap_group_trace)
        
    #axesdata['ax_trace'].set_xlabel('time (ms)')
    
    
    #%
    #key = {'subject_id': 479115, 'session': 1, 'cell_number': 8, 'sweep_number': 4, 'ap_group_number': 50, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': 'calculated', 'neuropil_number': 0}
    if plottrace:
        key = axesdata['key_dict']['apgroup_key'][event_idx]
        #%%
        #key = {'subject_id': 472181, 'session': 1, 'cell_number': 7, 'sweep_number': 2, 'ap_group_number': 112, 'gaussian_filter_sigma': 0, 'neuropil_subtraction': 'calculated', 'neuropil_number': 1}
        #sweep_start_time, ap_group_start_time = (ephys_cell_attached.Sweep()*ephysanal_cell_attached.APGroup()&key).fetch1('sweep_start_time','ap_group_start_time')
        #ax_ephys_trace.set_xlim([float(ap_group_start_time-sweep_start_time)-.025,float(ap_group_start_time-sweep_start_time)+.05])
        ap_max_times = np.asarray((ephysanal_cell_attached.ActionPotential()&key).fetch('ap_max_time'),float)
        
        fig_cawave = plt.figure()
        grid = plt.GridSpec(6, 1, wspace=0.4, hspace=0.3)
        ax_ophys_raw_trace = fig_cawave.add_subplot(grid[2, 0])
        ax_ophys_corrected_trace = fig_cawave.add_subplot(grid[3, 0],sharex = ax_ophys_raw_trace)
        ax_ophys_dff_trace = fig_cawave.add_subplot(grid[4, 0],sharex = ax_ophys_raw_trace)
        ax_ephys_trace = fig_cawave.add_subplot(grid[5, 0],sharex = ax_ophys_raw_trace)
        ax_mean_image = fig_cawave.add_subplot(grid[:2, 0])
        
        ax_mean_image.set_title(key)
        ax_ophys_raw_trace.set_ylabel('raw pixel values')
        ax_ophys_corrected_trace.set_ylabel('neuropil corrected pixel values')
        ax_ophys_dff_trace.set_ylabel('dF/F')
        ax_ephys_trace.set_ylabel('normalized voltage/current')
        ax_ephys_trace.set_xlabel('Time (s)')
        
        #%
        cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()&key).fetch1('cell_recording_start', 'session_time')
        
        cawave_rneu = (imaging_gt.CalciumWave.CalciumWaveProperties()&key).fetch1('cawave_rneu')
        roi_f,neuropil_f,roi_time_offset,frame_times = (imaging.MovieFrameTimes()*imaging.ROI()*imaging.ROINeuropilTrace()*imaging.ROITrace()*imaging_gt.ROISweepCorrespondance()&key&'channel_number = 1').fetch1('roi_f','neuropil_f','roi_time_offset','frame_times')
        
        frame_times = frame_times+ roi_time_offset
        frame_rate = 1/np.median(np.diff(frame_times))/1000
        if key['gaussian_filter_sigma']>0:
            roi_f = utils_plot.gaussFilter(roi_f,frame_rate,key['gaussian_filter_sigma'])
            neuropil_f = utils_plot.gaussFilter(neuropil_f,frame_rate,key['gaussian_filter_sigma'])        
        roi_f_corr = roi_f-neuropil_f*cawave_rneu
        f0 = np.percentile(roi_f_corr,10)
        roi_dff = (roi_f_corr-f0)/f0
        
        sweep_start_time =  (ephys_cell_attached.Sweep()&key).fetch1('sweep_start_time')
        response_trace, sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()&key).fetch1('response_trace','sample_rate')
        ephys_time = np.arange(len(response_trace))/sample_rate
        frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))

#%
        alpha = .3
        #registered_movie_mean_image_ch2 = (imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()*imaging.ROI()*imaging.ROINeuropil()&key&'channel_number = 2').fetch1('registered_movie_mean_image')
        registered_movie_mean_image_ch1,roi_xpix,roi_ypix,neuropil_xpix,neuropil_ypix,movie_pixel_size = (imaging_gt.ROISweepCorrespondance()*imaging.RegisteredMovieImage()*imaging.ROI()*imaging.ROINeuropil()*imaging.Movie()&key&'channel_number = 1').fetch1('registered_movie_mean_image','roi_xpix','roi_ypix','neuropil_xpix','neuropil_ypix','movie_pixel_size')
        ch_1_image = ax_mean_image.imshow(registered_movie_mean_image_ch1,cmap = 'gray', extent=[0,registered_movie_mean_image_ch1.shape[1]*float(movie_pixel_size),0,registered_movie_mean_image_ch1.shape[0]*float(movie_pixel_size)], aspect='auto')
        clim = np.percentile(registered_movie_mean_image_ch1.flatten(),cax_limits)
        ch_1_image.set_clim(clim)
        ROI = np.zeros_like(registered_movie_mean_image_ch1)*np.nan
        ROI[roi_ypix,roi_xpix] = 1
        ROIneuropil = np.zeros_like(registered_movie_mean_image_ch1)*np.nan
        ROIneuropil[neuropil_ypix,neuropil_xpix] = 1
        ax_mean_image.imshow(ROI,cmap = 'Greens', extent=[0,registered_movie_mean_image_ch1.shape[1]*float(movie_pixel_size),0,registered_movie_mean_image_ch1.shape[0]*float(movie_pixel_size)], aspect='auto',alpha = alpha,vmin = 0, vmax = 1)
        ax_mean_image.imshow(ROIneuropil,cmap = 'Reds', extent=[0,registered_movie_mean_image_ch1.shape[1]*float(movie_pixel_size),0,registered_movie_mean_image_ch1.shape[0]*float(movie_pixel_size)], aspect='auto',alpha = alpha,vmin = 0, vmax = 1)
   #%
        ax_ophys_raw_trace.plot(frame_times,roi_f,'g-')
        ax_ophys_raw_trace.plot(frame_times,neuropil_f,'r-',alpha = .5)
        ax_ophys_corrected_trace.plot(frame_times,roi_f_corr,'g-')
        ax_ophys_dff_trace.plot(frame_times,roi_dff,'g-')
        ax_ophys_dff_trace.plot(ap_max_times,np.zeros(len(ap_max_times))-.2,'r|',markersize = 8)
        ax_ephys_trace.plot(ephys_time,response_trace,'k-')
        #%%
    event.canvas.idxsofar = idxsofar
    event.canvas.cawavefig_dict = cawavefig_dict
    cawavefig_dict['fig_cawaves'].canvas.draw()
    event.canvas.draw()

    #%%
def scatter_plot_all_event_properties(ca_waves_dict,ca_wave_parameters,plot_parameters):
    #%%
    correct_f0_with_power_depth = plot_parameters['correct_f0_with_power_depth']
    alpha = plot_parameters['scatter_plot_alpha']
    sensor_colors=plot_parameters['sensor_colors']
    key= {'neuropil_number': ca_wave_parameters['neuropil_number'],
      'neuropil_subtraction': ca_wave_parameters['neuropil_subtraction'],#'none','0.7','calculated'
      'session_calcium_sensor':None,#'GCaMP7F',#'GCaMP7F','XCaMPgf','456','686','688'
      'gaussian_filter_sigma' : ca_wave_parameters['gaussian_filter_sigma'],
      'channel_number':1,
      }
    legend_elements = list()
    for sensor in ca_wave_parameters['sensors']:
        legend_elements.append(Line2D([0], [0], marker='o', color=sensor_colors[sensor], 
                                      label=sensor,markerfacecolor=sensor_colors[sensor], markersize=8))
        
    fig=plt.figure()
    axes_dict = dict()
    points_dict = dict()
    data_dict = dict()
    axes_dict['ax_1ap_snr_amplitude_cell'] = fig.add_subplot(4,4,1)
    axes_dict['ax_1ap_dff_amplitude_cell'] = fig.add_subplot(4,4,2,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_1ap_f_amplitude_cell'] = fig.add_subplot(4,4,3,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_1ap_f0_cell'] = fig.add_subplot(4,4,5,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    
    #axes_dict['ax_1ap_baseline_noise_cell'] = fig.add_subplot(4,4,6,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_f0_percentile_cell'] = fig.add_subplot(4,4,6,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_pixel_num_cell'] = fig.add_subplot(4,4,7,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_movie_power'] = fig.add_subplot(4,4,9,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_cell_depth'] = fig.add_subplot(4,4,10,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    
    axes_dict['ax_expression_time'] = fig.add_subplot(4,4,11,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_event_num_hist'] = fig.add_subplot(4,4,14)
    axes_dict['ax_event_num_cum_hist'] = axes_dict['ax_event_num_hist'].twinx()
    axes_dict['ax_neuropil'] = fig.add_subplot(4,4,15,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    #axes_dict['ax_f0_power'] = fig.add_subplot(4,4,15)
    
    axes_dict['ax_ap_peak_to_trough_time_median'] = fig.add_subplot(4,4,4,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_ap_halfwidth_median'] = fig.add_subplot(4,4,8,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_ap_rise_time_median'] = fig.add_subplot(4,4,12,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    axes_dict['ax_ap_ahp_amplitude_median'] = fig.add_subplot(4,4,16,sharex = axes_dict['ax_1ap_snr_amplitude_cell'])
    ca_wave_parameters['sensor']='all'
    
    axes_dict['ax_ap_peak_to_trough_time_median'].set_ylabel('AP peak to trough (ms)')
    axes_dict['ax_ap_halfwidth_median'].set_ylabel('AP half width (ms)')
    axes_dict['ax_ap_rise_time_median'].set_ylabel('AP rise time (ms)')
    axes_dict['ax_ap_ahp_amplitude_median'].set_ylabel('AP ahp/ap amplitude')
    
    axes_dict['ax_1ap_dff_amplitude_cell'].set_title(ca_wave_parameters)
    axes_dict['ax_pixel_num_cell'].set_ylabel('ROI pixel number')
    axes_dict['ax_1ap_dff_amplitude_cell'].set_ylabel('peak amplitude (dF/F)')
    if correct_f0_with_power_depth:
        axes_dict['ax_1ap_f0_cell'].set_ylabel('baseline fluorescence (F) - corrected**')
    else:
        axes_dict['ax_1ap_f0_cell'].set_ylabel('baseline fluorescence (F)')
    axes_dict['ax_1ap_f_amplitude_cell'].set_ylabel('peak amplitude (F)')
    axes_dict['ax_1ap_snr_amplitude_cell'].set_ylabel('peak amplitude SNR')
    #axes_dict['ax_1ap_baseline_noise_cell'].set_ylabel('baseline noise (std(F))')
    axes_dict['ax_f0_percentile_cell'].set_ylabel('percentile of F0 in session')
    axes_dict['ax_1ap_baseline_f_vs_noise_cell'] = fig.add_subplot(4,4,13)
    axes_dict['ax_1ap_baseline_f_vs_noise_cell'].set_xlabel('baseline fluorescence (F)')
    axes_dict['ax_1ap_baseline_f_vs_noise_cell'].set_ylabel('baseline noise (std(F))')
    axes_dict['ax_expression_time'].set_ylabel('expression time (days)')
    axes_dict['ax_cell_depth'].set_ylabel('imaging depth (microns)')
    axes_dict['ax_event_num_hist'].set_xlabel('Number of events per cell')
    axes_dict['ax_event_num_hist'].set_ylabel('Number of cells')
    axes_dict['ax_movie_power'].set_ylabel('laser power after objective (mW)')
    axes_dict['ax_neuropil'].set_ylabel('r_neuropil - calculated')
    #axes_dict['ax_f0_power'].set_ylabel('baseline fluorescence (F)')#
    #axes_dict['ax_f0_power'].set_xlabel('laser power after objective (mW)')#
    
    
    data_dict = dict()
    key_dict = dict()
    key_dict['apgroup_key']=list()
    for axname in axes_dict.keys():
        data_dict['{}_x'.format(axname)]=list()
        data_dict['{}_y'.format(axname)]=list()
    
    cell_index = 0
    cell_event_number_all = list()
    cell_event_number_by_sensor = dict()
    for sensor in ca_wave_parameters['sensors']:
        key['session_calcium_sensor'] = sensor
        cellwave_by_cell_list = np.asarray(ca_waves_dict[sensor])
        cellvals = list()
        cell_event_number = list()
        for cell_data in cellwave_by_cell_list: # set order
            apidx = cell_data['ap_group_ap_num_mean_per_ap']==plot_parameters['ap_num_to_plot']
            cell_event_number.extend(cell_data['event_num_per_ap'][apidx])
            if cell_data['event_num_per_ap'][apidx]>=plot_parameters['min_ap_per_apnum']:
                cellvals.append(cell_data[plot_parameters['order_cells_by']][apidx][0])#cawave_snr_median_per_ap#cawave_peak_amplitude_dff_median_per_ap
            else:
                cellvals.append(np.nan)
        cell_event_number_all.extend(cell_event_number)
        cell_event_number_by_sensor[sensor] = cell_event_number
        order = np.argsort(cellvals)
        cellwave_by_cell_list = cellwave_by_cell_list[order]
        for cell_data in cellwave_by_cell_list: # plot
            apidx = cell_data['ap_group_ap_num_mean_per_ap']==plot_parameters['ap_num_to_plot']
            if cell_data['event_num_per_ap'][apidx] < plot_parameters['min_ap_per_apnum']:
                continue
            cell_index += 1
            single_event_idx = cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']
            for subject_id,session,cell_number,sweep_number,ap_group_number in zip(cell_data['subject_id'][single_event_idx],
                                                                                   cell_data['session'][single_event_idx],
                                                                                   cell_data['cell_number'][single_event_idx],
                                                                                   cell_data['sweep_number'][single_event_idx],
                                                                                   cell_data['ap_group_number'][single_event_idx]):
                key_now = {'subject_id':subject_id,
                           'session':session,
                           'cell_number':cell_number,
                           'sweep_number':sweep_number,
                           'ap_group_number':ap_group_number,
                           'gaussian_filter_sigma':ca_wave_parameters['gaussian_filter_sigma'],
                           'neuropil_subtraction':ca_wave_parameters['neuropil_subtraction'],
                           'neuropil_number':ca_wave_parameters['neuropil_number']}
                key_dict['apgroup_key'].append(key_now)
            x = np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index
            y = cell_data['cawave_peak_amplitude_dff'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_1ap_dff_amplitude_cell'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_1ap_dff_amplitude_cell_x'].append(x)
            data_dict['ax_1ap_dff_amplitude_cell_y'].append(y)
            y = cell_data['cawave_peak_amplitude_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_1ap_f_amplitude_cell'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_1ap_f_amplitude_cell_x'].append(x)
            data_dict['ax_1ap_f_amplitude_cell_y'].append(y)

            if correct_f0_with_power_depth:
# =============================================================================
#                 f = cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
#                 power =  cell_data['movie_power'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
#                 depth = cell_data['movie_hmotors_sample_z'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
#                 y =f/(power**2)*depth**2
# =============================================================================
                y = cell_data['roi_f0_percentile_value'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            else:
                y = cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_1ap_f0_cell'].semilogy(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_1ap_f0_cell_x'].append(x)
            data_dict['ax_1ap_f0_cell_y'].append(y)
            y = cell_data['cawave_snr'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_1ap_snr_amplitude_cell'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_1ap_snr_amplitude_cell_x'].append(x)
            data_dict['ax_1ap_snr_amplitude_cell_y'].append(y)
# =============================================================================
#             y = cell_data['cawave_baseline_f_std'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
#             axes_dict['ax_1ap_baseline_noise_cell'].semilogy(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
#             data_dict['ax_1ap_baseline_noise_cell_x'].append(x)
#             data_dict['ax_1ap_baseline_noise_cell_y'].append(y)
# =============================================================================
            y = cell_data['roi_f0_percentile'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_f0_percentile_cell'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_f0_percentile_cell_x'].append(x)
            data_dict['ax_f0_percentile_cell_y'].append(y)
            y = cell_data['roi_pixel_num'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_pixel_num_cell'].semilogy(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)##roi_dwell_time_multiplier
            data_dict['ax_pixel_num_cell_x'].append(x)
            data_dict['ax_pixel_num_cell_y'].append(y)
            y = cell_data['movie_power'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_movie_power'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_movie_power_x'].append(x)
            data_dict['ax_movie_power_y'].append(y)
            y = cell_data['session_sensor_expression_time'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_expression_time'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_expression_time_x'].append(x)
            data_dict['ax_expression_time_y'].append(y)
            y = cell_data['cawave_rneu'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_neuropil'].semilogy(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_neuropil_x'].append(x)
            data_dict['ax_neuropil_y'].append(y)
            
            y = cell_data['ap_peak_to_trough_time_median'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_ap_peak_to_trough_time_median'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_ap_peak_to_trough_time_median_x'].append(x)
            data_dict['ax_ap_peak_to_trough_time_median_y'].append(y)
            
            y = cell_data['ap_ahp_amplitude_median'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_ap_ahp_amplitude_median'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_ap_ahp_amplitude_median_x'].append(x)
            data_dict['ax_ap_ahp_amplitude_median_y'].append(y)
            
            y = cell_data['ap_rise_time_median'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_ap_rise_time_median'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_ap_rise_time_median_x'].append(x)
            data_dict['ax_ap_rise_time_median_y'].append(y)
            
            y = cell_data['ap_halfwidth_median'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_ap_halfwidth_median'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_ap_halfwidth_median_x'].append(x)
            data_dict['ax_ap_halfwidth_median_y'].append(y)
            try:
                #ax_cell_depth.plot(cell_index,cell_data['cell_key']['depth'],'o',ms=8,color = sensor_colors[sensor])
                y = cell_data['movie_hmotors_sample_z'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
                axes_dict['ax_cell_depth'].plot(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
                data_dict['ax_cell_depth_x'].append(x)
                data_dict['ax_cell_depth_y'].append(y)
            except:
                pass
            x = cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            y = cell_data['cawave_baseline_f_std'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            axes_dict['ax_1ap_baseline_f_vs_noise_cell'].plot(x,y ,     'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
            data_dict['ax_1ap_baseline_f_vs_noise_cell_x'].append(x)
            data_dict['ax_1ap_baseline_f_vs_noise_cell_y'].append(y)
            
            x = cell_data['movie_power'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
            y = cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']]
# =============================================================================
#             axes_dict['ax_f0_power'].semilogy(x,y,'o',ms=plot_parameters['scatter_marker_size'],color = sensor_colors[sensor],alpha = alpha)
#             data_dict['ax_f0_power_x'].append(x)
#             data_dict['ax_f0_power_y'].append(y)
# =============================================================================
            
            
            
    axes_dict['ax_cell_depth'].invert_yaxis()
    axes_dict['ax_cell_depth'].set_ylim([axes_dict['ax_cell_depth'].get_ylim()[0],0])
    
    #plot_parameters['event_histogram_step'] =5
    bar_width = plot_parameters['event_histogram_step']*.9#/len(ca_wave_parameters['sensors'])*.9
    hist_range = np.arange(0,np.max(cell_event_number_all)+plot_parameters['event_histogram_step'],plot_parameters['event_histogram_step'])
    cum_hist_range = np.arange(0,np.max(cell_event_number_all)+1,1)
    ys = list()
    cum_ys = list()
    for sensor in ca_wave_parameters['sensors']:
        y,x = np.histogram(cell_event_number_by_sensor[sensor],hist_range)
        y_cum,x_cum = np.histogram(cell_event_number_by_sensor[sensor],cum_hist_range)
        cum_ys.append(y_cum)
        ys.append(y)
    hist_bins = np.mean([hist_range[:-1],hist_range[1:]],0)
    cum_hist_bins = np.mean([cum_hist_range[:-1],cum_hist_range[1:]],0)
    bottoms = np.zeros(len(hist_bins))
    for i,sensor in enumerate(ca_wave_parameters['sensors']):
        axes_dict['ax_event_num_hist'].bar(hist_bins, ys[i], bar_width,bottom=bottoms, color= sensor_colors[sensor],alpha = .5)#-plot_parameters['event_histogram_step']/2+bar_width*i
        axes_dict['ax_event_num_cum_hist'].plot(cum_hist_bins, np.cumsum(cum_ys[i])/sum(cum_ys[i]), color= sensor_colors[sensor],linewidth = 2)#-plot_parameters['event_histogram_step']/2+bar_width*i
        bottoms = ys[i]+bottoms
    points_dict = dict()
    for ax_name in axes_dict.keys():
        try:
            if axes_dict[ax_name].get_yscale() == 'log':
                points_dict[ax_name], = axes_dict[ax_name].semilogy(np.concatenate(data_dict['{}_x'.format(ax_name)]),np.concatenate(data_dict['{}_y'.format(ax_name)]),'ko',ms = 1, alpha = 0, picker = 5)
            else:
                points_dict[ax_name], = axes_dict[ax_name].plot(np.concatenate(data_dict['{}_x'.format(ax_name)]),np.concatenate(data_dict['{}_y'.format(ax_name)]),'ko',ms = 1, alpha = 0, picker = 5)
        except:
            #print(ax_name)
            pass
    axes_dict['points_dict'] = points_dict
    axes_dict['data_dict'] = data_dict
    axes_dict['key_dict'] = key_dict
    axes_dict['ca_wave_parameters'] = ca_wave_parameters
    axes_dict['ax_event_num_hist'].legend(handles=legend_elements)
    fig.canvas.axes_dict=axes_dict
    fig.canvas.idxsofar = np.asarray([],int)
    cid = fig.canvas.mpl_connect('pick_event', ca_wave_properties_scatter_onpick)
    
    plt.show()
    
#%%
def plot_all_traces(ca_traces_dict,ca_traces_dict_by_cell,ca_wave_parameters,plot_parameters):
    fig = plt.figure()
    sensornum = len(ca_wave_parameters['sensors'])
    trace_axes = dict()
    for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
        print(sensor)
        if sensor_i==0:
            trace_axes['{}_f_corr_cell'.format(sensor)] = fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1)
            trace_axes['{}_f_corr_cell'.format(sensor)].set_ylabel('raw pixel values - each cell')
            trace_axes['{}_f_corr_all'.format(sensor)]= fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum,sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])#cawave_f_corr_list
            trace_axes['{}_f_corr_all'.format(sensor)].set_ylabel('raw pixel values')
            trace_axes['{}_dff_cell'.format(sensor)] = fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum*2,sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])
            trace_axes['{}_dff_cell'.format(sensor)].set_ylabel('dF/F - each cell')
            trace_axes['{}_dff_all'.format(sensor)]= fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum*3,sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])#cawave_f_corr_list
            trace_axes['{}_dff_all'.format(sensor)].set_ylabel('dF/F')
        else:
            trace_axes['{}_f_corr_cell'.format(sensor)] = fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1,sharey =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])],sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])
            trace_axes['{}_f_corr_all'.format(sensor)]= fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum,sharey =  trace_axes['{}_f_corr_all'.format(ca_wave_parameters['sensors'][0])],sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])
            trace_axes['{}_dff_cell'.format(sensor)] = fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum*2,sharey =  trace_axes['{}_dff_cell'.format(ca_wave_parameters['sensors'][0])],sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])
            trace_axes['{}_dff_all'.format(sensor)]= fig.add_subplot(4,len(ca_wave_parameters['sensors']),sensor_i+1+sensornum*3,sharey =  trace_axes['{}_dff_all'.format(ca_wave_parameters['sensors'][0])],sharex =  trace_axes['{}_f_corr_cell'.format(ca_wave_parameters['sensors'][0])])
        trace_axes['{}_f_corr_cell'.format(sensor)].set_title(sensor)
        trace_axes['{}_dff_all'.format(sensor)].set_xlabel('time from AP (ms)')
        #x,y = utils_plot.generate_superresolution_trace(np.concatenate(ca_traces_dict[sensor]['cawave_time_list']),np.concatenate(ca_traces_dict[sensor]['cawave_f_corr_list']),plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'mean')
        x_conc = np.concatenate(ca_traces_dict[sensor]['cawave_time_list'])
        y_conc= np.concatenate(ca_traces_dict[sensor]['cawave_f_corr_list'])
        needed_idx = (x_conc>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc<=plot_parameters['end_time']+plot_parameters['bin_size'])
        x_conc = x_conc[needed_idx]
        y_conc = y_conc[needed_idx]
        trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'all') 
        x = np.asarray(trace_dict['bin_centers'])
        y = np.asarray(trace_dict['trace_mean'])
        y_sd = np.asarray(trace_dict['trace_std'])
        y_n = np.asarray(trace_dict['trace_n'])
        if not plot_parameters['normalize_raw_trace_offset']:
            y = y+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
        trace_axes['{}_f_corr_all'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = 1)
        if plot_parameters['shaded_error_bar']=='sd':
            trace_axes['{}_f_corr_all'.format(sensor)].fill_between(x, y-y_sd, y+y_sd,color =plot_parameters['sensor_colors'][sensor],alpha = .3)
        elif plot_parameters['shaded_error_bar']=='se':
            y_sd = y_sd/np.sqrt(y_n)
            trace_axes['{}_f_corr_all'.format(sensor)].fill_between(x, y-y_sd, y+y_sd,color =plot_parameters['sensor_colors'][sensor],alpha = .3)
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.tight_layout()
            
        #x,y = utils_plot.generate_superresolution_trace(np.concatenate(ca_traces_dict[sensor]['cawave_time_list']),np.concatenate(ca_traces_dict[sensor]['cawave_dff_list']),plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'mean')
        
        x_conc = np.concatenate(ca_traces_dict[sensor]['cawave_time_list'])
        y_conc= np.concatenate(ca_traces_dict[sensor]['cawave_dff_list'])
        needed_idx = (x_conc>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc<=plot_parameters['end_time']+plot_parameters['bin_size'])
        x_conc = x_conc[needed_idx]
        y_conc = y_conc[needed_idx]
        trace_dict = utils_plot.generate_superresolution_trace(x_conc,y_conc,plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'all') 
        x = np.asarray(trace_dict['bin_centers'])
        y = np.asarray(trace_dict['trace_mean'])
        y_sd = np.asarray(trace_dict['trace_std'])
        y_n = np.asarray(trace_dict['trace_n'])
        
        trace_axes['{}_dff_all'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = 1)
        if plot_parameters['shaded_error_bar']=='sd':
            trace_axes['{}_dff_all'.format(sensor)].fill_between(x, y-y_sd, y+y_sd,color =plot_parameters['sensor_colors'][sensor],alpha = .3)
        elif plot_parameters['shaded_error_bar']=='se':
            y_sd = y_sd/np.sqrt(y_n)
            trace_axes['{}_dff_all'.format(sensor)].fill_between(x, y-y_sd, y+y_sd,color =plot_parameters['sensor_colors'][sensor],alpha = .3)
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.tight_layout()
            
        #uniquelens = np.unique(ca_traces_dict[sensor]['cawave_lengths'])
        for cell in ca_traces_dict_by_cell[sensor]:
            if len(cell['cawave_time_list'])>=plot_parameters['min_ap_per_apnum']:
                x_conc_cell = np.concatenate(cell['cawave_time_list'])
                y_conc_cell = np.concatenate(cell['cawave_f_corr_list'])
                needed_idx = (x_conc_cell>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc_cell<=plot_parameters['end_time']+plot_parameters['bin_size'])
                x_conc_cell = x_conc_cell[needed_idx]
                y_conc_cell = y_conc_cell[needed_idx]
                x,y = utils_plot.generate_superresolution_trace(x_conc_cell,y_conc_cell,plot_parameters['bin_step_for_cells'],plot_parameters['bin_size_for_cells'],function = 'mean')
                if not plot_parameters['normalize_raw_trace_offset']:
                    y = y+np.mean(cell['cawave_f0_list'])
                trace_axes['{}_f_corr_cell'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .1)
                
                x_conc_cell = np.concatenate(cell['cawave_time_list'])
                y_conc_cell = np.concatenate(cell['cawave_dff_list'])
                needed_idx = (x_conc_cell>=plot_parameters['start_time']-plot_parameters['bin_size'])&(x_conc_cell<=plot_parameters['end_time']+plot_parameters['bin_size'])
                x_conc_cell = x_conc_cell[needed_idx]
                y_conc_cell = y_conc_cell[needed_idx]
                
                
                x,y = utils_plot.generate_superresolution_trace(x_conc_cell,y_conc_cell,plot_parameters['bin_step_for_cells'],plot_parameters['bin_size_for_cells'],function = 'mean')
                trace_axes['{}_dff_cell'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .1)
                
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.sca(trace_axes['{}_f_corr_cell'.format(sensor)])
            plt.tight_layout()
            plt.sca(trace_axes['{}_dff_cell'.format(sensor)])
            plt.tight_layout()
            
   