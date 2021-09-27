import matplotlib.pyplot as plt
from utils import utils_plot, utils_ephys
import numpy as np
import os 
import subprocess
import seaborn as sns
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
from matplotlib.lines import Line2D
import json
#%%

def plot_r_neu_distribution(sensor_colors,ca_wave_parameters,plot_parameters):
    #%%
    sensor_names = {'GCaMP7F': 'jGCaMP7f',
                'XCaMPgf': 'XCaMPgf',
                '456': 'jGCaMP8f',
                '686': 'jGCaMP8m',
                '688': 'jGCaMP8s'}
    depth_fit_x = np.arange(50,200)
    depth_fit_order = 1
    radius_fit_x = np.arange(5,20)
    radius_fit_order = 1
    min_rneu_corrcoeff = .7
    fig = plt.figure(figsize = [15,15])
    if 'ax' in plot_parameters.keys():
        ax_hist_rneu = plot_parameters['ax']
    else:
        ax_hist_rneu = fig.add_subplot(2,2,1)
    ax_cumhist_rneu = ax_hist_rneu.twinx()
    ax_depth_rneu = fig.add_subplot(2,2,2)
    ax_radius_rneu = fig.add_subplot(2,2,3)
    ax_radius_depth = fig.add_subplot(2,2,4)
    rneu_histogram_step = .1
    
    
    #rneu_all = (imaging_gt.SessionCalciumSensor()*imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>.7'&'channel_number=1'&'neuropil_number = 1').fetch('r_neu')
    bar_width = rneu_histogram_step*.9#/len(ca_wave_parameters['sensors'])*.9
    hist_range = np.arange(.2,2+rneu_histogram_step,rneu_histogram_step)
    cum_hist_range = np.arange(.2,2+.1,.01)
    #for sensor in sensor_colors.keys():
    ys = list()
    cum_ys = list()
    rneu_all = list()
    depth_all = list()
    for sensor in ca_wave_parameters['sensors']:
        data,depth,roi_radius,movie_pixel_size = (imaging.Movie()*imaging.ROI()*imaging.MoviePowerPostObjective()*imaging_gt.MovieDepth()*imaging_gt.SessionCalciumSensor()*imaging_gt.ROINeuropilCorrelation()&'r_neu_corrcoeff>{}'.format(min_rneu_corrcoeff)&'channel_number=1'&'neuropil_number = 1'&'session_calcium_sensor = "{}"'.format(sensor)).fetch('r_neu','movie_depth','roi_radius','movie_pixel_size')#'movie_depth'movie_power
        
        cell_radius = roi_radius*np.asarray(movie_pixel_size,float)
        needed = (np.isnan(depth)==False) & (np.isnan(data)==False) &(data<=2) &(cell_radius<12)
        rneu_all.extend(data[needed])
        depth_all.extend(depth[needed])
        y,x = np.histogram(data,hist_range)
        y_cum,x_cum = np.histogram(data,cum_hist_range)
        cum_ys.append(y_cum)
        ys.append(y)
        p=np.polyfit(depth[needed],data[needed],depth_fit_order)
        ax_depth_rneu.plot(depth,data,'o',color = sensor_colors[sensor],alpha = .3)
        ax_depth_rneu.plot(depth_fit_x,np.polyval(p,depth_fit_x),'-',color = sensor_colors[sensor])
        
        
        p=np.polyfit(depth[needed],cell_radius[needed],depth_fit_order)
        ax_radius_depth.plot(depth,cell_radius,'o',color = sensor_colors[sensor],alpha = .3)
        ax_radius_depth.plot(depth_fit_x,np.polyval(p,depth_fit_x),'-',color = sensor_colors[sensor])
        
        p=np.polyfit(cell_radius[needed],data[needed],depth_fit_order)
        ax_radius_rneu.plot(cell_radius,data,'o',color = sensor_colors[sensor],alpha = .3)
        ax_radius_rneu.plot(radius_fit_x,np.polyval(p,radius_fit_x),'-',color = sensor_colors[sensor])
    hist_bins = np.mean([hist_range[:-1],hist_range[1:]],0)
    cum_hist_bins = np.mean([cum_hist_range[:-1],cum_hist_range[1:]],0)
    bottoms = np.zeros(len(hist_bins))
    for i,sensor in enumerate(sensor_colors.keys()):
        ax_hist_rneu.bar(hist_bins, ys[i], bar_width,bottom=bottoms, color= sensor_colors[sensor],alpha = .3)#-plot_parameters['event_histogram_step']/2+bar_width*i
        ax_cumhist_rneu.plot(cum_hist_bins, np.cumsum(cum_ys[i])/sum(cum_ys[i]), color= sensor_colors[sensor],linewidth = 2)#-plot_parameters['event_histogram_step']/2+bar_width*i
        bottoms = ys[i]+bottoms 
    ax_cumhist_rneu.set_ylim([0,1])
    ax_hist_rneu.set_xlabel('calculated r_neu using ground truth ephys')
    ax_hist_rneu.set_ylabel('number of ROIs')
    
    p=np.polyfit(depth_all,rneu_all,depth_fit_order)
    ax_depth_rneu.plot(depth_fit_x,np.polyval(p,depth_fit_x),'k-',linewidth = 5)
    ax_depth_rneu.set_xlabel('depth from surface of pia (microns)')
    ax_depth_rneu.set_ylabel('calculated r_neu using ground truth ephys')
    ax_depth_rneu.set_ylim([.25,2])
    legend_elements = list()
    for sensor in sensor_colors.keys():
        legend_elements.append(Line2D([0], [0], marker='o', color=sensor_colors[sensor], 
                                      label=sensor_names[sensor],markerfacecolor=sensor_colors[sensor], markersize=8))
    ax_hist_rneu.legend(handles=legend_elements)   
    
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['fig_name']+'.{}')
    fig.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )
    
    #%%
def plot_stats(plot_properties,inclusion_criteria):
    #%%
    ephys_snr_cond = 'sweep_ap_snr_dv_median>={}'.format(inclusion_criteria['min_sweep_ap_snr_dv_median'])
    ephys_hwstd_cond = 'sweep_ap_hw_std_per_median<={}'.format(inclusion_criteria['max_sweep_ap_hw_std_per_median'])
    channel_offset_cond = 'channel_offset > {}'.format(inclusion_criteria['min_channel_offset'])
    channel_num_cond = 'channel_number = 1'
    
    sensornames = list()
    for sensor in plot_properties['sensors']:
        sensornames.append(plot_properties['sensor_names'][sensor])
    fig_exptime_movielen = plt.figure()
    ax_exptime_movielen = fig_exptime_movielen.add_subplot(1,1,1)
    fig_stats = plt.figure(figsize = [25,15])
    ax_subject_num = fig_stats.add_subplot(2,2,1)
    ax_subject_expression_time = ax_subject_num.twinx()
    ax_cell_num = fig_stats.add_subplot(2,2,2)
    ax_cell_num_per_subject = ax_cell_num.twinx()
    ax_movie_lengths_with_gt = fig_stats.add_subplot(2,2,3)
    ax_movie_lengths_with_gt_per_scell = ax_movie_lengths_with_gt.twinx()
    ax_ap_number = fig_stats.add_subplot(2,2,4)
    ax_ap_number_per_cell = ax_ap_number.twinx()
    linewidth = 3
    markersize = 4
    alpha = .2
    swarm_apnum_x = list()
    swarm_apnum_y = list()
    swarm_movielen_x = list()
    swarm_movielen_y = list()
    swarm_cellnum_x = list()
    swarm_cellnum_y = list()
    swarm_exptime_x = list()
    swarm_exptime_y = list()
    for sensor_i,sensor in enumerate(plot_properties['sensors']):
        sensor_filter = 'session_calcium_sensor = "{}"'.format(sensor)
        expression_time_filter = 'session_sensor_expression_time < {}'.format(plot_properties['max_expression_time'])
        sessions = lab.Subject()*imaging_gt.SessionCalciumSensor() & sensor_filter & expression_time_filter# sessions = subjects
        cells = ephys_cell_attached.Cell()*imaging_gt.SessionCalciumSensor() & sensor_filter & expression_time_filter
        #movies = imaging.Movie()*imaging_gt.SessionCalciumSensor() & sensor_filter & expression_time_filter
        movies_with_gt_roi = ephysanal_cell_attached.SweepAPQC()*imaging.MovieChannel()*imaging.Movie()*imaging_gt.SessionCalciumSensor()*imaging_gt.ROISweepCorrespondance() & sensor_filter & expression_time_filter& ephys_snr_cond&ephys_hwstd_cond&channel_offset_cond&channel_num_cond
        action_potentials = ephysanal_cell_attached.SweepAPQC()*imaging.MovieChannel()*ephysanal_cell_attached.ActionPotential()*imaging_gt.SessionCalciumSensor()*imaging_gt.ROISweepCorrespondance() & sensor_filter & expression_time_filter & ephys_snr_cond&ephys_hwstd_cond&channel_offset_cond&channel_num_cond
        cells_per_subject = list()
        for session in sessions:
            cells_per_subject.append(len(np.unique((imaging_gt.ROISweepCorrespondance()&session).fetch('cell_number'))))#len(cells&session)
        
        movie_lengths_per_cell_with_gt_roi = list()
        ap_number_per_cell = list()
        expression_time_per_cell = list()
        cell_num = 0
        for cell in cells:
            if len(movies_with_gt_roi&cell) > 0:
                cell_num += 1
                movie_frame_rates,movie_frame_nums,expression_times = (movies_with_gt_roi&cell).fetch('movie_frame_rate','movie_frame_num','session_sensor_expression_time')
                movie_lengths_per_cell_with_gt_roi.append(np.sum(movie_frame_nums/movie_frame_rates))
                expression_time_per_cell.append(np.mean(expression_times))
                ap_number_per_cell.append(len(action_potentials&cell))
    
        movie_lengths_per_cell_with_gt_roi = np.asarray(movie_lengths_per_cell_with_gt_roi)
        
        ax_exptime_movielen.plot(expression_time_per_cell,movie_lengths_per_cell_with_gt_roi/60,'o',color = plot_properties['sensor_colors'][sensor],markersize = markersize,alpha=alpha)
        
    
        subject_num = len(sessions) # subject = sessions because it's a terminal experiment
        expression_time = sessions.fetch('session_sensor_expression_time')
        ax_subject_num.bar(sensor_i,subject_num,edgecolor = plot_properties['sensor_colors'][sensor],fill=False,linewidth = linewidth)
        #ax_subject_expression_time.plot(np.ones(subject_num)*sensor_i,expression_time,'o',color = plot_properties['sensor_colors'][sensor],markersize = markersize,alpha=alpha)
        swarm_exptime_x.append(np.ones(len(expression_time))*sensor_i)
        swarm_exptime_y.append(expression_time)
        
        #cell_num = len(cells)
        ax_cell_num.bar(sensor_i,cell_num,edgecolor = plot_properties['sensor_colors'][sensor],fill=False,linewidth = linewidth)
        #ax_cell_num_per_subject.plot(np.ones(subject_num)*sensor_i,cells_per_subject,'o',color = plot_properties['sensor_colors'][sensor],markersize = markersize,alpha=alpha)
        swarm_cellnum_x.append(np.ones(len(cells_per_subject))*sensor_i)
        swarm_cellnum_y.append(cells_per_subject)
        
        
        ax_movie_lengths_with_gt.bar(sensor_i,np.sum(movie_lengths_per_cell_with_gt_roi)/3600,edgecolor = plot_properties['sensor_colors'][sensor],fill=False,linewidth = linewidth)
        #ax_movie_lengths_with_gt_per_scell.plot(np.ones(cell_num)*sensor_i,movie_lengths_per_cell_with_gt_roi/60,'o',color = plot_properties['sensor_colors'][sensor],markersize = markersize,alpha=alpha)
        swarm_movielen_x.append(np.ones(len(movie_lengths_per_cell_with_gt_roi))*sensor_i)
        swarm_movielen_y.append(movie_lengths_per_cell_with_gt_roi/60)
        
        ax_ap_number.bar(sensor_i,np.sum(ap_number_per_cell)/1000,edgecolor = plot_properties['sensor_colors'][sensor],fill=False,linewidth = linewidth)
        swarm_apnum_x.append(np.ones(len(ap_number_per_cell))*sensor_i)
        swarm_apnum_y.append(ap_number_per_cell)
        #ax_ap_number_per_cell.plot(np.ones(cell_num)*sensor_i,ap_number_per_cell,'o',color = plot_properties['sensor_colors'][sensor],markersize = markersize,alpha=alpha)
        #sns.swarmplot(x =np.ones(cell_num)*sensor_i,y =ap_number_per_cell ,color=plot_properties['sensor_colors'][sensor],ax = ax_ap_number_per_cell)
        #break
        #break
    #%
    sns.swarmplot(x =np.concatenate(swarm_exptime_x),y =np.concatenate(swarm_exptime_y) ,color='black',ax = ax_subject_expression_time)  
    sns.swarmplot(x =np.concatenate(swarm_cellnum_x),y =np.concatenate(swarm_cellnum_y) ,color='black',ax = ax_cell_num_per_subject)  
    sns.swarmplot(x =np.concatenate(swarm_movielen_x),y =np.concatenate(swarm_movielen_y) ,color='black',ax = ax_movie_lengths_with_gt_per_scell)  
    ax_ap_number_per_cell.set_yscale('log')
    x = np.concatenate(swarm_apnum_x)
    y = np.concatenate(swarm_apnum_y)
    sns.swarmplot(x =x[y>0],y =y[y>0] ,color='black',ax = ax_ap_number_per_cell)  
    ax_ap_number_per_cell.set_ylim([10,100000])
    #%
    ax_subject_num.set_xticks(np.arange(len(plot_properties['sensors'])))
    ax_subject_num.set_xticklabels(sensornames)
    ax_subject_num.set_ylabel('Number of mice per sensor')
    ax_subject_expression_time.set_ylabel('Expression time (days)')
    
    ax_cell_num.set_xticks(np.arange(len(plot_properties['sensors'])))
    ax_cell_num.set_xticklabels(sensornames)
    ax_cell_num.set_ylabel('Number of cells per sensor')
    ax_cell_num_per_subject.set_ylabel('Cell number per mouse')
    
    ax_movie_lengths_with_gt.set_xticks(np.arange(len(plot_properties['sensors'])))
    ax_movie_lengths_with_gt.set_xticklabels(sensornames)
    ax_movie_lengths_with_gt.set_ylabel('Total length of movies(hr)')
    ax_movie_lengths_with_gt_per_scell.set_ylabel('Length of movies per cell (min)')
    ax_movie_lengths_with_gt_per_scell.set_ylim([0,ax_movie_lengths_with_gt_per_scell.get_ylim()[1]])
    
    
    ax_ap_number.set_xticks(np.arange(len(plot_properties['sensors'])))
    ax_ap_number.set_xticklabels(sensornames)
    ax_ap_number.set_ylabel('Total number of APs (x1000)')
    ax_ap_number_per_cell.set_ylabel('Number of APs per cell')
    
    ax_exptime_movielen.set_xlabel('Expression time (days)')
    ax_exptime_movielen.set_ylabel('Length of movies per cell (min)')
    
    figfile = os.path.join(plot_properties['figures_dir'],plot_properties['fig_name']+'.{}')
    fig_stats.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )
   #%% 
def plot_ap_scatter_with_third_parameter(ca_waves_dict,plot_parameters):
    #%%
    x_list = list()
    y_list = list()
    z_list = list()
    x_per_sensor = dict()
    y_per_sensor = dict()
    z_per_sensor = dict()
    for sensor in plot_parameters['sensors']:
        x_per_sensor[sensor] = list()
        y_per_sensor[sensor] = list()
        z_per_sensor[sensor] = list()
        for cell in ca_waves_dict[sensor]:
            idx = cell['ap_group_ap_num_mean_per_ap'] == plot_parameters['ap_num_to_plot']
            x = np.mean(cell[plot_parameters['x_parameter']])
            y = np.mean(cell[plot_parameters['y_parameter']])
            try:
                z = cell[plot_parameters['z_parameter']][idx]
            except:
                z = cell[plot_parameters['z_parameter']][0]
            
            x_list.append(x)
            y_list.append(y)
            try:
                z_list.extend(z)
            except:
                z_list.append(z)
            
            x_per_sensor[sensor].append(x)
            y_per_sensor[sensor].append(y)
            try:
                z_per_sensor[sensor].extend(z)
            except:
                z_per_sensor[sensor].append(z)
    x_list = np.asarray(x_list)       
    y_list = np.asarray(y_list)       
    z_list = np.asarray(z_list)    
    
    z_edges = np.percentile(z_list,plot_parameters['z_parameter_percentiles'])
    #print(z_edges)
    ms_edges = plot_parameters['markersize_edges']
    fig_ephys_scatter = plt.figure(figsize = [10,5])
    ax_ephys_scatter =     fig_ephys_scatter.add_subplot(1,1,1)
    for sensor in plot_parameters['sensors']:
        if plot_parameters['normalize z by sensor']:
            z_edges = np.percentile(z_per_sensor[sensor],plot_parameters['z_parameter_percentiles'])
        for x,y,z in zip(x_per_sensor[sensor],y_per_sensor[sensor],z_per_sensor[sensor]):
            if z<z_edges[0]:
                z = z_edges[0]
            elif z>z_edges[1]:
                z = z_edges[1]
            ms = (z-z_edges[0])/np.diff(z_edges)*ms_edges[1]+ms_edges[0]
            if plot_parameters['invert y axis']:
                if y<.1:
                    y=.1
                #y[y<.001] = .001
                ax_ephys_scatter.plot(x,1/y,'o',color = plot_parameters['sensor_colors'][sensor],markersize = ms,alpha = plot_parameters['alpha'])
            else:
                ax_ephys_scatter.plot(x,y,'o',color = plot_parameters['sensor_colors'][sensor],markersize = ms,alpha = plot_parameters['alpha'])
    #ax_ephys_scatter.plot(x_list,y_list,'ko')
    legend_elements = list()
    for sensor in plot_parameters['sensors']:
        legend_elements.append(Line2D([0], [0], marker='o', color='white',#plot_parameters['sensor_colors'][sensor], 
                                      label=sensor,markerfacecolor=plot_parameters['sensor_colors'][sensor], markersize=8))
    legend_elements.append(Line2D([0], [0], marker='o', color = 'white',
                                      label='{} {}'.format(int(np.ceil(z_edges[0])),plot_parameters['z_parameter_unit']),markerfacecolor='black', markersize=(np.ceil(z_edges[0])-z_edges[0])/np.diff(z_edges)*ms_edges[1]+ms_edges[0]))
    legend_elements.append(Line2D([0], [0], marker='o',color = 'white',
                                      label='{} {}'.format(int(np.ceil(z_edges[1])),plot_parameters['z_parameter_unit']),markerfacecolor='black', markersize=(np.ceil(z_edges[1])-z_edges[0])/np.diff(z_edges)*ms_edges[1]+ms_edges[0]))
    ax_ephys_scatter.legend(handles=legend_elements)     
    ax_ephys_scatter.set_xlabel(plot_parameters['x_parameter'])
    ax_ephys_scatter.set_ylabel(plot_parameters['y_parameter'])
    ax_ephys_scatter.set_title(plot_parameters['z_parameter'])
    if 'xlim' in plot_parameters.keys():
        ax_ephys_scatter.set_xlim(plot_parameters['xlim'])
    if 'xlabel' in plot_parameters.keys():
        ax_ephys_scatter.set_xlabel(plot_parameters['xlabel'])
    if 'ylabel' in plot_parameters.keys():
        ax_ephys_scatter.set_ylabel(plot_parameters['ylabel'])
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['fig_name']+'.{}')
    fig_ephys_scatter.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )
 #%%       

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

#%%
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
    #%%
    risename = 'risetime_0_{}'.format(int(plot_parameters['partial_risetime_target']*100))
    dict_out = dict()
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
    palettes = []
    for sensor_i,sensor in enumerate(ca_wave_parameters['sensors']):
        
        palettes.append(plot_parameters['sensor_colors']['boxplot'][sensor])
        sensor_amplitudes = list()
        sensor_rise_times = list()
        sensor_half_decay_times = list()
        sensor_amplitudes_superresolution = list()
        sensor_rise_times_superresolution = list()
        sensor_half_decay_times_superresolution = list()
        sensor_partial_rise_times_superresolution = list()
        sensor_cell_names = list()
        # each transient separately
        for ca_waves_dict_cell in ca_waves_dict[sensor]:
            sensor_cell_name = 'anm{}_cell{}'.format(ca_waves_dict_cell['cell_key']['subject_id'],ca_waves_dict_cell['cell_key']['cell_number'])
            if sensor_cell_name in plot_parameters['blacklist']:
                continue
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
            
        #ax_amplitude.bar(sensor_i,np.median(sensor_amplitudes),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
        #ax_risetime.bar(sensor_i,np.median(sensor_rise_times),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
        #ax_half_decay_time.bar(sensor_i,np.median(sensor_half_decay_times),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
        
        #% # from superresolution traces
        xs = superresolution_traces[sensor]['{}_time_per_cell'.format(wave_parameters['trace_to_use'])]
        ys = superresolution_traces[sensor]['{}_{}_per_cell'.format(wave_parameters['trace_to_use'],wave_parameters['superresolution_function'])]
        ns = superresolution_traces[sensor]['{}_n_per_cell'.format(wave_parameters['trace_to_use'])]
        
        for x,y,n,ca_waves_dict_cell in zip(xs,ys,ns,ca_waves_dict[sensor]):
            
            sensor_cell_name = 'anm{}_cell{}'.format(ca_waves_dict_cell['cell_key']['subject_id'],ca_waves_dict_cell['cell_key']['cell_number'])
            if sensor_cell_name in plot_parameters['blacklist']:
                continue
            sensor_cell_names.append(sensor_cell_name)
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
            #half_decay_time_idx = np.argmax(y_norm[peak_idx:]<.5)+peak_idx
            half_decay_time_idx_1 = np.argmax(y_norm[peak_idx:]<.5)+peak_idx
            half_decay_time_idx_2 = len(y_norm[peak_idx:-1])-np.argmax(y_norm[-1:peak_idx:-1]>.5)+peak_idx
            half_decay_time_idx = int(np.mean([half_decay_time_idx_1,half_decay_time_idx_2]))
            
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
            
        dict_out[sensor] = {'cell_names':sensor_cell_names,
                            risename:sensor_partial_rise_times_superresolution,
                            'time_to_peak':sensor_rise_times_superresolution,
                            'amplitude':sensor_amplitudes_superresolution,
                            'half_decay_time':sensor_half_decay_times_superresolution}
        
        
        #%   
# =============================================================================
#         ax_amplitude_superres.bar(sensor_i,np.median(sensor_amplitudes_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4)
#         ax_risetime_superres.bar(sensor_i,np.median(sensor_rise_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 
#         ax_half_decay_time_superres.bar(sensor_i,np.median(sensor_half_decay_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 
#         ax_partial_risetime_superres.bar(sensor_i,np.median(sensor_partial_rise_times_superresolution),edgecolor = plot_parameters['sensor_colors'][sensor],fill=False,linewidth = 4) 
# =============================================================================
    
    sns.stripplot(x =amplitudes_idx_all,y = amplitudes_all,color='black',edgecolor = None,ax = ax_amplitude,alpha = .6,size = 5)     #swarmplot   
    sns.stripplot(x =amplitudes_idx_all,y = risetimes_all,color='black',ax = ax_risetime,alpha = .6,size = 5)    #swarmplot   
    sns.stripplot(x =amplitudes_idx_all,y = half_decay_times_all,color='black',ax = ax_half_decay_time,alpha = .6,size = 5)  #swarmplot   
    
    sns.boxplot(ax=ax_amplitude,    
                x=amplitudes_idx_all, 
                y=amplitudes_all, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    sns.boxplot(ax=ax_risetime,    
                x=amplitudes_idx_all, 
                y=risetimes_all, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    sns.boxplot(ax=ax_half_decay_time,    
                x=amplitudes_idx_all, 
                y=half_decay_times_all, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    
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
    
    
    sns.stripplot(x =amplitudes_idx_all_superresolution,y = amplitudes_all_superresolution,color='black',ax = ax_amplitude_superres,alpha = .5,size = 5)       
    sns.stripplot(x =amplitudes_idx_all_superresolution,y = risetimes_all_superresolution,color='black',ax = ax_risetime_superres,alpha = .5,size = 5)    
    sns.stripplot(x =amplitudes_idx_all_superresolution,y = half_decay_times_all_superresolution,color='black',ax = ax_half_decay_time_superres,alpha = .5,size = 5)  
    sns.stripplot(x =amplitudes_idx_all_superresolution,y = partial_risetimes_all_superresolution,color='black',ax = ax_partial_risetime_superres,alpha = .5,size = 5)  
    
    sns.boxplot(ax=ax_amplitude_superres,    
                x=amplitudes_idx_all_superresolution, 
                y=amplitudes_all_superresolution, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    sns.boxplot(ax=ax_risetime_superres,    
                x=amplitudes_idx_all_superresolution, 
                y=risetimes_all_superresolution, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    sns.boxplot(ax=ax_half_decay_time_superres,    
                x=amplitudes_idx_all_superresolution, 
                y=half_decay_times_all_superresolution, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    sns.boxplot(ax=ax_partial_risetime_superres,    
                x=amplitudes_idx_all_superresolution, 
                y=partial_risetimes_all_superresolution, 
                showfliers=False, 
                linewidth=2,
                width=.5,
                palette = palettes)
    
    
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
        
# =============================================================================
#         ax_amplitude_superres.plot(sensor_i,amplitude,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
#         ax_risetime_superres.plot(sensor_i,rise_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
#         ax_half_decay_time_superres.plot(sensor_i,half_decay_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
#         ax_partial_risetime_superres.plot(sensor_i,partial_rise_time,'_',color = plot_parameters['sensor_colors'][sensor], markersize = 35,markeredgewidth = 3)
# =============================================================================
    
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
     
    ax_amplitude.set_ylim([0,ax_amplitude.get_ylim()[1]])
    ax_risetime.set_ylim([0,ax_risetime.get_ylim()[1]])
    ax_half_decay_time.set_ylim([0,ax_half_decay_time.get_ylim()[1]])
    ax_partial_risetime_superres.set_ylim([0,ax_partial_risetime_superres.get_ylim()[1]])
    
    #ax_partial_risetime_superres.set_yscale('log')
    #ax_partial_risetime_superres.set_ylim([1,ax_partial_risetime_superres.get_ylim()[1]])
    figfile = os.path.join(plot_parameters['figures_dir'],plot_parameters['plot_name']+'.{}')
    fig_scatter_plots_1ap.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )   
    json.dump(dict_out, open(os.path.join(plot_parameters['figures_dir'],plot_parameters['plot_name']+'.json'), 'w'), sort_keys=True, indent='\t', separators=(',', ': '))
       #%% 
def plot_sample_traces(plot_parameters):
    sweeps = plot_parameters['sweeps']
    movie_length = plot_parameters['movie_length']
    zoom_length = plot_parameters['zoom_length']
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
            axs_meanimages.append(fig_meanimages.add_subplot(plot_parameters['rownum'],2,len(axs_meanimages)+1))
            im_green = axs_meanimages[-1].imshow(mean_image,cmap = 'gray', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)])#, aspect='auto')
            clim = np.percentile(mean_image.flatten(),plot_parameters['cax_limits'])
            im_green.set_clim(clim)
            ROI = np.zeros_like(mean_image)*np.nan
            ROI[roi_ypix,roi_xpix] = 1
            axs_meanimages[-1].imshow(ROI,cmap = 'Greens', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)],alpha = plot_parameters['alpha_roi'],vmin = 0, vmax = 1)#, aspect='auto'
            utils_plot.add_scalebar_to_fig(axs_meanimages[-1],
                                           axis = 'x',
                                           size = 20,
                                           conversion = 1,
                                           unit = r'$\mu$m',
                                           color = 'white',
                                           location = 'top right',
                                           linewidth = 5) 
            axs_meanimages.append(fig_meanimages.add_subplot(plot_parameters['rownum'],2,len(axs_meanimages)+1))
            im_red = axs_meanimages[-1].imshow(mean_image_red,cmap = 'gray', extent=[0,mean_image.shape[1]*float(movie_pixel_size),0,mean_image.shape[0]*float(movie_pixel_size)])#, aspect='auto')
            clim = np.percentile(mean_image_red.flatten(),plot_parameters['cax_limits'])
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
                axs_longtrace.append(fig_longtrace.add_subplot(plot_parameters['rownum'],1,len(axs_longtrace)+1))
            else:
                axs_longtrace.append(fig_longtrace.add_subplot(plot_parameters['rownum'],1,len(axs_longtrace)+1,sharey = axs_longtrace[0]))
            if len(axs_zoomtrace)==0:
                axs_zoomtrace.append(fig_zoom.add_subplot(plot_parameters['rownum'],1,len(axs_zoomtrace)+1))
            else:
                axs_zoomtrace.append(fig_zoom.add_subplot(plot_parameters['rownum'],1,len(axs_zoomtrace)+1,sharey = axs_zoomtrace[0]))
                
            
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
            roi_f_corr = roi_f-neuropil_f*plot_parameters['cawave_rneu']
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
            axs_longtrace[-1].plot(frame_times[dff_idx_needed],roi_dff[dff_idx_needed],color = plot_parameters['sensor_colors'][sensor])
            axs_longtrace[-1].set_xlim([start_time,start_time+movie_length])
            axs_longtrace[-1].axis('off')
            
            ephys_idx_needed = (ephys_time>=zoom_start_time) & (ephys_time<=zoom_start_time+zoom_length)
            dff_idx_needed = (frame_times>=zoom_start_time) & (frame_times<=zoom_start_time+zoom_length)
            
            axs_zoomtrace[-1].plot(ephys_time[ephys_idx_needed],response_trace_filt_scaled[ephys_idx_needed],color = 'gray')
            axs_zoomtrace[-1].plot(frame_times[dff_idx_needed],roi_dff[dff_idx_needed],color = plot_parameters['sensor_colors'][sensor])
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
    figfile = os.path.join(plot_parameters['figures_dir'],'fig_sample_traces_long_'+plot_parameters['filename']+'.{}')
    fig_longtrace.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )
            
    figfile = os.path.join(plot_parameters['figures_dir'],'fig_sample_traces_zoom_'+plot_parameters['filename']+'.{}')
    fig_zoom.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd ) 
    
    
    figfile = os.path.join(plot_parameters['figures_dir'],'fig_sample_traces_mean_images_'+plot_parameters['filename']+'.{}')
    fig_meanimages.savefig(figfile.format('svg'))
    inkscape_cmd = ['/usr/bin/inkscape', '--file',figfile.format('svg'),'--export-emf', figfile.format('emf')]
    subprocess.run(inkscape_cmd )    
    
    