from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import utils.utils_plot as utils_plot
def scatter_plot_all_event_properties(ca_waves_dict,ca_wave_parameters,plot_parameters):
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
    ax_1ap_dff_amplitude_cell = fig.add_subplot(4,3,2)
    ax_1ap_f_amplitude_cell = fig.add_subplot(4,3,3)
    ax_1ap_snr_amplitude_cell = fig.add_subplot(4,3,1)
    ax_1ap_f0_cell = fig.add_subplot(4,3,4)
    
    ax_1ap_baseline_noise_cell = fig.add_subplot(4,3,5)
    ax_pixel_num_cell = fig.add_subplot(4,3,6)
    ax_cell_depth = fig.add_subplot(4,3,8)
    ax_movie_power = fig.add_subplot(4,3,7)
    ax_expression_time = fig.add_subplot(4,3,9)
    ax_neuropil = fig.add_subplot(4,3,12)
    ca_wave_parameters['sensor']='all'
    ax_1ap_dff_amplitude_cell.set_title(ca_wave_parameters)
    ax_pixel_num_cell.set_ylabel('ROI pixel number')
    ax_1ap_dff_amplitude_cell.set_ylabel('peak amplitude (dF/F)')
    ax_1ap_f0_cell.set_ylabel('baseline fluorescence (F)')
    ax_1ap_f_amplitude_cell.set_ylabel('peak amplitude (F)')
    ax_1ap_snr_amplitude_cell.set_ylabel('peak amplitude SNR')
    ax_1ap_baseline_noise_cell.set_ylabel('baseline noise (std(F))')
    ax_1ap_baseline_f_vs_noise_cell = fig.add_subplot(4,3,10)
    ax_1ap_baseline_f_vs_noise_cell.set_xlabel('baseline fluorescence (F)')
    ax_1ap_baseline_f_vs_noise_cell.set_ylabel('baseline noise (std(F))')
    ax_expression_time.set_ylabel('expression time (days)')
    ax_cell_depth.set_ylabel('imaging depth (microns)')
    
    ax_movie_power.set_ylabel('power (mW)')
    ax_neuropil.set_ylabel('r_neuropil - calculated')
    cell_index = 0
    for sensor in ca_wave_parameters['sensors']:
        key['session_calcium_sensor'] = sensor
        cellwave_by_cell_list = np.asarray(ca_waves_dict[sensor])
        cellvals = list()
        for cell_data in cellwave_by_cell_list: # set order
            apidx = cell_data['ap_group_ap_num_mean_per_ap']==plot_parameters['ap_num_to_plot']
            if cell_data['event_num_per_ap'][apidx]>=plot_parameters['min_ap_per_apnum']:
                cellvals.append(cell_data['cawave_snr_mean_per_ap'][apidx][0])#cawave_snr_mean_per_ap#cawave_peak_amplitude_dff_mean_per_ap
            else:
                cellvals.append(np.nan)
        order = np.argsort(cellvals)
        cellwave_by_cell_list = cellwave_by_cell_list[order]
        for cell_data in cellwave_by_cell_list: # plot
            apidx = cell_data['ap_group_ap_num_mean_per_ap']==1
            if cell_data['event_num_per_ap'][apidx] < plot_parameters['min_ap_per_apnum']:
                continue
            cell_index += 1
            ax_1ap_dff_amplitude_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_peak_amplitude_dff'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            ax_1ap_f_amplitude_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_peak_amplitude_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            ax_1ap_f0_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            ax_1ap_snr_amplitude_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_snr'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            ax_1ap_baseline_noise_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_baseline_f_std'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            
            ax_1ap_baseline_f_vs_noise_cell.plot(cell_data['cawave_baseline_f'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],cell_data['cawave_baseline_f_std'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']] ,     'o',ms=1,color = sensor_colors[sensor])
            ax_pixel_num_cell.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['roi_pixel_num'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])##roi_dwell_time_multiplier
            ax_movie_power.plot(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['movie_power'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            ax_expression_time.plot(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['session_sensor_expression_time'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=3,color = sensor_colors[sensor])
            ax_neuropil.semilogy(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['cawave_rneu'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=3,color = sensor_colors[sensor])
            try:
                #ax_cell_depth.plot(cell_index,cell_data['cell_key']['depth'],'o',ms=8,color = sensor_colors[sensor])
                ax_cell_depth.plot(np.ones(sum(cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']))*cell_index,cell_data['movie_hmotors_sample_z'][cell_data['ap_group_ap_num']==plot_parameters['ap_num_to_plot']],'o',ms=1,color = sensor_colors[sensor])
            except:
                pass 
    ax_1ap_dff_amplitude_cell.legend(handles=legend_elements)
    ax_cell_depth.invert_yaxis()
    ax_cell_depth.set_ylim([ax_cell_depth.get_ylim()[0],0])
    plt.show()

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
        x,y = utils_plot.generate_superresolution_trace(np.concatenate(ca_traces_dict[sensor]['cawave_time_list']),np.concatenate(ca_traces_dict[sensor]['cawave_f_corr_list']),plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'mean')
        if not plot_parameters['normalize_raw_trace_offset']:
            y = y+np.mean(ca_traces_dict[sensor]['cawave_f0_list'])
        trace_axes['{}_f_corr_all'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = 1)
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.tight_layout()
            
        x,y = utils_plot.generate_superresolution_trace(np.concatenate(ca_traces_dict[sensor]['cawave_time_list']),np.concatenate(ca_traces_dict[sensor]['cawave_dff_list']),plot_parameters['bin_step'],plot_parameters['bin_size'],function = 'mean')
        trace_axes['{}_dff_all'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = 1)
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.tight_layout()
            
        #uniquelens = np.unique(ca_traces_dict[sensor]['cawave_lengths'])
        for cell in ca_traces_dict_by_cell[sensor]:
            if len(cell['cawave_time_list'])>=plot_parameters['min_ap_per_apnum']:
                x,y = utils_plot.generate_superresolution_trace(np.concatenate(cell['cawave_time_list']),np.concatenate(cell['cawave_f_corr_list']),plot_parameters['bin_step_for_cells'],plot_parameters['bin_size_for_cells'],function = 'mean')
                if not plot_parameters['normalize_raw_trace_offset']:
                    y = y+np.mean(cell['cawave_f0_list'])
                trace_axes['{}_f_corr_cell'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .1)
                
                x,y = utils_plot.generate_superresolution_trace(np.concatenate(cell['cawave_time_list']),np.concatenate(cell['cawave_dff_list']),plot_parameters['bin_step_for_cells'],plot_parameters['bin_size_for_cells'],function = 'mean')
                trace_axes['{}_dff_cell'.format(sensor)].plot(x,y,'-',color = plot_parameters['sensor_colors'][sensor],alpha = .1)
                
        if sensor_i==len(ca_wave_parameters['sensors'])-1:
            plt.sca(trace_axes['{}_f_corr_cell'.format(sensor)])
            plt.tight_layout()
            plt.sca(trace_axes['{}_dff_cell'.format(sensor)])
            plt.tight_layout()