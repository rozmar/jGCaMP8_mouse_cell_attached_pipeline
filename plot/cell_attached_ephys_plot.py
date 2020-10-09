import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached
import copy
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA  
from sklearn.manifold import TSNE  
import pandas as pd
from scipy import stats, interpolate
#%%
def ap_scatter_onpick(event):
    axesdata = event.canvas.axesdata
    data = axesdata['data']
    ap_data = data['ap_data']
    cc = False
    vc = False
    if event.artist in axesdata['vc_points']:#vc_points_hw_amplitude : 
        vc = True
        ap_idx = data['vcidx_where'][event.ind[0]]
    elif event.artist in axesdata['cc_points']:#==cc_points_hw_amplitude:
        cc = True
        ap_idx = data['ccidx_where'][event.ind[0]]
    else:
        return True
    cell_id = data['cell_IDs'][ap_idx]
    
    cellidx = np.where(data['uniquecells']==cell_id)[0][0]
    vc_idx = data['cell_stats']['voltage clamp']['ap_list_idx'][cellidx]
    cc_idx = data['cell_stats']['current clamp']['ap_list_idx'][cellidx]
    try:
        axesdata['cc_points_hw_amplitude'].selected.remove()
        axesdata['cc_points_hw_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['cc_points_fw_amplitude'].selected.remove()
        axesdata['cc_points_fw_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['cc_points_int_amplitude'].selected.remove()
        axesdata['cc_points_int_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['cc_points_sn_sn'].selected.remove()
        axesdata['cc_points_sn_sn'].clicked.remove()
    except:
        pass
    try:
        axesdata['vc_points_hw_amplitude'].selected.remove()
        axesdata['vc_points_hw_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['vc_points_fw_amplitude'].selected.remove()
        axesdata['vc_points_fw_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['vc_points_int_amplitude'].selected.remove()
        axesdata['vc_points_int_amplitude'].clicked.remove()
    except:
        pass
    try:
        axesdata['vc_points_sn_sn'].selected.remove()
        axesdata['vc_points_sn_sn'].clicked.remove()
    except:
        pass
    
    axesdata['cc_points_hw_amplitude'].selected, = axesdata['ax_hw_amplitude_cc'].plot(ap_data['ap_halfwidth'][cc_idx],ap_data['ap_amplitude'][cc_idx]*1000,'ko',ms=3)
    axesdata['cc_points_fw_amplitude'].selected, = axesdata['ax_fw_amplitude_cc'].plot(ap_data['ap_full_width'][cc_idx],ap_data['ap_amplitude'][cc_idx]*1000,'ko',ms=3)
    axesdata['cc_points_int_amplitude'].selected, = axesdata['ax_int_amplitude_cc'].plot(ap_data['ap_integral'][cc_idx]*1000,ap_data['ap_amplitude'][cc_idx]*1000,'ko',ms=3)
    axesdata['cc_points_sn_sn'].selected, = axesdata['ax_sn_sn_cc'].plot(ap_data['ap_snr_v'][cc_idx],ap_data['ap_snr_dv'][cc_idx],'ko',ms=3)
    axesdata['vc_points_hw_amplitude'].selected, = axesdata['ax_hw_amplitude_vc'].plot(ap_data['ap_halfwidth'][vc_idx],ap_data['ap_amplitude'][vc_idx]*1e12,'ko',ms=3)
    axesdata['vc_points_fw_amplitude'].selected, = axesdata['ax_fw_amplitude_vc'].plot(ap_data['ap_full_width'][vc_idx],ap_data['ap_amplitude'][vc_idx]*1e12,'ko',ms=3)
    axesdata['vc_points_int_amplitude'].selected, = axesdata['ax_int_amplitude_vc'].plot(ap_data['ap_integral'][vc_idx]*1e12,ap_data['ap_amplitude'][vc_idx]*1e12,'ko',ms=3)
    axesdata['vc_points_sn_sn'].selected, = axesdata['ax_sn_sn_vc'].plot(ap_data['ap_snr_v'][vc_idx],ap_data['ap_snr_dv'][vc_idx],'ko',ms=3)
    
    if cc:
        axesdata['cc_points_hw_amplitude'].clicked, = axesdata['ax_hw_amplitude_cc'].plot(ap_data['ap_halfwidth'][ap_idx],ap_data['ap_amplitude'][ap_idx]*1000,'rx',ms=6)
        axesdata['cc_points_fw_amplitude'].clicked, = axesdata['ax_fw_amplitude_cc'].plot(ap_data['ap_full_width'][ap_idx],ap_data['ap_amplitude'][ap_idx]*1000,'rx',ms=6)
        axesdata['cc_points_int_amplitude'].clicked, = axesdata['ax_int_amplitude_cc'].plot(ap_data['ap_integral'][ap_idx]*1000,ap_data['ap_amplitude'][ap_idx]*1000,'rx',ms=6)
        axesdata['cc_points_sn_sn'].clicked, = axesdata['ax_sn_sn_cc'].plot(ap_data['ap_snr_v'][ap_idx],ap_data['ap_snr_dv'][ap_idx],'rx',ms=6)
    else:
        axesdata['vc_points_hw_amplitude'].clicked, = axesdata['ax_hw_amplitude_vc'].plot(ap_data['ap_halfwidth'][ap_idx],ap_data['ap_amplitude'][ap_idx]*1e12,'rx',ms=6)
        axesdata['vc_points_fw_amplitude'].clicked, = axesdata['ax_fw_amplitude_vc'].plot(ap_data['ap_full_width'][ap_idx],ap_data['ap_amplitude'][ap_idx]*1e12,'rx',ms=6)
        axesdata['vc_points_int_amplitude'].clicked, = axesdata['ax_int_amplitude_vc'].plot(ap_data['ap_integral'][ap_idx]*1e12,ap_data['ap_amplitude'][ap_idx]*1e12,'rx',ms=6)
        axesdata['vc_points_sn_sn'].clicked, = axesdata['ax_sn_sn_vc'].plot(ap_data['ap_snr_v'][ap_idx],ap_data['ap_snr_dv'][ap_idx],'rx',ms=6)
    event.canvas.draw()
    
    #%
    ap_max_index_now = ap_data['ap_max_index'][ap_idx]
    
    key_sweep = {'subject_id':ap_data['subject_id'][ap_idx],
                 'session':ap_data['session'][ap_idx],
                 'cell_number':ap_data['cell_number'][ap_idx],
                 'sweep_number':ap_data['sweep_number'][ap_idx]}
    sample_rate = (ephys_cell_attached.SweepMetadata()&key_sweep).fetch1('sample_rate')
    ap_max_indices = np.array((ephysanal_cell_attached.ActionPotential()&key_sweep).fetch('ap_max_index'),int)
    trace = (ephys_cell_attached.SweepResponse()&key_sweep).fetch1('response_trace')
    time = np.arange(len(trace))/sample_rate
#%
    fig_trace = plt.figure()
    ax_trace = fig_trace.add_subplot(111)
    ax_trace.plot(time,trace,'k-')
    ax_trace.plot(time[ap_max_indices],trace[ap_max_indices],'bo')
    ax_trace.plot(time[ap_max_index_now],trace[ap_max_index_now],'ro')
    ax_trace.set_title(key_sweep)
    return True


def plot_ap_features_scatter(ap_data = None):
    if type(ap_data) != dict:
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
        apdatalist = (ephys_cell_attached.SweepMetadata()*ephysanal_cell_attached.ActionPotential()).fetch(*ap_data_to_fetch)
        #%
        ap_data = dict()
        for data,data_name in zip(apdatalist,ap_data_to_fetch):
            ap_data[data_name] = data    
        
    
    
    #%
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
    stats_to_calculate = {'ap_amplitude_mean':list(),
                          'ap_amplitude_sd':list(),
                          'ap_halfwidth_mean':list(),
                          'ap_halfwidth_sd':list(),
                          'ap_list_idx':list()}
    cell_stats = {'current clamp':copy.deepcopy(stats_to_calculate),
                  'voltage clamp':copy.deepcopy(stats_to_calculate)}
    for uniquecell in uniquecells:
        idxs = np.where((cell_IDs==uniquecell)&ccidx)[0]
        cell_stats['current clamp']['ap_amplitude_mean'].append(np.mean(ap_data['ap_amplitude'][idxs])*1000)
        cell_stats['current clamp']['ap_amplitude_sd'].append(np.std(ap_data['ap_amplitude'][idxs])*1000)
        cell_stats['current clamp']['ap_halfwidth_mean'].append(np.mean(ap_data['ap_halfwidth'][idxs]))
        cell_stats['current clamp']['ap_halfwidth_sd'].append(np.std(ap_data['ap_halfwidth'][idxs]))
        cell_stats['current clamp']['ap_list_idx'].append(idxs)
        
        idxs = np.where((cell_IDs==uniquecell)&vcidx)[0]
        cell_stats['voltage clamp']['ap_amplitude_mean'].append(np.mean(ap_data['ap_amplitude'][idxs])*1e12)
        cell_stats['voltage clamp']['ap_amplitude_sd'].append(np.std(ap_data['ap_amplitude'][idxs])*1e12)
        cell_stats['voltage clamp']['ap_halfwidth_mean'].append(np.mean(ap_data['ap_halfwidth'][idxs]))
        cell_stats['voltage clamp']['ap_halfwidth_sd'].append(np.std(ap_data['ap_halfwidth'][idxs]))
        cell_stats['voltage clamp']['ap_list_idx'].append(idxs)
        
    for key_recmode in cell_stats.keys():
        for key_val in cell_stats[key_recmode].keys():
            cell_stats[key_recmode][key_val]=np.asarray(cell_stats[key_recmode][key_val])
    
    #%
    pickertolerance=5
    fig = plt.figure()
    ax_hw_amplitude_cc = fig.add_subplot(241)
    ax_hw_amplitude_cc.set_xlabel('Halfwidth (ms)')
    ax_hw_amplitude_cc.set_ylabel('Amplitude (mV)')
    
    ax_fw_amplitude_cc = fig.add_subplot(242)
    ax_fw_amplitude_cc.set_xlabel('Full width (ms)')
    ax_fw_amplitude_cc.set_ylabel('Amplitude (mV)')
    
    ax_int_amplitude_cc = fig.add_subplot(243)
    ax_int_amplitude_cc.set_xlabel('AP integral (mVs)')
    ax_int_amplitude_cc.set_ylabel('Amplitude (mV)')
    
    ax_sn_sn_cc = fig.add_subplot(244)
    ax_sn_sn_cc.set_xlabel('SNR on trace')
    ax_sn_sn_cc.set_ylabel('SNR on dtrace')
    
    # =============================================================================
    # ax_hw_amplitude_cc_mean = fig.add_subplot(222, sharex=ax_hw_amplitude_cc,sharey=ax_hw_amplitude_cc)
    # ax_hw_amplitude_cc_mean.set_xlabel('Halfwidth (ms)')
    # ax_hw_amplitude_cc_mean.set_ylabel('Amplitude (mV)')
    # =============================================================================
    ax_hw_amplitude_vc = fig.add_subplot(245)
    ax_hw_amplitude_vc.set_xlabel('Halfwidth (ms)')
    ax_hw_amplitude_vc.set_ylabel('Amplitude (pA)')
    
    ax_fw_amplitude_vc = fig.add_subplot(246)
    ax_fw_amplitude_vc.set_xlabel('Full width (ms)')
    ax_fw_amplitude_vc.set_ylabel('Amplitude (pA)')
    
    ax_int_amplitude_vc = fig.add_subplot(247)
    ax_int_amplitude_vc.set_xlabel('AP integral (pAs)')
    ax_int_amplitude_vc.set_ylabel('Amplitude (pA)')
    
    ax_sn_sn_vc = fig.add_subplot(248)
    ax_sn_sn_vc.set_xlabel('SNR on trace')
    ax_sn_sn_vc.set_ylabel('SNR on dtrace')
    
    # =============================================================================
    # ax_hw_amplitude_vc_mean = fig.add_subplot(224, sharex=ax_hw_amplitude_vc,sharey=ax_hw_amplitude_vc)
    # ax_hw_amplitude_vc_mean.set_xlabel('Halfwidth (ms)')
    # ax_hw_amplitude_vc_mean.set_ylabel('Amplitude (pA)')
    # =============================================================================
    
    cc_points_hw_amplitude, = ax_hw_amplitude_cc.plot(ap_data['ap_halfwidth'][ccidx],ap_data['ap_amplitude'][ccidx]*1000,'ko',ms=0.5, picker=pickertolerance)
    vc_points_hw_amplitude, = ax_hw_amplitude_vc.plot(ap_data['ap_halfwidth'][vcidx],ap_data['ap_amplitude'][vcidx]*1e12,'ko',ms=0.5, picker=pickertolerance)
    cc_points_fw_amplitude, = ax_fw_amplitude_cc.plot(ap_data['ap_full_width'][ccidx],ap_data['ap_amplitude'][ccidx]*1000,'ko',ms=0.5, picker=pickertolerance)
    vc_points_fw_amplitude, = ax_fw_amplitude_vc.plot(ap_data['ap_full_width'][vcidx],ap_data['ap_amplitude'][vcidx]*1e12,'ko',ms=0.5, picker=pickertolerance)
    cc_points_int_amplitude, = ax_int_amplitude_cc.plot(ap_data['ap_integral'][ccidx]*1000,ap_data['ap_amplitude'][ccidx]*1000,'ko',ms=0.5, picker=pickertolerance)
    vc_points_int_amplitude, = ax_int_amplitude_vc.plot(ap_data['ap_integral'][vcidx]*1e12,ap_data['ap_amplitude'][vcidx]*1e12,'ko',ms=0.5, picker=pickertolerance)
    cc_points_sn_sn, = ax_sn_sn_cc.plot(ap_data['ap_snr_v'][ccidx],ap_data['ap_snr_dv'][ccidx],'ko',ms=0.5, picker=pickertolerance)
    vc_points_sn_sn, = ax_sn_sn_vc.plot(ap_data['ap_snr_v'][vcidx],ap_data['ap_snr_dv'][vcidx],'ko',ms=0.5, picker=pickertolerance)
    
    cc_points = [cc_points_hw_amplitude,cc_points_fw_amplitude,cc_points_int_amplitude,cc_points_sn_sn]
    vc_points = [vc_points_hw_amplitude,vc_points_fw_amplitude,vc_points_int_amplitude,vc_points_sn_sn]
    for cellidx,uniquecell in enumerate(uniquecells):
        #idxs = np.where((cell_IDs==uniquecell)&ccidx)[0]
        idxs = cell_stats['current clamp']['ap_list_idx'][cellidx]
        ax_hw_amplitude_cc.plot(ap_data['ap_halfwidth'][idxs],ap_data['ap_amplitude'][idxs]*1000,'o',ms=1, picker=0)
        ax_fw_amplitude_cc.plot(ap_data['ap_full_width'][idxs],ap_data['ap_amplitude'][idxs]*1000,'o',ms=1, picker=0)
        ax_int_amplitude_cc.plot(ap_data['ap_integral'][idxs]*1000,ap_data['ap_amplitude'][idxs]*1000,'o',ms=1, picker=0)
        ax_sn_sn_cc.plot(ap_data['ap_snr_v'][idxs],ap_data['ap_snr_dv'][idxs],'o',ms=1, picker=0)
    # =============================================================================
    #     ax_hw_amplitude_cc_mean. errorbar(cell_stats['current clamp']['ap_halfwidth_mean'][cellidx],
    #                                       cell_stats['current clamp']['ap_amplitude_mean'][cellidx], 
    #                                       xerr=cell_stats['current clamp']['ap_halfwidth_sd'][cellidx], 
    #                                       yerr=cell_stats['current clamp']['ap_amplitude_sd'][cellidx],
    #                                       fmt = 'o')   
    # =============================================================================
        
        #idxs = np.where((cell_IDs==uniquecell)&vcidx)[0]
        idxs = cell_stats['voltage clamp']['ap_list_idx'][cellidx]
        ax_hw_amplitude_vc.plot(ap_data['ap_halfwidth'][idxs],ap_data['ap_amplitude'][idxs]*1e12,'o',ms=1, picker=0)
        ax_fw_amplitude_vc.plot(ap_data['ap_full_width'][idxs],ap_data['ap_amplitude'][idxs]*1e12,'o',ms=1, picker=0)
        ax_int_amplitude_vc.plot(ap_data['ap_integral'][idxs]*1e12,ap_data['ap_amplitude'][idxs]*1e12,'o',ms=1, picker=0)
        ax_sn_sn_vc.plot(ap_data['ap_snr_v'][idxs],ap_data['ap_snr_dv'][idxs],'o',ms=1, picker=0)
    # =============================================================================
    #     ax_hw_amplitude_vc_mean.errorbar(cell_stats['voltage clamp']['ap_halfwidth_mean'][cellidx],
    #                                       cell_stats['voltage clamp']['ap_amplitude_mean'][cellidx], 
    #                                       xerr=cell_stats['voltage clamp']['ap_halfwidth_sd'][cellidx], 
    #                                       yerr=cell_stats['voltage clamp']['ap_amplitude_sd'][cellidx],
    #                                       fmt = 'o')   #break
    # =============================================================================
    outdata = {'ap_data':ap_data,
               'cell_stats':cell_stats,
               'figure_handle':fig,
               'vcidx_where':vcidx_where,
               'ccidx_where':ccidx_where,
               'cell_IDs':cell_IDs,
               'uniquecells':uniquecells,
               
               
               }
    
    axesdata ={'cc_points':cc_points,
               'vc_points':vc_points,
               'ax_hw_amplitude_cc':ax_hw_amplitude_cc,
               'ax_fw_amplitude_cc':ax_fw_amplitude_cc,
               'ax_int_amplitude_cc':ax_int_amplitude_cc,
               'ax_sn_sn_cc':ax_sn_sn_cc,
               'ax_hw_amplitude_vc':ax_hw_amplitude_vc,
               'ax_fw_amplitude_vc':ax_fw_amplitude_vc,
               'ax_int_amplitude_vc':ax_int_amplitude_vc,
               'ax_sn_sn_vc':ax_sn_sn_vc,
               'cc_points_hw_amplitude':cc_points_hw_amplitude,
               'vc_points_hw_amplitude':vc_points_hw_amplitude,
               'cc_points_fw_amplitude':cc_points_fw_amplitude,
               'vc_points_fw_amplitude':vc_points_fw_amplitude,
               'cc_points_int_amplitude':cc_points_int_amplitude,
               'vc_points_int_amplitude':vc_points_int_amplitude,
               'cc_points_sn_sn':cc_points_sn_sn,
               'vc_points_sn_sn':vc_points_sn_sn,
               'data':outdata}
    
    fig.canvas.axesdata=axesdata
    cid = fig.canvas.mpl_connect('pick_event', ap_scatter_onpick)
    
    return outdata 

def cell_clustering_plots(recording_mode = 'current clamp',
                          minapnum = 10,
                          minsnr = 10,
                          isi_percentile_needed = 3,
                          features_needed = ['ap_halfwidth_median',
                                             'ap_full_width_median',
                                             'ap_rise_time_median',
                                             'ap_peak_to_trough_time_median',
                                             'ap_integral_median',
                                             'ap_isi_percentiles',
                                             'ap_ahp_amplitude_median',
                                             'ap_amplitude_median'],
                         TSNE_perplexity = 20,
                         TSNE_learning_rate = 15,
                         UMAP_learning_rate = .5,
                         UMAP_n_neigbors =20,
                         UMAP_min_dist = 0.01,
                         UMAP_n_epochs = 1000,
                         pickertolerance = 5 ):
    reducer = umap.UMAP()
    reducer.verbose=True

    reducer.learning_rate = UMAP_learning_rate
    reducer.n_neigbors =UMAP_n_neigbors
    reducer.min_dist = UMAP_min_dist
    reducer.n_epochs=UMAP_n_epochs
    
    cell_id_features = ephys_cell_attached.Cell.primary_key
    cell_id_features.append('cell_type')
    average_wave_features = ['average_ap_wave','average_ap_wave_time']
    features_all = features_needed.copy()
    features_all.extend(cell_id_features)
    features_all.extend(average_wave_features)
    apfeatureslist = (ephys_cell_attached.Cell()*ephysanal_cell_attached.CellSpikeParameters()&'ap_snr_mean > {}'.format(minsnr)&'recording_mode = "{}"'.format(recording_mode)& 'ap_quantity > {}'.format(minapnum)).fetch(*features_all)
    ap_cell_data = dict()
    for data,data_name in zip(apfeatureslist,features_all):
        if data_name not in features_needed:
            continue
        if data_name == 'ap_isi_percentiles':
            percentilearray = list()
            for perc in data:
                percentilearray.append(1/((perc[isi_percentile_needed-1])))
            ap_cell_data['FR_percentile_{}'.format(isi_percentile_needed)]=np.asarray(percentilearray)
        else:
            ap_cell_data[data_name] = data  
    cell_id = dict()
    for data,data_name in zip(apfeatureslist,features_all):
        if data_name not in cell_id_features:
            continue
        cell_id[data_name] = data  
    apwaves_dict=dict()
    for data,data_name in zip(apfeatureslist,features_all):
        if data_name in average_wave_features:
            apwaves_dict[data_name]=data
    #average_ap_waves,average_ap_wave_times = (ephysanal_cell_attached.CellSpikeParameters()&'ap_snr_mean > {}'.format(minsnr)&'recording_mode = "{}"'.format(recording_mode)& 'ap_quantity > {}'.format(minapnum)).fetch('average_ap_wave','average_ap_wave_time')
    # =============================================================================
    # #%%average AP waveforms
    # average_ap_waves,average_ap_wave_times = (ephysanal_cell_attached.CellSpikeParameters()&'ap_snr_mean > {}'.format(minsnr)&'recording_mode = "{}"'.format(recording_mode)& 'ap_quantity > {}'.format(minapnum)).fetch('average_ap_wave','average_ap_wave_time')
    # #%
    # neededlen = 150
    # apwaves = list()
    # for x,y in zip(average_ap_wave_times,average_ap_waves):
    #     if len(x)==neededlen:
    #         apwaves.append(y)
    #     else:
    #         f = interpolate.interp1d(np.arange(0,len(y),1),y)
    #         y_upsampled = f(np.arange(0,len(y)-1,len(y)/neededlen))  
    #         missingpoints = neededlen- len(y_upsampled)
    #         y_upsampled = np.concatenate([np.ones(int(np.floor(missingpoints/2)))*y_upsampled[0],y_upsampled,np.ones(int(np.ceil(missingpoints/2)))*y_upsampled[-1]])
    #         apwaves.append(y_upsampled)
    # apwaves = np.asarray(apwaves)    
    # idxtodel = np.argmin(apwaves[:,40],0)
    # apwaves[idxtodel,:]=apwaves[idxtodel-1,:] #HOTFIX removing the outlier
    # dv_normalized =  np.diff(apwaves,1)/np.max(np.diff(apwaves,1),1)[:,None]#apwaves/np.max(apwaves,1)[:,None]#np.diff(apwaves,1)#stats.zscore(np.diff(apwaves,1),0)#stats.zscore(apwaves,1)#  # 
    # v_normalized = apwaves/np.max(apwaves,1)[:,None]
    # big_matrix_concatenated = np.concatenate([v_normalized,dv_normalized],1)
    # #%
    # big_matrix_normalized = stats.zscore(big_matrix_concatenated,0)[:,np.percentile(np.abs(big_matrix_concatenated),80,axis=0)>.1]#*(np.mean(big_matrix_normalized,0)>.1)
    # #%
    # #big_matrix_normalized=dv_normalized
    # # =============================================================================
    # # ap_df = pd.DataFrame(ap_cell_data) 
    # # ap_df_values = ap_df.values
    # # ap_df_values[idxtodel,:]=ap_df_values[idxtodel-1,:] #HOTFIX removing the outlier
    # # big_matrix_normalized = np.concatenate([big_matrix_normalized,stats.zscore(ap_df_values,0) ],1)
    # # =============================================================================
    # =============================================================================
    #%
    
    
    ap_df = pd.DataFrame(ap_cell_data) 
    ap_df_values = ap_df.values
    #ap_df_values[idxtodel,:]=ap_df_values[idxtodel-1,:] #HOTFIX removing the outlier
    big_matrix = ap_df.values  
    big_matrix_normalized = stats.zscore(ap_df.values,0) 
    #big_matrix_normalized[big_matrix_normalized>4]=4
    #% PCA
    # Z-scored values..
    PCA_max_components=np.min([20,big_matrix_normalized.shape[1]])
    
    pca = PCA(n_components=PCA_max_components)      
    pcs = pca.fit_transform(big_matrix_normalized)##volumes_matrix
    #%
    noise_constant = 100
    non_noise_components = np.abs(pca.components_[:,-1]**2)*pca.explained_variance_ratio_<noise_constant*np.max(np.abs(pca.components_[:,-1]**2)*pca.explained_variance_ratio_)
    #big_matrix_normalized = pcs[:,non_noise_components]
    
    fig = plt.figure(figsize = [10,10])
    ax_pca = fig.add_subplot(331, projection='3d')
    ax_tsne = fig.add_subplot(332)
    ax_trace = fig.add_subplot(333)
    ax_pca_var = fig.add_subplot(334)
    ax_umap = fig.add_subplot(335)
    ax_prop_vs_prop_1 = fig.add_subplot(336)
    ax_components = fig.add_subplot(337)
    ax_clust = fig.add_subplot(338)
    
    propnames = list(ap_cell_data.keys())
    selectedpropertyindices = [0,1]
    points_prop_vs_prop_1, = ax_prop_vs_prop_1.plot(ap_cell_data[propnames[selectedpropertyindices[0]]],ap_cell_data[propnames[selectedpropertyindices[1]]],'ko',picker=pickertolerance)
    idx = (cell_id['cell_type'] == 'pyr')
    ax_prop_vs_prop_1.plot(ap_cell_data[propnames[selectedpropertyindices[0]]][idx],ap_cell_data[propnames[selectedpropertyindices[1]]][idx],'ro')
    idx = (cell_id['cell_type'] == 'int')
    ax_prop_vs_prop_1.plot(ap_cell_data[propnames[selectedpropertyindices[0]]][idx],ap_cell_data[propnames[selectedpropertyindices[1]]][idx],'bo')
    
    ax_prop_vs_prop_1.set_xlabel(propnames[selectedpropertyindices[0]])
    ax_prop_vs_prop_1.set_ylabel(propnames[selectedpropertyindices[1]])
    
    points_pca, = ax_pca.plot(pcs[:,0],pcs[:,1],pcs[:,2], 'ko',alpha = 1,picker=pickertolerance) #+' projecting, motor related'
    idx = (cell_id['cell_type'] == 'pyr')
    ax_pca.plot(pcs[idx,0],pcs[idx,1],pcs[idx,2], 'ro',alpha = 1,label='putative pyramidal cell')
    idx = (cell_id['cell_type'] == 'int')   
    ax_pca.plot(pcs[idx,0],pcs[idx,1],pcs[idx,2], 'bo',alpha = 1,label='putative interneuron')
    ax_pca.legend()
    ax_pca.set_xlabel('PC{}'.format(0+1))
    ax_pca.set_ylabel('PC{}'.format(1+1))
    ax_pca.set_zlabel('PC{}'.format(2+1))
    ax_pca_var.plot(np.arange(1,len(pca.explained_variance_ratio_[non_noise_components])+1),np.cumsum(pca.explained_variance_ratio_[non_noise_components]),'ko-')
    ax_pca_var.set_ylabel('Cumulative explained variance')
    ax_pca_var.set_xlabel('PCs')
    ax_pca_var.set_ylim([0,1])
    #%
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=TSNE_perplexity, n_iter=1000,learning_rate=TSNE_learning_rate)
    tsne_results = tsne.fit_transform(big_matrix_normalized)#pcs
    
    
    points_tsne, = ax_tsne.plot(tsne_results[:,0],tsne_results[:,1],'ko',ms=8,picker=pickertolerance)
    
    idx = (cell_id['cell_type'] == 'pyr')
    ax_tsne.plot(tsne_results[idx,0],tsne_results[idx,1],'ro',label='putative pyramidal cell',ms=8, alpha = 1)#+' projecting, motor related'
    idx = (cell_id['cell_type'] == 'int')
    ax_tsne.plot(tsne_results[idx,0],tsne_results[idx,1],'bo',label='putative interneuron',ms=8, alpha = 1)#+' projecting, motor related'
    # =============================================================================
    # for i,txt in enumerate(cell_names):
    #     ax_tsne.annotate(txt,(tsne_results[i,0],tsne_results[i,1]))
    # =============================================================================
    ax_tsne.set_title('tSNE - perplexity: {}, learning rate: {}'.format(TSNE_perplexity,TSNE_learning_rate))    
    plt.legend()
    #%
    
    #% UMAP
    
    umap_results = reducer.fit_transform(big_matrix_normalized)
    #plt.scatter(embedding[:, 0],embedding[:, 1])
    
    
    points_umap, = ax_umap.plot(umap_results[:,0],umap_results[:,1],'ko',ms=8,picker=pickertolerance)
    
    idx = (cell_id['cell_type'] == 'pyr')
    ax_umap.plot(umap_results[idx,0],umap_results[idx,1],'ro',label='putative pyramidal cell',ms=8, alpha = 1)#+' projecting, motor related'
    idx = (cell_id['cell_type'] == 'int')
    ax_umap.plot(umap_results[idx,0],umap_results[idx,1],'bo',label='putative interneuron',ms=8, alpha = 1)
    ax_umap.set_title('uMAP - neighbors: {}, min_dist: {}, learning rate: {}'.format(reducer.n_neigbors,
                                                                                     reducer.min_dist,
                                                                                     reducer.learning_rate))        
    
    plt.legend()
    
    
    #%
    
    
    image_pca_components = ax_components.imshow(pca.components_, picker=True)
    for idx in np.where(non_noise_components==False)[0]:
        ax_components.plot([0,pca.components_.shape[1]-1],[idx,idx],'r-',linewidth = 5)
    ax_components.set_xlabel('features')
    ax_components.set_xticks(range(0,len(list(ap_cell_data.keys()))))
    ax_components.set_xticklabels(list(ap_cell_data.keys()))
    ax_components.tick_params(labelrotation = 90)
    #plt.xticks(range(0,len(list(ap_cell_data.keys()))),list(ap_cell_data.keys()),rotation='vertical')
    ax_components.set_ylabel('PCs')
    #%
    linked = linkage(big_matrix_normalized,'complete')#'ward')#'weighted')#'average')#big_matrix_normalized
    
    #ax_clust.figure()
    R = dendrogram(linked,
                   orientation='top')
    #%
    for i,cell_ID in enumerate(R['ivl']):
        cell_idx = int(cell_ID)
        if cell_id['cell_type'][cell_idx] == 'pyr':
            color = 'r'
        elif cell_id['cell_type'][cell_idx] == 'int':
            color = 'b'
        else:
            color = 'k'
        ax_clust.plot(5+i*10,0,color+'o',markersize = 15)
    ax_clust.tick_params(axis='x', which='major', labelsize=8) 
    #trace_fig = plt.figure(figsize = [10,10])
    data_out = {'umap_results':umap_results,
               'reducer':reducer,
               'tsne_results':tsne_results,
               'pca':pca,
               'pcs':pcs,
               'big_matrix_normalized':big_matrix_normalized,
               'ap_cell_data':ap_cell_data,
               'cell_id':cell_id,
               'ap_df':ap_df}
    for key in apwaves_dict.keys():
        data_out[key]=apwaves_dict[key]
    axesdata = {'points_tsne':points_tsne,
                'points_umap':points_umap,
                'points_pca':points_pca,
                'points_prop_vs_prop_1':points_prop_vs_prop_1,
                'image_pca_components': image_pca_components,
                'fig':fig,
                'ax_pca':ax_pca,
                'ax_tsne':ax_tsne,
                'ax_trace':ax_trace,
                'ax_pca_var':ax_pca_var,
                'ax_umap':ax_umap,
                'ax_clust':ax_clust,
                'ax_components':ax_components,
                'ax_prop_vs_prop_1':ax_prop_vs_prop_1,
                'data':data_out,
                'trace_fig':None}
    fig.canvas.axesdata=axesdata
    fig.canvas.idxsofar = np.asarray([],int)
    fig.canvas.selectedpropertyindices = selectedpropertyindices 
    cid = fig.canvas.mpl_connect('pick_event', ap_cell_onpick)
#%%

#%%
def ap_cell_onpick(event):
    #print([event.mouseevent.__dict__])
    
    axesdata = event.canvas.axesdata
    data = axesdata['data']
    if not(event.artist == axesdata['points_tsne'] or event.artist == axesdata['points_umap'] or event.artist == axesdata['image_pca_components'] or event.artist == axesdata['points_prop_vs_prop_1']):
        return None
    if event.artist == axesdata['image_pca_components']:
        idxsofar = event.canvas.idxsofar
        new_propidx = int(np.floor(event.mouseevent.xdata+.5))
        if event.mouseevent.button == 1:
            event.canvas.selectedpropertyindices[0] = new_propidx
        elif event.mouseevent.button ==3:
            event.canvas.selectedpropertyindices[1] = new_propidx
    else:
        cell_idx = event.ind[0]
        if cell_idx in event.canvas.idxsofar:
            idxsofar = event.canvas.idxsofar[event.canvas.idxsofar != cell_idx]
        else:
            idxsofar = np.concatenate([event.canvas.idxsofar,[cell_idx]])
    try:
        for sel in axesdata['image_pca_components'].selected:
            sel.remove()        
    except:
        pass
    try:
        for sel in axesdata['points_tsne'].selected:
            sel.remove()
    except:
        pass
    try:
        for sel in axesdata['points_umap'].selected:
            sel.remove()
    except:
        pass
    try:
        for sel in axesdata['points_pca'].selected:
            sel.remove()
    except:
        pass
    try:
        for sel in axesdata['points_prop_vs_prop_1'].selected:
            sel.remove()
    except:
        pass
    componentdims = data['pca'].components_.shape
    propidxs = event.canvas.selectedpropertyindices
    pointslist = list()
    for i,color in enumerate(['b','r']):
        sel, = axesdata['ax_components'].plot([propidxs[i]-.4,propidxs[i]+.4,propidxs[i]+.4,propidxs[i]-.4,propidxs[i]-.4],[0-.4,0-.4,componentdims[0]-.6,componentdims[0]-.6,0-.4],'{}-'.format(color),linewidth=4)
        pointslist.append(sel)
    axesdata['image_pca_components'].selected = pointslist
    
    pointslist = list()
    axesdata['ax_umap'].set_prop_cycle(None)
    for cell_idx in idxsofar:
        sel, = axesdata['ax_umap'].plot(data['umap_results'][cell_idx,0],data['umap_results'][cell_idx,1],'x',ms=14,markeredgewidth=4)
        pointslist.append(sel)
    axesdata['points_umap'].selected = pointslist
    
    pointslist = list()
    axesdata['ax_tsne'].set_prop_cycle(None)
    for cell_idx in idxsofar:
        sel, = axesdata['ax_tsne'].plot(data['tsne_results'][cell_idx,0],data['tsne_results'][cell_idx,1],'x',ms=14,markeredgewidth=4)
        pointslist.append(sel)
    axesdata['points_tsne'].selected = pointslist
    
    axesdata['ax_trace'].cla()
    axesdata['ax_trace'].set_prop_cycle(None)
    for cell_idx in idxsofar:
        time = data['average_ap_wave_time'][cell_idx]
        trace = data['average_ap_wave'][cell_idx]
        axesdata['ax_trace'].plot(time,trace,'-')
    axesdata['ax_trace'].set_xlabel('time (ms)')
    
    axesdata['ax_prop_vs_prop_1'].cla()
    
    propnames = list(data['ap_cell_data'].keys())
    selectedpropertyindices = event.canvas.selectedpropertyindices
    
    points_prop_vs_prop_1, = axesdata['ax_prop_vs_prop_1'].plot(data['ap_cell_data'][propnames[selectedpropertyindices[0]]],data['ap_cell_data'][propnames[selectedpropertyindices[1]]],'ko',picker=5)
    event.canvas.axesdata['points_prop_vs_prop_1'] = points_prop_vs_prop_1
    idx = (data['cell_id']['cell_type'] == 'pyr')
    axesdata['ax_prop_vs_prop_1'].plot(data['ap_cell_data'][propnames[selectedpropertyindices[0]]][idx],data['ap_cell_data'][propnames[selectedpropertyindices[1]]][idx],'ro')
    idx = (data['cell_id']['cell_type'] == 'int')
    axesdata['ax_prop_vs_prop_1'].plot(data['ap_cell_data'][propnames[selectedpropertyindices[0]]][idx],data['ap_cell_data'][propnames[selectedpropertyindices[1]]][idx],'bo')
    axesdata['ax_prop_vs_prop_1'].set_xlabel(propnames[selectedpropertyindices[0]])
    axesdata['ax_prop_vs_prop_1'].set_ylabel(propnames[selectedpropertyindices[1]])
    axesdata['ax_prop_vs_prop_1'].set_prop_cycle(None)
    for cell_idx in idxsofar:
        sel, = axesdata['ax_prop_vs_prop_1'].plot(data['ap_cell_data'][propnames[selectedpropertyindices[0]]][cell_idx],data['ap_cell_data'][propnames[selectedpropertyindices[1]]][cell_idx],'x',ms=14,markeredgewidth=4)
    
    
    
    event.canvas.idxsofar = idxsofar
    event.canvas.draw()