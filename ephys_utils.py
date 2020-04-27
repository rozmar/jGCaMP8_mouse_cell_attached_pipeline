import matplotlib.pyplot as plt
import numpy as np
from pywavesurfer import ws
import utils
from matplotlib import cm as colormap

#%% if __name__ == "__main__":
def extract_tuning_curve(WS_path = './test/testWS.h5',vis_path = './test/test.mat',plot_data=False,plot_title = None):
    #%%
# =============================================================================
#     WS_path = './test/testWS.h5'
#     vis_path = './test/test.mat'
#     plot_data = True
# =============================================================================
    vis = utils.processVisMat(vis_path)
    ws_data = ws.loadDataFile(filename=WS_path, format_string='double' )
    
    units = np.array(ws_data['header']['AIChannelUnits']).astype(str)
    channelnames = np.array(ws_data['header']['AIChannelNames']).astype(str)
    ephysidx = (channelnames == 'Voltage').argmax()
    framatimesidx = (channelnames == 'FrameTrigger').argmax()
    
    sweep = ws_data['sweep_0001']
    sRate = ws_data['header']['AcquisitionSampleRate'][0][0]
    voltage = sweep['analogScans'][ephysidx,:]
    if units[ephysidx] == 'mV':
        voltage =voltage/1000
        units[ephysidx] = 'V'
    frame_trigger = sweep['analogScans'][framatimesidx,:]
    
    # get frame indices
    frame_idx = utils.getEdges(frame_trigger, minDist_samples=20, diffThresh=.02, edge = 'positive')
    
    # check to make sure num detected frames matches num frames in mat file
    nFrames_ws = len(frame_idx)
    nFrames_vis = vis['total_frames']
    if not (nFrames_ws == nFrames_vis):
        print('Number of frames in WS file (' + str(nFrames_ws) + ') does not match num frames in MAT file (' + str(nFrames_vis) + ')')
        raise ValueError('Number of frames in WS file (' + str(nFrames_ws) + ') does not match num frames in MAT file (' + str(nFrames_vis) + ')')
    
    t_ephys = np.arange(len(voltage))/sRate
    #
    v_filt = utils.gaussFilter(voltage,sRate,sigma = .00005)
    #%
    # AP indices in voltage trace
    AP_idx,AP_snr,noise_level = utils.findAPs(v_filt, sRate)
    v_filt = utils.hpFilter(v_filt, 10, 1, sRate)
    AP_time = t_ephys[AP_idx]
    
    angle = vis['angle']
    nAngles = len(angle)
    spikes = np.zeros(nAngles)
    spikes_base = np.zeros(nAngles)
    
    vis_stim_fs = vis['fs']# visual stim sampling rate
    stim_dur_samples = vis['stim_dur'] * vis_stim_fs
    gray_dur_samples = vis['gray_dur'] * vis_stim_fs
    stim_init_samples = vis['stim_init']
    
    for i,start_stim_idx in enumerate(stim_init_samples):
        end_stim_idx = start_stim_idx + stim_dur_samples-1
        start_gray_idx = start_stim_idx - gray_dur_samples
        nSpikes_base = np.sum((AP_idx >= frame_idx[start_gray_idx]) & (AP_idx <= frame_idx[start_stim_idx]))
        nSpikes = np.sum((AP_idx >= frame_idx[start_stim_idx]) & (AP_idx <= frame_idx[end_stim_idx]))
        # calculate num spikes in interval
        spikes[i] = nSpikes
        spikes_base[i] = nSpikes_base
        
    # collect all spikes and plot
    uniqueAngles = np.unique(angle)
    totalSpikes = np.zeros_like(uniqueAngles,dtype = int)
    meanSpikes = np.zeros_like(uniqueAngles,dtype = float)
    totalSpikes_baseline= np.zeros_like(uniqueAngles,dtype = int)
    meanSpikes_baseline = np.zeros_like(uniqueAngles,dtype = float)
    
    allSpikes = list()
    allSpikes_baseline = list()

    for i,a in enumerate(uniqueAngles):
        
        allSpikes.append(np.asarray(spikes[angle == a]))
        allSpikes_baseline.append(np.asarray(spikes_base[angle == a]))
        
        totalSpikes[i] = sum(spikes[angle == a])
        meanSpikes[i] = np.mean(spikes[angle == a])
        
        totalSpikes_baseline[i] = sum(spikes_base[angle == a])
        meanSpikes_baseline[i] = np.mean(spikes_base[angle == a])
        
    if plot_data:
        # calculate insantenous firing rates  
        ap_time_back = .003
        ap_time_forward = .005
        cmap = colormap.get_cmap('jet')
        
        
        ap_step_back = int(sRate*ap_time_back)
        ap_step_forward = int(sRate*ap_time_forward)
        ap_time = np.arange(-ap_step_back,ap_step_forward)/sRate*1000
        
# =============================================================================
#         fr_kernel = np.ones(int(firing_rate_window/frbinwidth))/(firing_rate_window/frbinwidth)
#         fr_bincenters = np.arange(frbinwidth/2,np.max(t_ephys)+frbinwidth,frbinwidth)
#         fr_binedges = np.concatenate([fr_bincenters-frbinwidth/2,[fr_bincenters[-1]+frbinwidth/2]])
#         fr_e = np.histogram(AP_time,fr_binedges)[0]/frbinwidth
#         fr_e = np.convolve(fr_e, fr_kernel,'same')
# =============================================================================
        
        fr_e, fr_bincenters = utils.AP_times_to_rate(AP_time,firing_rate_window=2,frbinwidth = 0.01)
        
        frame_times = t_ephys[frame_idx]
        stim_start_t = frame_times[stim_init_samples]
        
        fig = plt.figure(figsize = [15,15])
        ax_raw = fig.add_subplot(421)
        ax_raw.plot(t_ephys, v_filt,'k-') # plot v trace
        ax_raw.plot(AP_time, v_filt[AP_idx], 'ro',alpha = .5) # show peaks
        ax_raw.set_ylabel(units[ephysidx])
        ax_raw.set_xlim([t_ephys[0],t_ephys[-1]])
        ax_raw.set_xlabel('time (s)')
        
        ax_rate = fig.add_subplot(422)
        ax_rate.plot(fr_bincenters, fr_e, 'k-')
        ax_rate.set_ylabel("Firing rate")
        for stim_start_t_now,a in zip(stim_start_t,angle):
            ax_rate.fill_between(t_ephys, 
                                 0, 
                                 np.max(fr_e), 
                                 where= (t_ephys > stim_start_t_now) & (t_ephys < stim_start_t_now+vis['stim_dur']), 
                                 facecolor=cmap(a/360), 
                                 alpha=0.5)
        ax_rate.set_xlim([t_ephys[0],t_ephys[-1]])
        ax_rate.set_xlabel('time (s)')
        
        ax_snr = fig.add_subplot(423)
        ax_snr.plot(AP_time, AP_snr, 'ro',alpha = .5)
        ax_snr.set_ylabel('SNR')
        ax_snr.set_xlim([t_ephys[0],t_ephys[-1]])
        ax_snr.set_xlabel('time (s)')
        
        ax_noiselevel = ax_snr.twinx()
        ax_noiselevel.plot(t_ephys,noise_level,'k-')
        
        
        ax_tot_spikes = fig.add_subplot(424)
        ax_tot_spikes.bar(uniqueAngles, totalSpikes, width=30,color = cmap(uniqueAngles/360) )
        ax_tot_spikes.set_xlabel("Angle")
        ax_tot_spikes.set_ylabel("Total num spikes")
        ax_tot_spikes.set_xticks(uniqueAngles)
        
        
        ax_rel_spikes = fig.add_subplot(426)
        ax_rel_spikes.bar(uniqueAngles, meanSpikes-meanSpikes_baseline, width=30,color = cmap(uniqueAngles/360))
        ax_rel_spikes.plot(uniqueAngles,np.zeros(len(uniqueAngles)),'k-')
        ax_rel_spikes.set_xlabel("Angle")
        ax_rel_spikes.set_ylabel("mean AP num change")
        ax_rel_spikes.set_xticks(uniqueAngles)
        
        ax_ap = fig.add_subplot(425)
        for ap_idx_now in AP_idx:
            try:
                ax_ap.plot(ap_time,v_filt[ap_idx_now-ap_step_back:ap_idx_now+ap_step_forward])  
            except:
                pass
        ax_ap.set_xlim([ap_time[0],ap_time[-1]])
        ax_ap.set_ylabel('mV')
        ax_ap.set_xlabel('ms')
        if plot_title:
            ax_raw.set_title(plot_title)
            
        plt.show()
    
    outdata = {'uniqueAngles':uniqueAngles,
               'allSpikes':np.asarray(allSpikes),
               'allSpikes_baseline':np.asarray(allSpikes_baseline),
               'totalSpikes':totalSpikes,
               'meanSpikes':meanSpikes,
               'totalSpikes_baseline':totalSpikes_baseline,
               'meanSpikes_baseline':meanSpikes_baseline,
               'ephys_t':t_ephys,
               'ephys_v_filt':v_filt,
               'ap_idx':AP_idx,
               'frame_times':t_ephys[frame_idx]}
    
    
 #%%   
    return(outdata)