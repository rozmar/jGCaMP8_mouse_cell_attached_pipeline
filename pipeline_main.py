import os

import numpy as np
from pywavesurfer import ws
import h5py 
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import scipy.signal as signal

import utils_pipeline
import utils_ephys
import utils

import notebook_google.notebook_main as online_notebook

import pandas as pd
import re
import scipy
from ingest import datapipeline_metadata
#%% set parameters

s2p_params = {'cell_soma_size':20, # microns - obsolete
              'max_reg_shift':20, # microns
              'max_reg_shift_NR': 10, # microns
              'block_size': 60, #microns
              'smooth_sigma':.5, # microns
              'smooth_sigma_time':.001, #seconds,
              'basedir':'/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig', # folder where the raw ephys and movies are found
              'basedir_out':'/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig', #'/home/rozmar/Data/Calcium_imaging/suite2p_fast/Genie_2P_rig'
              'basedir_out_slow':'/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig',
              'overwrite': False} # folder where the suite2p output is saved

metadata_params = {'notebook_name':"Genie Calcium Imaging",
                   'metadata_dir':"/home/rozmar/Data/Calcium_imaging/Metadata"}
anatomy_metadata_params = {'notebook_name':"Genie anit-GFP immunostaining",
                           'metadata_dir':"/home/rozmar/Data/Anatomy/Metadata"}

ephys_params = {'basedir':'/home/rozmar/Network/rozsam-nas-01-smb/Calcium_imaging/raw/Genie_2P_rig'}#'/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig'

visual_stim_params = {'basedir': '/home/rozmar/Data/Visual_stim/raw/Genie_2P_rig'}

#%% imaging analysis parameters
F0win = 20 #s running F0 window for min-max baseline
F_filter_sigma = .001 #sigma for gaussian filtering in seconds
neu_r =0#.7 # r value for neuropil subtraction

time_back = .05 # seconds from event start when cutting out ephys and ophys responses
time_forward = .45 # seconds from event start when cutting out ephys and ophys responses

min_baseline_time =1 #seconds before an event - for bucketing
integration_time = .4 # seconds after the last AP - for bucketing
#%% registration and ROI detection
utils_pipeline.batch_run_suite2p(s2p_params)
#%%
# feeding movies in datajoint
# feeding ROIs in datajoint
# feeding ephys in datajoint
# feeding visual stim to datajoint (trials..?)
# ROI-ephys correspondance
# analysis

#%%  Manual part
# here comes a manual step when you find the corresponding ROI with suite2p in 
# each recording and update it to the google spreadsheet
#%%


#%% update metadata from google spreadsheet
online_notebook.update_metadata(metadata_params['notebook_name'],metadata_params['metadata_dir'])
online_notebook.update_metadata(anatomy_metadata_params['notebook_name'],anatomy_metadata_params['metadata_dir'])
#%% QC for imaging and ephys
event_ap_num = list()
event_ap_times = list()
event_length = list()
event_start_time = list()
event_cell_num = list()
event_roi_num = list()
event_min_ap_SNR = list()
event_ap_amplitude_accomodation = list()
event_sensor = list()
event_session = list()
event_subject = list()
event_samplingrate = list()
event_framerate = list()
event_unique_cell_id =list()
event_amplitude = list()
event_amplitude_local = list()
event_baseline_time = list()
event_r_neu=list()
event_r_neu_cell_id= list()
event_r_neu_corrcoef = list()
dFF_all = list()
dFF_local_all= list()
F0_local_all = list()
dFF_t_all = list()
V_all = list()
V_t_all = list()
event_baseline_dff = list()
ignored_aps = 0


metadata_files = os.listdir(metadata_params['metadata_dir'])
for metadata_file in metadata_files:
    if '-anm' in metadata_file and 'visual' not in metadata_file:
         df_session = pd.read_csv(os.path.join(metadata_params['metadata_dir'],metadata_file))
         session = metadata_file[:metadata_file.find('-anm')]
         subject = metadata_file[metadata_file.find('-anm')+4:-4]
    else:
        continue
         #break

    #vstimdirs=os.listdir(visual_stim_params['basedir'])
    #counter = 0
    sensor = df_session['sensor'][0]
    if len(df_session)<2:
        continue # 1 line means no movies
    for df_run in df_session.iterrows():
        
        idx = df_run[0]
        df_run = df_run[1]
        cell_num = int(df_run['Cell'])
        roi_num = df_run['suite2p-ROI-num']
        run_num =int(df_run['Run'])
        imaging_quality = df_run['good?']
        if imaging_quality ==1 and ',' not in roi_num:
            
            sessions = os.listdir(ephys_params['basedir'])
            sessiondir = os.path.join(ephys_params['basedir'],'{}-anm{}'.format(session,subject))
            cells = os.listdir(sessiondir)
            for celldir in cells:
                cell_found = False
                if 'cell{}'.format(cell_num) in celldir.lower():
                    cell_found = True
                    break
                
            if cell_found == False:  
                print('ERROR: ephys cell directory not found for session {}, subject {} cell {}'.format(session,subject,cell_num))
                break
            
            celldir = os.path.join(sessiondir,celldir)
            ephysfiles = os.listdir(celldir)
            ephysfiles_real = list()
            for ephysfile in ephysfiles:
                if '.h5' in ephysfile:
                    separators = [m.start() for m in re.finditer('_| |-', ephysfile)]
                    if int(re.findall(r'\d+', ephysfile[separators[0]+1:separators[1]])[0]) == run_num:
                        ephysfiles_real.append(ephysfile)
                            
            for ephysfile in ephysfiles_real:
                #% get ephys data
                voltage, frame_trigger, sRate, recording_mode, timestamp  , unit =  utils_ephys.load_wavesurfer_file(os.path.join(celldir,ephysfile))
                t_ephys = np.arange(len(voltage))/sRate
                
                frame_idx_pos = utils.getEdges(frame_trigger, minDist_samples=20, diffThresh=.02, edge = 'positive')
                frame_idx_neg = utils.getEdges(frame_trigger, minDist_samples=20, diffThresh=.02, edge = 'negative')
                
                if any(np.diff(frame_idx_pos)>2*np.median(np.diff(frame_idx_pos))): # frame triggers not belonging to the movie are removed
                    valid_framenum = np.argmax(np.diff(frame_idx_pos)>2*np.median(np.diff(frame_idx_pos)))+1
                    frame_idx_pos = frame_idx_pos[:valid_framenum]
                    frame_idx_neg = frame_idx_neg[:valid_framenum]
                frame_idx = np.asarray(np.mean([frame_idx_pos[:len(frame_idx_neg)],frame_idx_neg],0),dtype = int)
                frame_times = t_ephys[frame_idx]
                
                v_filt = utils.gaussFilter(voltage,sRate,sigma = .00005)
                AP_idx,AP_snr,noise_level = utils.findAPs(v_filt, sRate, SN_min = 10)
                v_filt = utils.hpFilter(v_filt, 10, 1, sRate)
                AP_times = t_ephys[AP_idx]
                
                        
                #%
                #% get extracted ROI
                suite2p_cell_path = s2p_params['basedir_out_slow'] +celldir[len(ephys_params['basedir']):]
                movie_dirs = os.listdir(suite2p_cell_path)
                for movie_dir in movie_dirs:
                    separators = [m.start() for m in re.finditer('_| |-', movie_dir)]
                    if len(separators)==1: # HOTFIX first recordings the movie name doesn't include the cell identity
                        separators=[0]+separators
                    if len(re.findall(r'\d+', movie_dir[separators[0]+1:separators[1]]))>0 and int(re.findall(r'\d+', movie_dir[separators[0]+1:separators[1]])[0]) == run_num:
                        suite2p_movie_path = os.path.join(suite2p_cell_path,movie_dir,'plane0')
                        try:
                            F_all = np.load(os.path.join(suite2p_movie_path,'F.npy'))
                        except:
                            print('no fluorescence traces found')
                            continue
                        Fneu_all = np.load(os.path.join(suite2p_movie_path,'Fneu.npy'))
                        framerate = int(1/np.mean(np.diff(frame_times)))
                        F = F_all[int(roi_num)]
                        Fneu = Fneu_all[int(roi_num)]
                        F0step= int(F0win/np.mean(np.diff(frame_times)))
                        F_orig = F.copy()
                        Fneu_orig = Fneu.copy()
                        F = utils.gaussFilter(F,framerate,sigma = F_filter_sigma)
                        Fneu = utils.gaussFilter(Fneu,framerate,sigma = F_filter_sigma)
                        if len(F) > len(frame_times):
                            print('ephys trace is shorter than expected - not enough frame times')
                            continue # no time for all frames
                        
                        
                        # - ------ calculate r

                        decay_time = 5 #s
                        decay_step = int(decay_time*framerate)
                        F_activity = np.zeros(len(frame_times))
                        for ap_time in AP_times:
                            F_activity[np.argmax(frame_times>ap_time)] = 1
                        F_activitiy_conv= np.convolve(F_activity,np.concatenate([np.zeros(decay_step),np.ones(decay_step)]),'same')
                        F_activitiy_conv[:decay_step]=1
                        needed =F_activitiy_conv==0
                        if sum(needed)> 100:
                            F_orig_filt = utils.gaussFilter(F_orig,framerate,sigma = F_filter_sigma)
                            Fneu_orig_filt = utils.gaussFilter(Fneu_orig,framerate,sigma = F_filter_sigma)
                            p=np.polyfit(Fneu_orig_filt[needed],F_orig_filt[needed],1)
                            R_neuopil_fit = np.corrcoef(Fneu_orig_filt[needed],F_orig_filt[needed])
                            r_neu_now =p[0]
                            event_r_neu.append(r_neu_now)
                            event_r_neu_cell_id.append('{}-anm{}-cell{}'.format(session,subject,cell_num))
                            event_r_neu_corrcoef.append(R_neuopil_fit[1][0])
                            neuropilvals = np.asarray([np.min(Fneu_orig_filt[needed]),np.max(Fneu_orig_filt[needed])])
                            fittedvals = np.polyval(p,neuropilvals)
                            
# =============================================================================
#                             plt.plot(Fneu_orig_filt[needed],F_orig_filt[needed],'ko')
#                             plt.plot(neuropilvals,fittedvals,'r-')
#                             plt.title([p,R_neuopil_fit])
#                             plt.ylabel('F')
#                             plt.xlabel('neuropil')
#                             plt.show()
# =============================================================================
                        else:
                            print('no part without activity, r_neu set to 0')
                            r_neu_now =0
                            event_r_neu.append(r_neu_now)
                            event_r_neu_cell_id.append('{}-anm{}-cell{}'.format(session,subject,cell_num))
                            event_r_neu_corrcoef.append(0)
                            
                        
                        # - ------ calculate r
                        
                        
                        F = utils.gaussFilter(F,framerate,sigma = F_filter_sigma)
                        Fneu = utils.gaussFilter(Fneu,framerate,sigma = F_filter_sigma)
                        F_corr = F-Fneu*p[0]#neu_r
                        p=np.polyfit(F_corr,utils.gaussFilter(F,framerate,sigma = F_filter_sigma),1)
                        F_corr =F_corr +p[1]
                        # correct neuropil fluctuations
                        
                        
                        
                        
                       # F_corr = F-Fneu*neu_r
                        
                        F0 = utils.rollingfun(F_corr, window = F0step, func = 'min')
                        F0 = utils.rollingfun(F0, window = int(F0step*1), func = 'max')
                        dFF = (F_corr-F0)/F0
                print(ephysfile)
            
                
                #% cut out ephys and ophys segments
                
                ephys_step_back = int(sRate*time_back)
                ephys_step_forward = int(sRate*time_forward)
                ephys_time = np.arange(-ephys_step_back,ephys_step_forward)/sRate*1000
                ophys_step_back = int(framerate*time_back)
                ophys_step_forward = int(framerate*time_forward)
                ophys_time = np.arange(-ophys_step_back,ophys_step_forward)/framerate*1000
                
    
                AP_times_remaining = AP_times.copy()
                AP_SNR_remaining = AP_snr.copy()
                AP_time_diff_pre = np.concatenate([[min_baseline_time*2],np.diff(AP_times_remaining)])
                
                while len(AP_times_remaining)>0:
                    if AP_time_diff_pre[0]<min_baseline_time:
                        AP_times_remaining = np.delete(AP_times_remaining,0)
                        AP_time_diff_pre = np.delete(AP_time_diff_pre,0)
                        AP_SNR_remaining = np.delete(AP_SNR_remaining,0)
                        ignored_aps += 1
                    else:
                        if len(AP_time_diff_pre) > 1:
                            ap_in_event = np.argmax(AP_time_diff_pre[1:]>integration_time)+1
                            if ap_in_event ==0:
                                ap_in_event = len(AP_time_diff_pre)
                        else:
                            ap_in_event=1
                        if AP_times_remaining[0]+time_forward>frame_times[-1]: # not collecting anymore APs if the imaging is over
                            break
                        event_ap_num.append(ap_in_event)
                        event_ap_times.append(AP_times_remaining[:ap_in_event])
                        event_start_time.append(AP_times_remaining[0])
                        event_length.append(AP_times_remaining[ap_in_event-1]-AP_times_remaining[0])
                        event_min_ap_SNR.append(np.min(AP_SNR_remaining[:ap_in_event]))
                        event_baseline_time.append(AP_time_diff_pre[0])
                        
                        v = np.ones(len(ephys_time))*np.nan
                        start_idx = np.argmin(np.abs(t_ephys-event_start_time[-1]))
                        idxs = np.arange(start_idx-ephys_step_back,start_idx+ephys_step_forward)
                        idxs_valid = (idxs>=0) & (idxs<len(t_ephys))
                        v[idxs_valid] = v_filt[idxs[idxs_valid]]
                        V_all.append(v)
                        
                        dff = np.ones(len(ophys_time))*np.nan
                        f = np.ones(len(ophys_time))*np.nan
                        start_idx = np.argmin(np.abs(frame_times-event_start_time[-1]))
                        idxs = np.arange(start_idx-ophys_step_back,start_idx+ophys_step_forward)
                        idxs_valid = (idxs>=0) & (idxs<len(frame_times))
                        dff[idxs_valid] = dFF[idxs[idxs_valid]]
                        f[idxs_valid] = F_corr[idxs[idxs_valid]]
                        f0_local = np.nanmean(f[:ophys_step_back])
                        dff_local = (f-f0_local)/f
                        dFF_all.append(dff)
                        dFF_local_all.append(dff_local)
                        F0_local_all.append(f0_local)
                        time_diff = frame_times[start_idx]-event_start_time[-1]
                        dFF_t_all.append(ophys_time+time_diff*1000)
                        
                        event_baseline_dff.append(np.nanmean(dff[:ophys_step_back]))
                        
                        
                        event_amplitude.append(np.max(dff[ophys_step_back:ophys_step_back+int((event_length[-1] + .05)*framerate)])-np.nanmean(dff[:ophys_step_back]))
                        event_amplitude_local.append(np.max(dff_local[ophys_step_back:ophys_step_back+int((event_length[-1] + .05)*framerate)])-np.nanmean(dff_local[:ophys_step_back]))
                        
                        event_cell_num.append(cell_num)
                        event_roi_num.append(roi_num)
                        event_ap_amplitude_accomodation.append(AP_SNR_remaining[ap_in_event-1]/AP_SNR_remaining[0]) 
                        event_sensor.append(sensor)
                        event_session.append(session)
                        event_subject.append(subject)
                        event_unique_cell_id.append('{}-anm{}-cell{}'.format(session,subject,cell_num))
                        
                        event_samplingrate.append(sRate)
                        event_framerate.append(framerate)
                        V_t_all.append(ephys_time)
                        
                        AP_times_remaining = np.delete(AP_times_remaining,np.arange(ap_in_event))
                        AP_time_diff_pre = np.delete(AP_time_diff_pre,np.arange(ap_in_event))
                        AP_SNR_remaining = np.delete(AP_SNR_remaining,np.arange(ap_in_event))
# =============================================================================
#             break
#         if len(event_cell_num)>0:
#             break
# =============================================================================
    # =============================================================================
    #         counter+= 1
    #     if counter ==2:
    #         break
    # =============================================================================
                        
#%% narrowing down events
    
    
bin_step = .5
bin_size = 1
    
min_accomodation = 0.5
max_baseline_dff = 1
min_ap_frequency = 40 
sensor = 'XCaMP'#'G7F'#'UF456'#
#cell_id ='20200322-anm472004-cell5'
#cell_id ='20200417-anm472001-cell4'
plot_stuff=True
#%
neuropil_F = 0  
new_amplitudes = np.asarray(event_amplitude_local)*np.asarray(F0_local_all)/(np.asarray(F0_local_all)-neuropil_F)
apnums=list()
amplitude_mean = list()
amplitude_std = list()
amplitude_median = list()
amplitude_mean_new = list()
amplitude_std_new = list()
amplitude_median_new = list()
amplitude_mean_AU = list()
amplitude_std_AU = list()
amplitude_median_AU = list()
for apnum in np.unique(event_ap_num):
    if apnum>1:
        break
    #%
    #apnum=np.unique(event_ap_num)[1]
    try:
        idxs_all = np.where(event_ap_num==apnum)[0] 
        idxs = np.where((event_ap_num==apnum) &#(np.asarray(event_unique_cell_id) == cell_id)&
                        (np.asarray(event_ap_amplitude_accomodation) > min_accomodation) &
                        (np.asarray(event_baseline_dff)<max_baseline_dff) & 
                        (np.asarray(event_length)<apnum/min_ap_frequency)&
                        (np.asarray(event_sensor) == sensor))[0] 
        framerates = np.asarray(event_framerate)[idxs]
        samplingrates = np.asarray(event_samplingrate)[idxs]
        if len(idxs)>0:
            
            apnums.append(apnum)
            amplitude_mean.append(np.mean(np.asarray(event_amplitude_local)[idxs]))
            amplitude_std.append(np.std(np.asarray(event_amplitude_local)[idxs])) 
            amplitude_median.append(np.median(np.asarray(event_amplitude_local)[idxs]))
            amplitude_mean_new.append(np.mean(np.asarray(new_amplitudes)[idxs]))
            amplitude_std_new.append(np.std(np.asarray(new_amplitudes)[idxs])) 
            amplitude_median_new.append(np.median(np.asarray(new_amplitudes)[idxs]))
            
            amplitude_mean_AU.append(np.mean(np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs]))
            amplitude_std_AU.append(np.std(np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs])) 
            amplitude_median_AU.append(np.median(np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs]))
            
            if plot_stuff:
                fig = plt.figure(figsize = [15,10])
                ax_ophys = fig.add_subplot(331)
                ax_hist_amplitude = fig.add_subplot(332)
                ax_dff0_amplitude = fig.add_subplot(333)
                
                ax_ophys_local = fig.add_subplot(334)
                ax_hist_amplitude_local = fig.add_subplot(335)
                ax_dff0_amplitude_local = fig.add_subplot(336)
                
                ax_ephys = fig.add_subplot(337)
                ax_hist1 = fig.add_subplot(338)
                ax_hist2 = fig.add_subplot(339)
                
                fig_timecourse = plt.figure(figsize = [15,10])
                #ax_global_timecourse =fig_timecourse.add_subplot(211)
                ax_local_timecourse =fig_timecourse.add_subplot(211)
                ax_fit=fig_timecourse.add_subplot(212, sharex=ax_local_timecourse)
                fig_timediff = plt.figure(figsize = [5,5])
                ax_timediff_hist =fig_timediff.add_subplot(111)
                unique_framerates=np.unique(framerates)
                for framerate in unique_framerates:
                    framerate_idx = framerates==framerate
                    x = np.squeeze(np.stack(np.asarray(dFF_t_all)[idxs][framerate_idx])).T
                    ax_timediff_hist.hist(x[6,:],10)
                    ax_timediff_hist.set_xlabel('ophys-ephys time difference (ms)')
                    y = np.stack(np.asarray(dFF_all)[idxs][framerate_idx]).T
                    ax_ophys.plot(x,y)
                    ax_ophys.plot(np.nanmean(x,1),np.nanmean(y,1),'k-',lw=2)
# =============================================================================
#                     if framerate == scipy.stats.mode(framerates).mode[0]:
#                         
#                         x_conc = np.concatenate(x)
#                         y_conc = np.concatenate(y)
#                         bin_centers = np.arange(np.min(x_conc),np.max(x_conc),bin_step)
#                         bin_mean = list()
#                         for bin_center in bin_centers:
#                             bin_mean.append(np.mean(y_conc[(x_conc>bin_center-bin_size/2) & (x_conc<bin_center+bin_size/2)]))
#                         ax_global_timecourse.plot(np.nanmean(x,1),np.nanmean(y,1),'k-',lw=2)
#                         ax_global_timecourse.plot(bin_centers,bin_mean,'r-',lw=2)
# =============================================================================
                        
                    
                    y = np.stack(np.asarray(dFF_local_all)[idxs][framerate_idx]).T
                    ax_ophys_local.plot(x,y)
                    ax_ophys_local.plot(np.nanmean(x,1),np.nanmean(y,1),'k-',lw=2)
                    if framerate == scipy.stats.mode(framerates).mode[0]:
                        
                        x_conc = np.concatenate(x)
                        y_conc = np.concatenate(y)
                        bin_centers = np.arange(np.min(x_conc),np.max(x_conc),bin_step)
                        bin_mean = list()
                        for bin_center in bin_centers:
                            bin_mean.append(np.mean(y_conc[(x_conc>bin_center-bin_size/2) & (x_conc<bin_center+bin_size/2)]))
                        #break
                        #%
                        risetime = 40
                        startidx = np.argmax(bin_centers>risetime)
                        xvals= bin_centers[startidx:]-bin_centers[startidx]
                        xvals_plot =bin_centers[startidx:]
                        yvals = bin_mean[startidx:]
                        out = scipy.optimize.curve_fit(lambda t,a,b,c,d,e: a*np.exp(-t/b) + c + d*np.exp(-t/e),  xvals,  yvals,bounds=((0,0,-0.000001,0,0),(np.inf,np.inf,0.00000001,np.inf,np.inf)))
                        yfit = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2] +out[0][3]*np.exp(-xvals/out[0][4])
                        #%
                        
                        ax_local_timecourse.plot(bin_centers,bin_mean,'r-',lw=2)
                        ax_local_timecourse.plot(np.nanmean(x,1),np.nanmean(y,1),'k-',lw=2)
                        ax_fit.plot(bin_centers,bin_mean,'ro')
                        ax_fit.plot(xvals_plot,yfit,'k-')
                        
                        if out[0][0]>out[0][3]:
                            ax_fit.set_title('tau1: {:.2f} ms, tau2: {:.2f} ms, w_tau1: {:.2f}%'.format(out[0][1],out[0][4],out[0][0]/(out[0][0]+out[0][3])))
                        else:
                            ax_fit.set_title('tau1: {:.2f} ms, tau2: {:.2f} ms, w_tau1: {:.2f}%'.format(out[0][4],out[0][1],out[0][3]/(out[0][0]+out[0][3])))
                        ax_fit.set_xlim([np.min(bin_centers),np.max(bin_centers)])
                        #break
                        ax_local_timecourse.set_title('sensor: {}, n: {}, bin size: {} ms'.format(sensor,len(idxs),bin_size))
                        
                    
                unique_samplingrates=np.unique(samplingrates)   
                for x,y in zip(np.asarray(V_t_all)[idxs],np.asarray(V_all)[idxs]):
                    ax_ephys.plot(x,y)
                    
        # =============================================================================
        #         for samplingrate in unique_samplingrates:
        #             samplingrate_idx = samplingrates==samplingrate
        #             x = np.asarray(V_t_all)[idxs][samplingrate_idx].T
        #             y = np.asarray(V_all)[idxs][samplingrate_idx].T
        #             ax_ephys.plot(x,y)
        #             ax_ephys.plot(np.nanmean(x,1),np.nanmean(y,1),'k-',lw=2)
        #         try:
        #             ax_ephys.plot(np.nanmean(np.asarray(V_t_all)[idxs].T,1),np.nanmean(np.asarray(V_all)[idxs].T,1))
        #             ax_ephys.plot(np.asarray(V_t_all)[idxs].T,np.asarray(V_all)[idxs].T)
        #             
        #         except:
        #             pass
        # =============================================================================
                
                ax_hist1.hist(np.asarray(event_length)[idxs])
                ax_hist1.set_xlabel('event length (s)')
                ax_hist2.hist(np.asarray(event_ap_amplitude_accomodation)[idxs])
                ax_hist2.set_xlabel('AP amplitude accomodation')
                ax_ophys.set_title('{} - {} AP, n={}'.format(sensor,apnum,len(idxs)))
                ax_ophys.set_ylabel('dF/F - global F0')
                ax_ophys_local.set_ylabel('dF/F - local F0')
                ax_hist_amplitude.hist(np.asarray(event_amplitude)[idxs],np.arange(0,1,.05))
                ax_hist_amplitude.set_xlabel('event amplitude (dF/F)')
                ax_hist_amplitude_local.hist(np.asarray(event_amplitude_local)[idxs],np.arange(0,1,.05))
                ax_hist_amplitude_local.set_xlabel('event amplitude (dF/F)')
                ax_dff0_amplitude.plot(np.asarray(event_baseline_dff)[idxs],np.asarray(event_amplitude)[idxs],'ko')
                ax_dff0_amplitude.set_xlabel('baseline dF/F')
                ax_dff0_amplitude.set_ylabel('amplitude dF/F')
        # =============================================================================
        #         ax_dff0_amplitude.set_ylim([0,.5])
        #         ax_dff0_amplitude_local.set_ylim([0,.5])
        # =============================================================================
                ax_dff0_amplitude_local.plot(np.asarray(event_baseline_dff)[idxs],np.asarray(event_amplitude_local)[idxs],'ko')
                ax_dff0_amplitude_local.set_xlabel('baseline dF/F')
                ax_dff0_amplitude_local.set_ylabel('amplitude dF/F')
                plt.show()
                #break
    except:
        pass
#%%
        
    
#%%
import seaborn as sns
idxs = np.where((np.asarray(event_ap_amplitude_accomodation) > min_accomodation) &#(np.asarray(event_unique_cell_id) == cell_id)&
                (np.asarray(event_baseline_dff)<max_baseline_dff) & 
                (np.asarray(event_length)<apnum/min_ap_frequency)&
                (np.asarray(event_sensor) == sensor))[0]
fig = plt.figure(figsize = [20,5])
ax_amplitudes = fig.add_subplot(131)
#
sns.lineplot(x=np.asarray(event_ap_num)[idxs]-1, y=np.asarray(event_amplitude_local)[idxs] ,color='r',ax=ax_amplitudes,ci="sd")
sns.swarmplot(x=np.asarray(event_ap_num)[idxs], y=np.asarray(event_amplitude_local)[idxs] ,color='k',size=1,ax=ax_amplitudes)
#ax_amplitudes.plot(apnums,amplitude_mean,'ro-',lw=3)
#ax_amplitudes.errorbar(apnums,amplitude_mean,amplitude_std,color='red',lw=5)
ax_amplitudes.set(ylim=(-.1, .8))
ax_amplitudes.set(xlim=(-1, 6))
ax_amplitudes.set_xlabel('AP number')
ax_amplitudes.set_ylabel('dF/F - local F0')
#ax_amplitudes.plot(np.asarray(event_ap_num)[idxs],np.asarray(event_amplitude_local)[idxs],'ko',ms=1)
# =============================================================================
# ax_amplitudes.set_ylim([-.1, 1])
# ax_amplitudes.set_xlim([-1, 9])
# =============================================================================

ax_amplitudes_new = fig.add_subplot(132)
sns.lineplot(x=np.asarray(event_ap_num)[idxs]-1, y=np.asarray(event_amplitude)[idxs] ,color='b',ax=ax_amplitudes_new,ci="sd")
sns.swarmplot(x=np.asarray(event_ap_num)[idxs], y=np.asarray(event_amplitude)[idxs] ,color='k',size=1,ax=ax_amplitudes_new)
ax_amplitudes_new.set(ylim=(-.1, .8))
ax_amplitudes_new.set(xlim=(-1, 6))
ax_amplitudes_new.set_xlabel('AP number')
ax_amplitudes_new.set_ylabel('dF/F - global F0')
# =============================================================================
# ax_amplitudes_new.errorbar(apnums,amplitude_mean_new,amplitude_std_new,color='blue',lw=5)
# ax_amplitudes_new.plot(np.asarray(event_ap_num)[idxs],new_amplitudes[idxs],'ko',ms=1)
# ax_amplitudes_new.set_ylim([-.1, 1])
# ax_amplitudes_new.set_xlim([0, 9])
# ax_amplitudes_new.set_xlabel('AP number')
# ax_amplitudes_new.set_ylabel('dF/F')
# =============================================================================


ax_amplitudes_au = fig.add_subplot(133)
sns.lineplot(x=np.asarray(event_ap_num)[idxs]-1, y=np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs] ,color='k',ax=ax_amplitudes_au,ci="sd")
sns.swarmplot(x=np.asarray(event_ap_num)[idxs], y=np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs] ,color='k',size=1,ax=ax_amplitudes_au)
#ax_amplitudes_au.set(ylim=(-.1, .8))
ax_amplitudes_au.set(xlim=(-1, 6))
ax_amplitudes_au.set_xlabel('AP number')
ax_amplitudes_au.set_ylabel('dF - AU')

# =============================================================================
# ax_amplitudes_au.plot(np.asarray(event_ap_num)[idxs],np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs],'bx',ms=3)
# ax_amplitudes_au.errorbar(apnums,amplitude_mean_AU,amplitude_std_AU,color='blue',lw=3)
# =============================================================================

#ax_amplitudes_au.set_ylim([-.1, .5])
#%%
cmap = colormap.get_cmap('jet') 
cellids = np.unique(event_unique_cell_id)
r_neu_bins = np.arange(0,2,.1)
Y=list()
sensor = 'G7F'
for i,cellid in enumerate(cellids):
    print(cellid)
    idxs = np.where((np.asarray(event_r_neu_cell_id) == cellid) & (np.asarray(event_r_neu_corrcoef)>.4) & (np.asarray(event_sensor) == sensor))[0] 
    if len(idxs)>0:
       y,x = np.histogram(np.asarray(event_r_neu)[idxs],bins = 10, range=[0,2])
       if len(Y)>0:
           plt.bar(np.mean([x[:-1],x[1:]],0),y,bottom=sum(Y), width = .2, facecolor=cmap(i/len(cellids)))
       else:
           plt.bar(np.mean([x[:-1],x[1:]],0),y, width = .2)
       
       
       Y.append(y)
plt.xlabel('r_neuropil')
plt.ylabel('count')
    #%%

from mpl_toolkits.mplot3d import Axes3D

idxs = np.where((event_ap_num==np.unique(event_ap_num)[1]) &
                (np.asarray(event_ap_amplitude_accomodation) > min_accomodation) &#(np.asarray(event_unique_cell_id) == cell_id)&
                (np.asarray(event_baseline_dff)<max_baseline_dff) & 
                (np.asarray(event_length)<apnum/min_ap_frequency)&
                (np.asarray(event_sensor) == sensor))[0]


fig = plt.figure(figsize = [10,10])
ax_ap = fig.add_subplot(111,projection='3d')
ax_ap.scatter(np.asarray(event_ap_num)[idxs],np.asarray(F0_local_all)[idxs],np.asarray(event_amplitude_local)[idxs]*np.asarray(F0_local_all)[idxs],color ='red')
ax_ap.set_xlabel('ap num')
ax_ap.set_zlabel('dF')
ax_ap.set_ylabel('F0')
plt.show()