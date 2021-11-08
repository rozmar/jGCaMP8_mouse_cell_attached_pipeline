#%%
import pynwb
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import numpy as np
import scipy.ndimage as ndimage
%matplotlib qt
sensor_colors ={'jGCaMP7f':'green',
                'XCaMPgf':'#0099ff',
                'jGCaMP8f':'blue',
                'jGCaMP8m':'red',
                'jGCaMP8s':'black'}
sensor = 'jGCaMP8s'
subject_id = 472182
cell_number = 2
movie_number = 1

# =============================================================================
# sensor = 'jGCaMP8m'
# subject_id = 479118
# cell_number = 6
# movie_number = 2
# =============================================================================

nwbfile = '/home/rozmar/Mount/HDD_RAID_2_16TB/GCaMP8_NWB/{}_ANM{}_cell{}.nwb'.format(sensor,subject_id,str(cell_number).zfill(2))
gauss_filter_sigma = 10 #ms

def gaussFilter(sig,sRate,sigma = .01):
    si = 1/sRate
    #sigma = .00005
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f



io = pynwb.NWBHDF5IO(nwbfile, 'r')
nwbfile = io.read()

print(nwbfile.acquisition.keys()) # data in this file
print(nwbfile.processing.keys()) # fluorescence traces in this file

#%
imaging_plane = nwbfile.imaging_planes['Movie_{}'.format(movie_number)]
ephys = nwbfile.acquisition['loose seal recording for movie {}'.format(movie_number)]
ophys = nwbfile.processing['ophys of movie {}'.format(movie_number)]
meanimage = nwbfile.acquisition['mean images of movie {}'.format(movie_number)]
mean_image_GCaMP = meanimage.images['{} at 940nm'.format(sensor)].data[:]
movie_pixel_size = imaging_plane.grid_spacing[0]

ephys_trace = ephys.data[:]
if ephys.unit == 'volts':
    ephys_trace = ephys_trace*1000
    ephys_unit = 'mV'
else:
    ephys_trace = ephys_trace*1e12
    ephys_unit = 'pA'
ephys_time = np.arange(len(ephys_trace))/ephys.rate+ephys.starting_time
ap_times = nwbfile.analysis['responses']['ap_time'].data[:]
ap_times = ap_times[(ap_times>np.min(ephys_time)) &(ap_times<np.max(ephys_time))]
ap_indices = []
for ap_now in nwbfile.analysis['responses']['response'].data[:]:
    if ap_now[2].sweep_number==movie_number:
        ap_indices.append(ap_now[0])
ap_indices = np.asarray(ap_indices)    

roi_masks = ophys['ImageSegmentation']['MovieSegmentation']['image_mask'].data[:]
neuropil_masks = ophys['ImageSegmentation']['MovieSegmentation']['neuropil_mask'].data[:]
roi_traces = ophys['Fluorescence']['RoiResponseSeries'].data[:]
neuropil_traces = ophys['Fluorescence']['NeuropilResponseSeries'].data[:]
roi_time = np.arange(len(roi_traces[0]))/ophys['Fluorescence']['RoiResponseSeries'].rate+ophys['Fluorescence']['RoiResponseSeries'].starting_time

roi_image = np.zeros_like(mean_image_GCaMP)
roi_center_indices = []
for roi_i,(roi_now,neuropil_now) in enumerate(zip(roi_masks,neuropil_masks)):
    roi_image+=roi_now/np.max(roi_now)
    roi_image[neuropil_now>0] = -1*neuropil_now[neuropil_now>0]/np.max(neuropil_now)
    y_ind,x_ind = np.where(roi_now)
    
    roi_center_indices.append([np.median(x_ind),np.median(y_ind)])

fig = plt.figure(figsize = [20,10])
ax_ophys_all = fig.add_subplot(3,2,6)
ax_roi_shape = fig.add_subplot(3,2,4)
ax_meanimage = fig.add_subplot(3,2,2)
ax_ophys_recorded = fig.add_subplot(3,2,1,sharex = ax_ophys_all)
ax_ephys = fig.add_subplot(3,2,3,sharex = ax_ophys_all)
ax_visual_stim = fig.add_subplot(3,2,5)
ax_visual_stim_dff = ax_visual_stim.twinx()
ax_ephys.plot(ephys_time,ephys_trace,'k-')
ax_ephys.plot(ephys_time[ap_indices],ephys_trace[ap_indices],'ro')
ax_ophys_all.axis('off')

ax_meanimage.set_title('Mean image of functional channel')
ax_ophys_all.set_title('dF/F of all somatic ROIs')
ax_roi_shape.set_title('somatic and neuropil pixels')
ax_ophys_recorded.set_title('{} - ANM{} Cell {} Movie {}'.format(sensor,472182,cell_number,movie_number))
ax_ophys_recorded.set_ylabel('dF/F')
ax_ophys_recorded.set_xlabel('time from cell recording start (s)')
ax_ophys_recorded.spines['top'].set_visible(False)
ax_ophys_recorded.spines['right'].set_visible(False)
ax_ephys.set_ylabel('{}'.format(ephys_unit))
ax_ephys.set_xlabel('time from cell recording start (s)')
ax_ephys.spines['top'].set_visible(False)
ax_ephys.spines['right'].set_visible(False)
ax_visual_stim.set_xlabel('Orientation of drifting gratings (degree)')
ax_visual_stim.set_ylabel('Tuning curve (spikes)')
ax_visual_stim_dff.set_ylabel('Tuning curve (dF/F)')
ax_visual_stim_dff.spines['right'].set_color('green')
ax_visual_stim_dff.tick_params(axis='y', colors='green')
ax_visual_stim_dff.yaxis.label.set_color('green')
ax_visual_stim.spines['top'].set_visible(False)
ax_visual_stim.spines['right'].set_visible(False)
ax_visual_stim_dff.spines['top'].set_visible(False)

ax_meanimage.imshow(mean_image_GCaMP)
im_green = ax_meanimage.imshow(mean_image_GCaMP, extent=[0,mean_image_GCaMP.shape[1]*float(movie_pixel_size),mean_image_GCaMP.shape[0]*float(movie_pixel_size),0], aspect='auto')
clim = np.percentile(mean_image_GCaMP.flatten(),[.1,99.9])
im_green.set_clim(clim)
im_rois = ax_roi_shape.imshow(roi_image, extent=[0,roi_image.shape[1]*float(movie_pixel_size),roi_image.shape[0]*float(movie_pixel_size),0], aspect='auto')
for roi_i,center_idx in enumerate(roi_center_indices):
    ax_roi_shape.text(np.median(center_idx[0])*movie_pixel_size,np.median(center_idx[1])*movie_pixel_size,'{}'.format(roi_i))
offset = 0
for roi_i,(roi_trace,neuropil_trace) in enumerate(zip(roi_traces,neuropil_traces)):
    f_corrected = roi_trace-0.8*neuropil_trace
    f0 = np.percentile(f_corrected,20)
    if f0<0:
        continue
    dff = (f_corrected-f0)/f0
    dff = gaussFilter(dff,ophys['Fluorescence']['RoiResponseSeries'].rate,sigma = gauss_filter_sigma/1000)
    
    offset -= np.percentile(dff,100)
    ax_ophys_all.plot(roi_time,dff+offset,'g-',alpha=1)
    ax_ophys_all.text(roi_time[0]-.1*(np.max(roi_time)-np.min(roi_time)),offset,'roi {}'.format(roi_i))
    if roi_i ==0:
        ax_ophys_recorded.plot(roi_time,dff,'g-',alpha=1,label = 'dF/F of loose-seal recorded cell (roi 0)')
        ax_ophys_recorded.plot(ap_times,np.zeros(len(ap_times))-.1*(np.max(dff)-np.min(dff)),'r|',label = 'Action Potential')
        f_corrected_of_recorded_cell = f_corrected
        ax_ophys_recorded.legend()
#%
cmap = colormap.get_cmap('jet')
ylimits_ephys = ax_ephys.get_ylim()
angle_list = []
apnum_list = []
dff_list = []
for stim_start_t_now,stim_end_t_now,a in zip(nwbfile.trials['start_time'].data[:],nwbfile.trials['stop_time'].data[:],nwbfile.trials['angle'].data[:]):
    if stim_start_t_now>roi_time[0] and stim_start_t_now<roi_time[-1]:
        ax_ephys.fill_between(ephys_time,
                                    ylimits_ephys[0],
                                    ylimits_ephys[1]   , 
                                    where= (ephys_time > stim_start_t_now) & (ephys_time < stim_end_t_now), 
                                    facecolor=cmap(a/360), 
                                    alpha=0.5)
        angle_list.append(a)
        visual_stim_ap_num = np.sum((ap_times>stim_start_t_now)&(ap_times<stim_end_t_now))
        baseline_ap_num = np.sum((ap_times<stim_start_t_now)&(ap_times>stim_start_t_now-(stim_end_t_now-stim_start_t_now)))
        apnum_list.append(visual_stim_ap_num-baseline_ap_num)
        
        stim_start_imaging = np.argmax(roi_time>stim_start_t_now)
        if np.argmax(roi_time>stim_end_t_now)==0:  
            stim_end_imaging = len(roi_time) 
        else:  
            stim_end_imaging = np.argmax(roi_time>stim_end_t_now)
        recorded_roi_f = f_corrected_of_recorded_cell[stim_start_imaging:stim_end_imaging]
        recorded_roi_f0 = np.mean(f_corrected_of_recorded_cell[np.argmax(roi_time>stim_start_t_now-.7):np.argmax(roi_time>stim_start_t_now)])
        recorded_roi_dff = (recorded_roi_f-recorded_roi_f0)/recorded_roi_f0
        dff_list.append(np.mean(recorded_roi_dff[recorded_roi_dff>=np.percentile(recorded_roi_dff,75)]))
unique_angles = np.unique(angle_list)

dff_mean_list = []
dff_std_list = []

ap_mean_list = []
ap_std_list = []
for a in unique_angles:
    apnum_mean = np.mean(np.asarray(apnum_list)[np.asarray(angle_list)==a])
    apnum_std = np.std(np.asarray(apnum_list)[np.asarray(angle_list)==a])
    
    ap_mean_list.append(apnum_mean)
    ap_std_list.append(apnum_std)
# =============================================================================
#     ax_visual_stim.bar(a,apnum_mean,width = width,color = cmap(a/360))
#     ax_visual_stim.errorbar(a,apnum_mean,apnum_std,color = cmap(a/360))
# =============================================================================
    
    dff_mean_list.append(np.mean(np.asarray(dff_list)[np.asarray(angle_list)==a]))
    dff_std_list.append(np.std(np.asarray(dff_list)[np.asarray(angle_list)==a]))
    
ax_visual_stim.plot(unique_angles,ap_mean_list,'k-',linewidth = 3,alpha = .7)
ax_visual_stim.errorbar(unique_angles,ap_mean_list,ap_std_list,color = 'black',alpha = .7)


ax_visual_stim_dff.plot(unique_angles,dff_mean_list,'g-',linewidth = 3,alpha = .7)
ax_visual_stim_dff.errorbar(unique_angles,dff_mean_list,dff_std_list,color = 'green',alpha = .7)

y_limits_visual_stim = ax_visual_stim.get_ylim()
width = 10#np.mean(np.diff(unique_angles))*.8
for a in range(0,370,10):#unique_angles:
    ax_visual_stim.bar(a,y_limits_visual_stim,width = width,color = cmap(a/360),alpha = .5)
    
ax_visual_stim.set_xticks(unique_angles)

io.close()