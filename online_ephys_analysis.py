from ephys_utils import extract_tuning_curve
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
#%%

ephys_basedir = '/home/rozmar/Network/Genie2Prig/imaging_ephys/rozsam'
vis_stim_basedir = '/home/rozmar/Network/Genie2Prig/visual_stim/Visual stim'

ephys_basedir = '/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/rozsam'
vis_stim_basedir = '/home/rozmar/Data/Visual_stim/raw/Genie_2P_rig'
suite2p_basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig/rozsam'

session = '20200418'
subject = '471997'
cell = '5'
runnum = [3] # visual stimulus number needed
roi_num = [2] # suite2p roi number
plot_suite2p_output = True
F0win = 20 #s
F_filter_sigma = .01 #seconds
neu_r =.7
uniqueAngles = [ 45,  90, 135, 180, 225, 270, 315, 360]
stim_ap_dict =  {key: list() for key in [str(i) for i in uniqueAngles]} 
baseline_ap_dict = {key: list() for key in [str(i) for i in uniqueAngles]} 
#%
sessions = os.listdir(ephys_basedir)
sessiondir = os.path.join(ephys_basedir,'{}-anm{}'.format(session,subject))
cells = os.listdir(sessiondir)
for celldir in cells:
    if 'cell{}'.format(cell) in celldir.lower():
        break
celldir = os.path.join(sessiondir,celldir)
ephysfiles = os.listdir(celldir)
ephysfiles_real = list()
stimnums=list()
stimidxs = list()
for ephysfile in ephysfiles:
    if '.h5' in ephysfile:
        
        separators = [m.start() for m in re.finditer('_| |-', ephysfile)]
        if int(re.findall(r'\d+', ephysfile[separators[0]+1:separators[1]])[0]) in runnum:
            ephysfiles_real.append(ephysfile)
            stimnums.append(ephysfile[separators[0]+1:separators[1]])
            stimidxs.append(int(re.findall(r'\d+', ephysfile[separators[0]+1:separators[1]])[0]))
stimnums = np.asarray(stimnums)[np.argsort(stimidxs)]
vstimdirs=os.listdir(vis_stim_basedir)
data_dicts = list()
for stim,ephysfile in zip(stimnums,ephysfiles_real):  
    try:
        stimnum = re.findall(r'\d+', stim)[0]
        dirnow_needed = '{}-anm{}'.format(session,subject)
        if dirnow_needed in vstimdirs:
            sessiondir_vstim = os.path.join(vis_stim_basedir,dirnow_needed)
            stimdirs = os.listdir(sessiondir_vstim)
            stimdirs_cellnum=list()
            stimdirs_runnum=list()
            for stimdir_now in stimdirs:
                sepidx = stimdir_now.find('_')
                try:
                    stimdirs_cellnum.append(int(stimdir_now[4:sepidx]))
                except:
                    stimdirs_cellnum.append(np.nan)
                try:
                    stimdirs_runnum.append(int(stimdir_now[sepidx+4:]))
                except:
                    stimdirs_runnum.append(np.nan)
                
            stimdirnow_needed = 'cell{}_Run{}'.format(cell,stimnum)
            if any((np.asarray(stimdirs_cellnum)==float(cell)) & (np.asarray(stimdirs_runnum)==float(stimnum))):
                stimdirnow_needed = stimdirs[np.argmax((np.asarray(stimdirs_cellnum)==float(cell)) & (np.asarray(stimdirs_runnum)==float(stimnum)))]
                vstimdir_now = os.path.join(sessiondir_vstim,stimdirnow_needed)
                vstimfile= os.listdir(vstimdir_now)[0]
                try:
                    data_now = extract_tuning_curve(WS_path = os.path.join(celldir,ephysfile),vis_path = os.path.join(vstimdir_now,vstimfile),plot_data=True,plot_title = 'anm{}-cell{}-{}'.format(subject,cell,stim))
                    
                    if plot_suite2p_output:
                        #%%
                        suite2p_cell_path = suite2p_basedir +celldir[len(ephys_basedir):]
                        movie_dirs = os.listdir(suite2p_cell_path)
                        for movie_dir in movie_dirs:
                            separators = [m.start() for m in re.finditer('_| |-', movie_dir)]
                            if len(separators)==1: # HOTFIX first recordings the movie name doesn't include the cell identity
                                separators=[0]+separators
                            if len(re.findall(r'\d+', movie_dir[separators[0]+1:separators[1]]))>0 and int(re.findall(r'\d+', movie_dir[separators[0]+1:separators[1]])[0]) == int(stimnum):
                                suite2p_movie_path = os.path.join(suite2p_cell_path,movie_dir,'plane0')
                                F_all = np.load(os.path.join(suite2p_movie_path,'F.npy'))
                                Fneu_all = np.load(os.path.join(suite2p_movie_path,'Fneu.npy'))
                                framerate = int(1/np.mean(np.diff(data_now['frame_times'])))
                                ROI_idx = roi_num[runnum==int(stimnum)]
                                F = F_all[ROI_idx]
                                Fneu = Fneu_all[ROI_idx]
                                
                                
                                F0step= int(F0win/np.mean(np.diff(data_now['frame_times'])))
                                
                                
                                
                                F_orig = F.copy()
                                Fneu_orig = Fneu.copy()
                                F = utils.gaussFilter(F,framerate,sigma = F_filter_sigma)
                                Fneu = utils.gaussFilter(Fneu,framerate,sigma = F_filter_sigma)
                                F_corr = F-Fneu*neu_r
                                
                                F0 = utils.rollingfun(F_corr, window = F0step, func = 'min')
                                F0 = utils.rollingfun(F0, window = int(F0step*1), func = 'max')
                                dFF = (F-F0)/F0
                                #s2p_metadata = np.load(os.path.join(suite2p_movie_path,'ops.npy')).tolist()
                                #%
                                xlims= [np.min(data_now['ephys_t']),np.max(data_now['ephys_t'])]
                                #xlims=[104,105]
                                fig = plt.figure(figsize = [15,10])
                                ax_raw = fig.add_subplot(411)
                                ax_img = fig.add_subplot(412)
                                ax_dff = fig.add_subplot(413)
                                ax_ephys = fig.add_subplot(414)
                                ax_rate = ax_ephys.twinx()
                                ax_raw.plot(data_now['frame_times'],F_orig,'k-')
                                ax_raw.plot(data_now['frame_times'],Fneu_orig,'r-')
                                
                                ax_img.plot(data_now['frame_times'],F_corr,'g-')
                                ax_img.plot(data_now['frame_times'],F0,'k-')
                                #ax_img.plot(data_now['ephys_t'][data_now['ap_idx']],np.ones(len(data_now['ap_idx']))*np.min(F),'r|')
                                
                                ax_dff.plot(data_now['frame_times'],dFF,'g-',linewidth=1)
                                ax_dff.plot(data_now['ephys_t'][data_now['ap_idx']],np.ones(len(data_now['ap_idx']))*np.max(dFF)*-0.1,'r|')
                                ax_ephys.plot(data_now['ephys_t'],data_now['ephys_v_filt'],'k-')
                                
                                fr_e, fr_bincenters = utils.AP_times_to_rate(data_now['ephys_t'][data_now['ap_idx']],firing_rate_window=1)
                                ax_rate.plot(fr_bincenters,fr_e,'r-')
                                
                                
                                ax_raw.set_xlim(xlims)
                                ax_img.set_xlim(xlims)
                                ax_dff.set_xlim(xlims)
                                ax_ephys.set_xlim(xlims)
                                plt.show()
                                
                            
                        #%%
                    if type(data_now)==dict:
                        data_dicts.append(data_now)
                        for a in uniqueAngles:
                            idx = data_now['uniqueAngles'] == a
                            if any(idx):
                                stim_ap_dict[str(a)].extend(data_now['allSpikes'][idx][0])
                                baseline_ap_dict[str(a)].extend(data_now['allSpikes_baseline'][idx][0])
                except ValueError:
                    pass
            else:
                print(stimdirnow_needed+' not found')
    except:
        pass
#%
stim_ap_dict_orig = stim_ap_dict.copy()        
baseline_ap_dict_orig = baseline_ap_dict.copy()   
#%%
stim_ap_dict = stim_ap_dict_orig.copy()        
baseline_ap_dict = baseline_ap_dict_orig.copy()        
maxlength = 0
for keynow in stim_ap_dict.keys():
    maxlength = np.max([len(stim_ap_dict[keynow]),maxlength])
    #%
    
for keynow in stim_ap_dict.keys():
    if maxlength>len(stim_ap_dict[keynow]):
        stim_ap_dict[keynow] = stim_ap_dict[keynow].extend(np.zeros(maxlength - len(stim_ap_dict[keynow]))*np.nan)
        baseline_ap_dict[keynow] = baseline_ap_dict[keynow].extend(np.zeros(maxlength - len(baseline_ap_dict[keynow]))*np.nan)
        
df_stim = pd.DataFrame(stim_ap_dict)
df_baseline = pd.DataFrame(baseline_ap_dict)
df_apdiff = df_stim-df_baseline
#%
sns.swarmplot(data=df_apdiff,color='k')
#plt.errorbar(df_apdiff.keys(),df_apdiff.mean(skipna=True),df_apdiff.std(skipna=True),color='red',linewidth=4)
plt.show()
