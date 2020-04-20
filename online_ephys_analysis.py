from ephys_utils import extract_tuning_curve
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
#%%

ephys_basedir = '/home/rozmar/Network/Genie2Prig/imaging_ephys/rozsam'
vis_stim_basedir = '/home/rozmar/Network/Genie2Prig/visual_stim/Visual stim'
#vis_stim_basedir = '/home/rozmar/Data/Visual_stim/raw/Genie_2P_rig'
session = '20200418'
subject = '471997'
cell = '1'
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
for ephysfile in ephysfiles:
    if '.h5' in ephysfile:
        ephysfiles_real.append(ephysfile)
        separators = [m.start() for m in re.finditer('_', ephysfile)]
        stimnums.append(ephysfile[separators[0]+1:separators[1]])
        
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
                    data_now = extract_tuning_curve(WS_path = os.path.join(celldir,ephysfile),vis_path = os.path.join(vstimdir_now,vstimfile),plot_data=True)
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
plt.errorbar(df_apdiff.keys(),df_apdiff.mean(skipna=True),df_apdiff.std(skipna=True),color='red',linewidth=4)
plt.show()
