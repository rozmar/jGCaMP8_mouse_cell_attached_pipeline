import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time as timer
import pandas as pd
import numpy as np
import scipy
import datajoint as dj
import os
import re
from pywavesurfer import ws
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_cell_attached,ephysanal_cell_attached #, ephysanal, imaging
import time
import ray

#%%
@ray.remote
def populatemytables_core_paralel(arguments,runround):
    if runround == 1:
        ephysanal_cell_attached.SweepFrameTimes().populate(**arguments)
        ephysanal_cell_attached.SquarePulse().populate(**arguments)
    elif runround == 2:
        ephysanal_cell_attached.ActionPotential().populate(**arguments)
    elif runround ==3:
        ephysanal_cell_attached.CellSpikeParameters().populate(**arguments)
        
# =============================================================================
#     elif runround == 2:
#         ephysanal.SquarePulseSeriesResistance().populate(**arguments)
#         ephysanal.SweepSeriesResistance().populate(**arguments)
#     elif runround == 3:
#         ephysanal.SweepResponseCorrected().populate(**arguments)
#     elif runround == 4:
#         ephysanal.ActionPotential().populate(**arguments)
#         ephysanal.ActionPotentialDetails().populate(**arguments)
# =============================================================================
        

def populatemytables_core(arguments,runround):
    if runround == 1:
        ephysanal_cell_attached.SweepFrameTimes().populate(**arguments)
        ephysanal_cell_attached.SquarePulse().populate(**arguments)
    elif runround == 2:
        ephysanal_cell_attached.ActionPotential().populate(**arguments)
    elif runround ==3:
        ephysanal_cell_attached.CellSpikeParameters().populate(**arguments)
# =============================================================================
#     elif runround == 2:
#         ephysanal.SquarePulseSeriesResistance().populate(**arguments)
#         ephysanal.SweepSeriesResistance().populate(**arguments)
#         extrapolate_sweep_series_resistance()
#     elif runround == 3:
#         ephysanal.SweepResponseCorrected().populate(**arguments)
#     elif runround == 4:
#         ephysanal.ActionPotential().populate(**arguments)
#         ephysanal.ActionPotentialDetails().populate(**arguments)
# =============================================================================
        
def populatemytables(paralel=True,cores = 9):
    if paralel:
        ephysanal_cell_attached.schema.jobs.delete()
        ray.init(num_cpus = cores)
        for runround in [1,2,3,4]:
            arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
            print('round '+str(runround)+' of populate')
            result_ids = []
            for coreidx in range(cores):
                result_ids.append(populatemytables_core_paralel.remote(arguments,runround))        
            ray.get(result_ids)
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)
            
        ray.shutdown()
    else:
        for runround in [1,2,3,4]:
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)

    
    #%%
    
def load_wavesurfer_file(WS_path): # works only for single sweep files
    #%%
    ws_data = ws.loadDataFile(filename=WS_path, format_string='double' )
    units = np.array(ws_data['header']['AIChannelUnits']).astype(str)
    channelnames = np.array(ws_data['header']['AIChannelNames']).astype(str)
    commandchannel = np.asarray(ws_data['header']['AOChannelNames'],str) =='Command'
    commandunit = np.asarray(ws_data['header']['AOChannelUnits'],str)[np.where(commandchannel)[0][0]]
    if 'Primary' in channelnames:
        ephysidx = (channelnames == 'Primary').argmax()
    elif 'Voltage' in channelnames:
        ephysidx = (channelnames == 'Voltage').argmax()
    else:
        print('cannot find primary ephys trace - FIX ME: {}'.format(WS_path))
   
    if 'Command' in channelnames:
        stimidx = (channelnames == 'Command').argmax()
    elif 'Secondary' in channelnames:
        stimidx = (channelnames == 'Secondary').argmax()
    else:
        print('no stimulus found')
        stimidx = np.nan
    
        
    if 'A' in commandunit:
        recording_mode = 'CC'
    else:
        recording_mode = 'VC'
    
    keys = list(ws_data.keys())
    del keys[keys=='header']
    if len(keys)>1:
        print('MULTIPLE SWEEPS! HANDLE ME!! : {}'.format(WS_path))
        timer.sleep(10000)
    sweep = ws_data[keys[0]]
    sRate = ws_data['header']['AcquisitionSampleRate'][0][0]
    voltage = sweep['analogScans'][ephysidx,:]
    if not np.isnan(stimidx):
        stimulus = sweep['analogScans'][stimidx,:]
    else:
        stimulus = None
    if units[ephysidx] == 'mV':
        voltage =voltage/1000
        units[ephysidx] = 'V'
    if units[ephysidx] == 'pA':
        voltage =voltage/10**12
        units[ephysidx] = 'A'
    if units[ephysidx] == 'nA':
        voltage =voltage/10**9
        units[ephysidx] = 'A' 
        
    if not np.isnan(stimidx) and units[stimidx] == 'V' and recording_mode == 'CC': # HOTFIX wavesurfer doesn't save stimulus in current clamp mode..
        units[stimidx] = 'A'
        stimulus= stimulus/10**9
    elif not np.isnan(stimidx) and units[stimidx] == 'mV' and recording_mode == 'CC': # HOTFIX wavesurfer doesn't save stimulus in current clamp mode..
        units[stimidx] = 'A'
        stimulus= stimulus/10**12
    elif not np.isnan(stimidx) and units[stimidx] == 'pA':
        stimulus =stimulus/10**12
        units[stimidx] = 'A'
    elif not np.isnan(stimidx) and units[stimidx] == 'nA':
        stimulus =stimulus/10**9
        units[stimidx] = 'A' 

    frametimesidx = (channelnames == 'FrameTrigger').argmax()
    frame_trigger = sweep['analogScans'][frametimesidx,:]
    if 'VisualStimulus' in channelnames:
        visualstimidx = (channelnames == 'VisualStimulus').argmax()
        visual_stim_trace = sweep['analogScans'][visualstimidx,:]
    else:
        visual_stim_trace = None
    
    timestamp = ws_data['header']['ClockAtRunStart']
    if not np.isnan(stimidx):
        stimunit =units[stimidx]
    else:
        stimunit= None
    outdict = {'response_trace':voltage,
               'stimulus_trace':stimulus,
               'sampling_rate':sRate,
               'recording_mode':recording_mode,
               'timestamp':timestamp,
               'response_trace_unit':units[ephysidx],
               'stimulus_trace_unit':stimunit,
               'visual_stim_trace':visual_stim_trace,
               'frame_trigger':frame_trigger}
    
    #%%
    return outdict#voltage, frame_trigger, sRate, recording_mode, timestamp  , units[ephysidx]
#%%
    
def populateelphys_wavesurfer(cores = 4):

    ray.init(num_cpus = cores)
    result_ids = []
    subject_ids=np.unique(lab.Subject().fetch('subject_id'))
    for subject_id in subject_ids:
        result_ids.append(populateelphys_wavesurfer_core.remote(subject_id))            
    ray.get(result_ids)       
    ray.shutdown()

@ray.remote
def populateelphys_wavesurfer_core(subject_id):
  #%
    df_subject_wr_sessions=pd.DataFrame(lab.WaterRestriction() * experiment.Session()&'subject_id = {}'.format(subject_id))# * experiment.SessionDetails
    df_subject_ids = pd.DataFrame(lab.Subject()&'subject_id = {}'.format(subject_id))
    if len(df_subject_wr_sessions)>0:
        subject_names = df_subject_wr_sessions['water_restriction_number'].unique()
        subject_names.sort()
    else:
        subject_names = list()
    subject_ids = df_subject_ids['subject_id'].unique()
    basedir = Path(dj.config['locations.elphysdata_wavesurfer'])
    for setup_dir in basedir.iterdir():
        setup_name=setup_dir.name
        for session_dir in setup_dir.iterdir():
            if not session_dir.is_dir(): # break out from session loop
                continue
            try:
                session_date = datetime.datetime.strptime(session_dir.name[:8], '%Y%m%d').strftime('%Y-%m-%d')
            except:
                print('couldn''t assign date to {}'.format(session_dir.name))
                continue
            wrname_ephys = session_dir.name[-6:]
            wrname = None
            for wrname_potential in subject_ids: # look for subject ID number
                if str(wrname_potential) in wrname_ephys.lower():
                    subject_id = wrname_potential
                    if len(df_subject_wr_sessions) > 0 and len((df_subject_wr_sessions.loc[df_subject_wr_sessions['subject_id'] == subject_id, 'water_restriction_number']).unique())>0:
                        wrname = (df_subject_wr_sessions.loc[df_subject_wr_sessions['subject_id'] == subject_id, 'water_restriction_number']).unique()[0]
                    else:
                        wrname = 'no water restriction number for this mouse'
            if not wrname: # break out from session loop
               # print('no matching subject on DJ for {}'.format(session_dir.name))
                continue
            
            print('animal: '+ str(subject_id)+'  -  '+wrname)##
            if setup_name == 'Voltage_rig_1P':
                setupname = 'Voltage-Imaging-1p'
            elif setup_name == 'Genie_2P_rig':
                setupname = 'Genie-2p-Resonant-MIMMS'
            else:
                print('unkwnown setup, please add')
                timer.wait(1000)
            
            username = 'rozsam' ## not saved in wavesurfer file..
            print('username not specified in acq4 file, assuming rozsam')
            ### check if session already exists
            sessiondata = {
                            'subject_id': subject_id,#(lab.WaterRestriction() & 'water_restriction_number = "'+df_behavior_session['subject'][0]+'"').fetch()[0]['subject_id'],
                            'session' : np.nan,
                            'session_date' : session_date,
                            'session_time' : np.nan,#session_time.strftime('%H:%M:%S'),
                            'username' : username,
                            'rig': setupname
                            }   
            
            #%
            cells = os.listdir(session_dir)
            cell_nums = list()
            cell_is_dir = list()
            for cell in cells:
# =============================================================================
#                 try:
#                     cell_nums.append(int(re.findall(r'cell.\d+', cell)[0][5:]))
#                 except:
# =============================================================================
                try:
                    cell_nums.append(int(re.findall(r'cell\d+', cell.lower())[0][4:]))
                except:
                    cell_nums.append(np.nan)
                cell_is_dir.append((session_dir/cell).is_dir())
            cells = np.array(cells)[~np.array(np.isnan(cell_nums)) & cell_is_dir]
            cell_nums = np.array(cell_nums)[~np.array(np.isnan(cell_nums))& cell_is_dir]
            #%
            if len(cells)==0:
                continue
            cell_order = np.argsort(cell_nums)
            cell_nums = cell_nums[cell_order]
            cells = cells[cell_order]
           # print('cells: {}, cellnums: {}'.format(cells,cell_nums))
            df_session_metadata = pd.read_csv(dj.config['locations.metadata_surgery_experiment']+'{}-anm{}.csv'.format(session_dir.name[:8],subject_id))
            cell_data_list = list()
            cell_start_times_list = list()
            for cell,cell_number in zip(cells,cell_nums):
                dj.conn().ping()
                try:
                    session = (experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"').fetch('session')[0]
                except: # session nonexistent
                    session = len((experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"').fetch('session')) +1
                #add cell if not added already
                #%
                celldata= {
                        'subject_id': subject_id,
                        'session': session,
                        'cell_number': int(cell_number),
                        }
                if len(ephys_cell_attached.Cell() & celldata) > 0:
                    print('cell already uploaded: {}'.format(celldata))
                    continue

                ephisdata_cell = list()
                sweepstarttimes = list()
                cell_dir =  session_dir.joinpath(cell)
                ephysfiles = os.listdir(cell_dir)
                ephysfiles_real = list()
                for ephysfile in ephysfiles:
                    if '.h5' in ephysfile:
                        ephysfiles_real.append(ephysfile)
                for ephysfile in ephysfiles_real:
                    try:
                        ephysdata = load_wavesurfer_file(cell_dir.joinpath(ephysfile))#voltage, frame_trigger, sRate, recording_mode, timestamp, response_unit
                    except OSError:
                        print('file not readable, skipping: {}'.format(cell_dir.joinpath(ephysfile)))
                        continue
                    timestamp = ephysdata['timestamp']
                    metadata = {'sRate':ephysdata['sampling_rate'],
                                'recording_mode':ephysdata['recording_mode']}
                    #ephisdata = dict()
                    ephysdata['time']=np.arange(len(ephysdata['response_trace']))/ephysdata['sampling_rate']
                    ephysdata['metadata']=metadata
                    ephysdata['sweepstarttime']= datetime.datetime(timestamp[0],timestamp[1],timestamp[2],timestamp[3],timestamp[4],int(timestamp[5]),(timestamp[5]-int(timestamp[5]))*10**6)
                    ephysdata['filename']= ephysfile
                    sweepstarttimes.append(ephysdata['sweepstarttime'])
                    ephisdata_cell.append(ephysdata)
                    
                sweep_order = np.argsort(sweepstarttimes)
                sweepstarttimes = np.asarray(sweepstarttimes)[sweep_order]
                ephisdata_cell = np.asarray(ephisdata_cell)[sweep_order]
                

                if len(ephisdata_cell) == 0 :
                    continue
                 # add session to DJ if not present

# =============================================================================
#                 if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"') == 0:
#                     if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"') == 0:
#                         sessiondata['session'] = 1
#                     else:
#                         sessiondata['session'] = len((experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"').fetch()['session']) + 1
#                     sessiondata['session_time'] = (sweepstarttimes[0]).strftime('%H:%M:%S') # the time of the first sweep will be the session time
#                     
#                     experiment.Session().insert1(sessiondata)
#                 session = (experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"').fetch('session')[0]
#                 #add cell if not added already
#                 #%
#                 celldata= {
#                         'subject_id': subject_id,
#                         'session': session,
#                         'cell_number': int(cell_number),
#                         }
#                 if len(ephys_cell_attached.Cell() & celldata) == 0:
# =============================================================================
                print('adding new recording:' )
                print(celldata)
                try:
                    celltype = np.array(df_session_metadata.loc[df_session_metadata['Cell']==cell_number,'cell type'].values,str)[0]
                except:
                    celltype = 'unknown'
                        
                print(celltype)
                if 'pyr' in celltype.lower():
                    celldata['cell_type'] ='pyr'
                elif 'int' in celltype.lower() or 'fs' in celltype.lower():
                    celldata['cell_type'] ='int'
                else:
                    celldata['cell_type'] = 'unidentified' # no chance to say
                #%
                celldata['cell_recording_start'] = (sweepstarttimes[0]).strftime('%H:%M:%S')
                #%
                try:
                    celldata['depth'] = np.nanmean(np.array(df_session_metadata.loc[df_session_metadata['Cell']==cell_number,'depth'].values,float))
                except:
                    celldata['depth']= None
                    
                if celldata['depth'] == None or np.isnan(celldata['depth']):
                    celldata['depth']= None
                #ephys_cell_attached.Cell().insert1(celldata, allow_direct_insert=True)
                #%
                if 'cell notes' in df_session_metadata.keys():
                    cellnotes = ' - '.join(np.asarray(df_session_metadata.loc[df_session_metadata['Cell']==cell_number,'cell notes'].values,str))
                else:
                    cellnotes = ''
                cellnotesdata = {
                        'subject_id': subject_id,
                        'session': session,
                        'cell_number': cell_number,
                        'notes': cellnotes
                        }                            
                #ephys_cell_attached.CellNotes().insert1(cellnotesdata, allow_direct_insert=True)

                #%
                Sweep_list = list()
                SweepMetadata_list = list()
                SweepResponse_list = list()
                SweepImagingExposure_list = list()
                SweepStimulus_list = list()
                SweepVisualStimulus_list = list()
                for i, ephisdata in enumerate(ephisdata_cell):
                    
                        #%
                    sweep_number = i
                    sweep_data = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'sweep_start_time':(ephisdata['sweepstarttime']-sweepstarttimes[0]).total_seconds(),
                            'sweep_end_time': (ephisdata['sweepstarttime']-sweepstarttimes[0]).total_seconds()+ephisdata['time'][-1],
                            'protocol_name': ephisdata['filename'][:-3],
                            'protocol_sweep_number': 1
                            }
                    #%
                    if ephisdata['metadata']['recording_mode'] == 'CC':
                        recording_mode = 'current clamp'
                    else:
                        recording_mode = 'voltage clamp'

                    sweepmetadata_data = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'recording_mode': recording_mode,
                            'sample_rate': ephisdata['metadata']['sRate']
                            }
                    sweepdata_data = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'response_trace': ephisdata['response_trace'],
                            'response_units': ephisdata['response_trace_unit']
                            }
                    Sweep_list.append(sweep_data)
                    SweepMetadata_list.append(sweepmetadata_data)
                    SweepResponse_list.append(sweepdata_data)
# =============================================================================
#                         ephys_cell_attached.Sweep().insert1(sweep_data, allow_direct_insert=True)
#                         ephys_cell_attached.SweepMetadata().insert1(sweepmetadata_data, allow_direct_insert=True)
#                         ephys_cell_attached.SweepResponse().insert1(sweepdata_data, allow_direct_insert=True)
# =============================================================================

                    #%
                    if 'frame_trigger' in ephisdata.keys():
                        sweepimagingexposuredata = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'imaging_exposure_trace' :  ephisdata['frame_trigger']
                            }
                        SweepImagingExposure_list.append(sweepimagingexposuredata)
                        #ephys_cell_attached.SweepImagingExposure().insert1(sweepimagingexposuredata, allow_direct_insert=True)
                    if 'stimulus_trace' in ephisdata.keys() and not type(ephisdata['stimulus_trace']) == type(None):
                        sweepstimulusdata = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'stimulus_trace' :  ephisdata['stimulus_trace'],
                            'stimulus_units' :  ephisdata['stimulus_trace_unit']
                            }
                        SweepStimulus_list.append(sweepstimulusdata)
                        #ephys_cell_attached.SweepStimulus().insert1(sweepstimulusdata, allow_direct_insert=True)
                        
                    if 'visual_stim_trace' in ephisdata.keys() and not type(ephisdata['visual_stim_trace']) == type(None):
                        visualstimulusdata = {
                            'subject_id': subject_id,
                            'session': session,
                            'cell_number': cell_number,
                            'sweep_number': sweep_number,
                            'visual_stimulus_trace' :  ephisdata['visual_stim_trace']
                            }
                        SweepVisualStimulus_list.append(visualstimulusdata)
                        #ephys_cell_attached.SweepVisualStimulus().insert1(visualstimulusdata, allow_direct_insert=True)
                cell_data = {'celldata':celldata,
                             'cellnotesdata':cellnotesdata,
                             'Sweep_list':Sweep_list,
                             'SweepMetadata_list':SweepMetadata_list,
                             'SweepResponse_list':SweepResponse_list,
                             'SweepImagingExposure_list':SweepImagingExposure_list,
                             'SweepStimulus_list':SweepStimulus_list,
                             'SweepVisualStimulus_list':SweepVisualStimulus_list}
                cell_data_list.append(cell_data)
                cell_start_times_list.append(celldata['cell_recording_start'])
            
            #%
            if len(cell_data_list) == 0:#nothing to insert
                continue
            if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"' & 'session_date = "'+str(sessiondata['session_date'])+'"') == 0:
                if len(experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"') == 0:
                    sessiondata['session'] = 1
                else:
                    sessiondata['session'] = len((experiment.Session() & 'subject_id = "'+str(sessiondata['subject_id'])+'"').fetch()['session']) + 1
                sessiondata['session_time'] = np.sort(cell_start_times_list)[0]#(sweepstarttimes[0]).strftime('%H:%M:%S') # the time of the first sweep will be the session time
                experiment.Session().insert1(sessiondata)
            for cell_data in cell_data_list:
                dj.conn().ping()
                with dj.conn().transaction: #inserting one cell
                    print('uploading to datajoint: {}'.format(cell_data['celldata']))
                    ephys_cell_attached.Cell().insert1(cell_data['celldata'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.CellNotes().insert1(cell_data['cellnotesdata'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.Sweep().insert(cell_data['Sweep_list'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.SweepMetadata().insert(cell_data['SweepMetadata_list'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.SweepResponse().insert(cell_data['SweepResponse_list'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.SweepImagingExposure().insert(cell_data['SweepImagingExposure_list'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.SweepStimulus().insert(cell_data['SweepStimulus_list'], allow_direct_insert=True)
                    dj.conn().ping()
                    ephys_cell_attached.SweepVisualStimulus().insert(cell_data['SweepVisualStimulus_list'], allow_direct_insert=True)
                    dj.conn().ping()

#%%
def ingest_AP_groups():
    for cell in ephys_cell_attached.Cell():
        key = {'subject_id': cell['subject_id'],
               'session' : cell['session'],
               'cell_number' : cell['cell_number']}
        if len(ephysanal_cell_attached.APGroup()&key)==0:
            ingest_AP_groups_core(key)
            
                    
def ingest_AP_groups_core(key):
    min_baseline_time = .1
    max_integration_time = .1
    
    sweep_numbers = (ephys_cell_attached.Sweep()&key).fetch('sweep_number')
    apgroup_list = list()
    groupmember_list = list()
    ap_group_number = 0
    for sweep_number in sweep_numbers:
        dj.conn().ping()
        key['sweep_number'] = sweep_number
        sweep_start_time,sweep_end_time = (ephys_cell_attached.Sweep()&key).fetch1('sweep_start_time','sweep_end_time')
        sweep_end_time = float(sweep_end_time-sweep_start_time)
        sweep_start_time = float(sweep_start_time)
        ap_num,ap_max_time,ap_snr_v,ap_snr_dv = (ephys_cell_attached.Sweep()*ephysanal_cell_attached.ActionPotential()&key).fetch('ap_num','ap_max_time','ap_snr_v','ap_snr_dv')
        ap_max_time = np.asarray(ap_max_time,float)
        
        isis_pre = np.diff(np.concatenate([[0],ap_max_time]))
        isis_post = np.diff(np.concatenate([ap_max_time,[sweep_end_time]]))
        
        group_start_idxes = np.where(isis_pre>min_baseline_time)[0]
        group_end_idxes = np.where(isis_post>max_integration_time)[0]
        for group_start_idx in group_start_idxes:
            if any(group_end_idxes>=group_start_idx):
                apgroup_key = key.copy()
                ap_group_number += 1
                group_end_idx = group_end_idxes[np.argmax(group_end_idxes>=group_start_idx)]
                ap_indices_now = np.arange(group_start_idx,group_end_idx+1)
                
                apgroup_key['ap_group_number'] = ap_group_number
                apgroup_key['ap_group_ap_num'] = len(ap_indices_now)
                apgroup_key['ap_group_start_time'] = ap_max_time[group_start_idx] + sweep_start_time
                apgroup_key['ap_group_end_time'] = ap_max_time[group_end_idx] + sweep_start_time
                apgroup_key['ap_group_min_snr_v'] = np.min(ap_snr_v[ap_indices_now])
                apgroup_key['ap_group_min_snr_dv'] = np.min(ap_snr_dv[ap_indices_now])
                apgroup_key['ap_group_pre_isi'] = isis_pre[group_start_idx]
                apgroup_key['ap_group_post_isi'] = isis_post[group_end_idx]            
                apgroup_list.append(apgroup_key)
                for ap_idx in ap_indices_now:
                    groupmember_key = key.copy()
                    groupmember_key['ap_group_number'] = ap_group_number
                    groupmember_key['ap_num'] = ap_num[ap_idx]
                    groupmember_list.append(groupmember_key)
                    
    with dj.conn().transaction: #inserting one cell
        print('uploading AP groups to datajoint: {}'.format(key))
        ephysanal_cell_attached.APGroup().insert(apgroup_list, allow_direct_insert=True)
        dj.conn().ping()
        ephysanal_cell_attached.APGroup.APGroupMember().insert(groupmember_list, allow_direct_insert=True)            
#%%


