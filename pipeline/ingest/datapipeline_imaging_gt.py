import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
import datajoint as dj
import re 
import pandas as pd
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, imaging, ephys_cell_attached, ephysanal_cell_attached, imaging_gt
#import ray
import time
import ray
import scipy.ndimage as ndimage
import scipy.signal as signal
import os
import random
def gaussFilter(sig,sRate,sigma = .00005):
    si = 1/sRate
    sig_f = ndimage.gaussian_filter(sig,sigma/si)
    return sig_f
homefolder = dj.config['locations.mr_share']
#%%
@ray.remote
def populatemytables_core_paralel(arguments,runround):
    if runround == 1:        
        imaging.MoviePowerPostObjective().populate(**arguments)
        imaging_gt.SessionCalciumSensor().populate(**arguments)
        imaging_gt.ROIDwellTime().populate(**arguments)
        imaging_gt.ROINeuropilCorrelation().populate(**arguments)
        imaging_gt.MovieDepth().populate(**arguments)
        imaging_gt.CellDynamicRange().populate(**arguments)
        imaging_gt.CellMovieDynamicRange().populate(**arguments)
        
    elif runround == 2:
        imaging_gt.LFPNeuropilCorrelation().populate(**arguments)
        imaging_gt.SessionROIFDistribution().populate(**arguments)
        imaging_gt.IngestCalciumWave().populate(**arguments)
        imaging_gt.IngestDoubletCalciumWave().populate(**arguments)
        
        
    elif runround == 3:
        imaging_gt.MovieCalciumWaveSNR().populate(**arguments)
        imaging_gt.CellBaselineFluorescence().populate(**arguments)
        
        

def populatemytables_core(arguments,runround):
    if runround == 1:
        imaging.MoviePowerPostObjective().populate(**arguments)
        imaging_gt.SessionCalciumSensor().populate(**arguments)
        imaging_gt.ROIDwellTime().populate(**arguments)
        imaging_gt.ROINeuropilCorrelation().populate(**arguments)
        imaging_gt.MovieDepth().populate(**arguments)
        imaging_gt.CellDynamicRange().populate(**arguments)
        imaging_gt.CellMovieDynamicRange().populate(**arguments)
        
    elif runround == 2:
        imaging_gt.IngestCalciumWave().populate(**arguments)
        imaging_gt.IngestDoubletCalciumWave().populate(**arguments)
        imaging_gt.LFPNeuropilCorrelation().populate(**arguments)
        imaging_gt.SessionROIFDistribution().populate(**arguments)
    elif runround == 3:
        imaging_gt.MovieCalciumWaveSNR().populate(**arguments)
        imaging_gt.CellBaselineFluorescence().populate(**arguments)
        
def populatemytables(paralel=True,cores = 9):
    finished = False
    while not finished:
        try:
            if paralel:
                imaging_gt.schema.jobs.delete()
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
            finished = True
        except dj.errors.LostConnectionError:
            print('datajoint connection error, trying again')
            if paralel:
                ray.shutdown()

#%%
def upload_ground_truth_ROIs_and_notes():
    #%%
    sessions = experiment.Session()
    for session in sessions:
        if len(experiment.SessionComment()&session)==0:
            print('loading ROIs for {}'.format(session['subject_id']))
            subject_id = session['subject_id']
            session = session['session']
            upload_ground_truth_ROIs_and_notes_core(subject_id,session)

#%%
def upload_ground_truth_ROIs_and_notes_core(subject_id,session):  
    #%%
    roisweepcorrespondance_list = list()
    movienote_list = list()
    session_date,session_time = (experiment.Session()&'subject_id = {}'.format(subject_id)&'session = {}'.format(session)).fetch1('session_date','session_time')
    movies = imaging.Movie()&'subject_id = {}'.format(subject_id)&'session = {}'.format(session)
    movie_start_times = np.asarray(np.sort(movies.fetch('movie_start_time')),float)
    sweeps = ephys_cell_attached.Cell()*ephys_cell_attached.Sweep()&'subject_id = {}'.format(subject_id)&'session = {}'.format(session)
    df_session_metadata = pd.read_csv(dj.config['locations.metadata_surgery_experiment']+'{}-anm{}.csv'.format(str(session_date).replace('-',''),subject_id))
    session_comment = df_session_metadata['general comments'][0]
    sessioncomment = {'subject_id':subject_id,
                      'session':session,
                      'session_comment':session_comment}
    pipette_fill = df_session_metadata['pipette fill'][0]
    if '594' in pipette_fill:
        dye_name = 'Alexa 594'
    else:
        print('unknown dye: {}'.format(pipette_fill))
    if 'um' in pipette_fill.lower():
        dye_concentration = float(pipette_fill[:pipette_fill.lower().find('um')])
    elif 'ul' in pipette_fill.lower():#stupid typo
        dye_concentration = float(pipette_fill[:pipette_fill.lower().find('ul')])
    else:
        print('unknown dye concentration: {}'.format(dye_concentration))
    sessionpipettefill = {'subject_id':subject_id,
                          'session':session,
                          'dye_name':dye_name,
                          'dye_concentration':dye_concentration}
    for sweep in sweeps:
        cell_time = sweep['cell_recording_start']
    
        sweep_start_time = (cell_time - session_time).total_seconds() + float(sweep['sweep_start_time'])
        movie_start_time= movie_start_times[np.argmax(movie_start_times>sweep_start_time)]
        try:
            movie = list(movies & 'movie_start_time = {}'.format(movie_start_time))[0]
            timediff = np.abs(float(movie['movie_start_time'])-sweep_start_time)
            if timediff>.1:
                print('error in finding movie for sweep - timediff too big: {}'.format(sweep))
                #break
                continue
        except:
            print('error in finding movie for sweep: {}'.format(sweep))
            #break
            continue
        movie_match = re.search('(stim_(\d+)|stm(\d+)|sim(\d+)|sitm(\d+)|stim(\d+))', movie['movie_name'])
        if not movie_match:
            print('no reference to stim number in movie name: {}'.format(movie))
            continue
        movie_stimnum = int(re.findall("\d+",movie_match.string[movie_match.span()[0]:movie_match.span()[1]])[0])
        ephys_match = re.search('(stim_(\d+)|stm(\d+)|sim(\d+)|sitm(\d+)|stim(\d+))', sweep['protocol_name'])
        if not ephys_match:
            print('no reference to stim number in movie name: {}'.format(movie))
            continue
        ephys_stimnum = int(re.findall("\d+",ephys_match.string[ephys_match.span()[0]:ephys_match.span()[1]])[0])
        if ephys_stimnum != movie_stimnum:
            print('movie and ephys stimnum is not the same: {} - {}'.format(sweep,movie))
            continue
        row = df_session_metadata.loc[np.where(((df_session_metadata['Cell']==sweep['cell_number']) & (df_session_metadata['Run']==ephys_stimnum)).values)[0][0]]
        #%
        
        movienote = dict()
        for primary_key in imaging.Movie().primary_key:
            movienote[primary_key] = movie[primary_key]
        for key in row.keys():
            if ('expander' in key or 'good' in key or'comment' in key or'note' in key) and 'general' not in key and len(str(row[key]))>0:
                movienote_now = movienote.copy()
                movienote_now['movie_note'] = str(row[key])
                if 'phys' in key:
                    movienote_now['movie_note_type'] = 'ephys'
                elif 'anal' in key:
                    movienote_now['movie_note_type'] = 'analysis'
                elif 'expander' in key:
                    movienote_now['movie_note_type'] = 'beam expander'
                elif 'good' in key:
                    movienote_now['movie_note_type'] = 'quality'
                elif 'cell' in key:
                    movienote_now['movie_note_type'] = 'cell'
                else:
                    movienote_now['movie_note_type'] = 'imaging'
                movienote_list.append(movienote_now)
        try:
            roi_number = int(row['suite2p-ROI-num'])
        except:
            print('no valid ROI number: {}'.format(row['suite2p-ROI-num']))
            continue
        try:
            roi = list(imaging.ROI()&movie&'roi_number = {}'.format(roi_number))[0]
        except:
            print('ROI {} not found in this movie: {}'.format(roi_number,movie))
        roisweepcorrespondance = dict()
        for primary_key in imaging.ROI().primary_key:
            roisweepcorrespondance[primary_key] = roi[primary_key]
        for primary_key in ephys_cell_attached.Sweep().primary_key:
            roisweepcorrespondance[primary_key] = sweep[primary_key]
        roisweepcorrespondance_list.append(roisweepcorrespondance)
    
    with dj.conn().transaction: #inserting one movie
        print('uploading notes and groundtruth ROIs for subject {}'.format(movie['subject_id'])) #movie['movie_name']
        imaging_gt.ROISweepCorrespondance().insert(roisweepcorrespondance_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.MovieNote().insert(movienote_list, allow_direct_insert=True)
        dj.conn().ping()
        experiment.SessionComment().insert1(sessioncomment, allow_direct_insert=True)
        dj.conn().ping()
        ephys_cell_attached.SessionPipetteFill().insert1(sessionpipettefill, allow_direct_insert=True)
#%%
# =============================================================================
#         
# cells = ephys_cell_attached.Cell()      
# for cell in cells:
#     rois = imaging_gt.ROISweepCorrespondance()&cell
#     neuropil_numbers = np.unique((imaging.ROINeuropil()&rois).fetch('neuropil_number'))
#     for neuropil_number in neuropil_numbers:
#         imaging_gt.ROINeuropilCorrelation()&rois&'neuropil_number = {}'.format(neuropil_number)
#         break
#     break
# 
# =============================================================================

#%%
        

# =============================================================================
# def calculate_calcium_waves(cores = 4):
#     ray.init(num_cpus = cores)
#     #%%
#     subject_ids,sessions,movie_numbers = imaging.Movie().fetch('subject_id', 'session', 'movie_number')
#     for coreidx in range(cores):
#         result_ids = []
#         #%%
#         for subject_id,session,movie_number in zip(subject_ids,sessions,movie_numbers):
#             key = {'subject_id' : subject_id, 'session' : session,'movie_number' : movie_number}
#             #break
#         #%%
#             result_ids.append(calculate_calcium_waves_core.remote(key))    
#             #%
#         #result_ids = random.shuffle(result_ids)
#         ray.get(result_ids)
#     ray.shutdown()
#     
# @ray.remote    
# def calculate_calcium_waves_core(key):
#     #%%
#     #try:
#     #%%
#     if len(imaging_gt.CalciumWave()&key)>0:
#         #print('calcium waves for {} already calculated'.format(key))
#         return None
# # =============================================================================
# #     #%%
# # for subject_id,session,movie_number in zip(subject_ids,sessions,movie_numbers):
# #     key = {'subject_id' : subject_id, 'session' : session,'movie_number' : movie_number}
# # =============================================================================
#     #%%
# # =============================================================================
# #     if len(imaging_gt.CalciumWave()&key)>0:
# #         continue
# # =============================================================================
#     neuropil_correlation_min_corr_coeff=.5
#     neuropil_correlation_min_pixel_number=100
#     neuropil_r_fallback = .7
#     ca_wave_time_back = .5
#     ca_wave_time_forward = 2
#     global_f0_time = 10 #seconds
#     max_rise_time = .1 #seconds to find maximum after last AP in a group
#     gaussian_filter_sigmas = imaging_gt.CalciumWaveFilter().fetch('gaussian_filter_sigma')
#     neuropil_subtractions = imaging_gt.CalciumWaveNeuropilHandling().fetch('neuropil_subtraction')
#     groundtruth_rois = imaging_gt.ROISweepCorrespondance()&key#*lab.Surgery.VirusInjection()&'virus_id = 10257'
#     
#     calcium_wave_list = list()
#     calcium_wave_neuropil_list = list()
#     calcium_wave_properties_list = list()
#     #%
#     for groundtruth_roi in groundtruth_rois:
#         channels = imaging.MovieChannel()&groundtruth_roi
#         apgroups = ephysanal_cell_attached.APGroup()&groundtruth_roi
#         session_start_time = (experiment.Session&groundtruth_roi).fetch1('session_time')
#         cell_start_time = (ephys_cell_attached.Cell&groundtruth_roi).fetch1('cell_recording_start')
#         sweep_start_time = (ephys_cell_attached.Sweep&groundtruth_roi).fetch1('sweep_start_time')
#         sweep_timeoffset = (cell_start_time -session_start_time).total_seconds()
#         sweep_start_time = float(sweep_start_time)+sweep_timeoffset
#         roi_timeoffset = (imaging.ROI()&groundtruth_roi).fetch1('roi_time_offset')
#         frame_times = (imaging.MovieFrameTimes()&groundtruth_roi).fetch1('frame_times') + roi_timeoffset
#         framerate = 1/np.median(np.diff(frame_times))
#         ca_wave_step_back = int(ca_wave_time_back*framerate)
#         ca_wave_step_forward =int(ca_wave_time_forward*framerate)
#         global_f0_step = int(global_f0_time*framerate)
#         ophys_time = np.arange(-ca_wave_step_back,ca_wave_step_forward)/framerate*1000
#         #break
#         #%
#         for channel in channels:
#             groundtruth_roi_now = groundtruth_roi.copy()
#             groundtruth_roi_now['channel_number'] = channel['channel_number']
#             F = (imaging.ROITrace()&groundtruth_roi_now).fetch1('roi_f')
#             F_filt_list = list()
#             for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered ROI traces
#                 if gaussian_filter_sigma>0:
#                     F_filt_list.append(gaussFilter(F,sRate = framerate,sigma = gaussian_filter_sigma/1000))
#                 else:
#                     F_filt_list.append(F)
#             #F_filt = np.asarray(F_filt)
#             neuropils = np.asarray(list(imaging.ROINeuropil()&groundtruth_roi_now))
#             Fneu_list = list()
#             Fneu_filt_list = list()
#             for neuropil in neuropils: # extracting neuropils
#                 groundtruth_roi_neuropil = groundtruth_roi_now.copy()
#                 groundtruth_roi_neuropil['neuropil_number'] = neuropil['neuropil_number']
#                 try:
#                     Fneu_list.append((imaging.ROINeuropilTrace()&groundtruth_roi_neuropil).fetch1('neuropil_f'))   
#                     Fneu_filt_list_now = list()
#                     for gaussian_filter_sigma in gaussian_filter_sigmas: # create filtered neuropil traces
#                         if gaussian_filter_sigma>0:
#                             Fneu_filt_list_now.append(gaussFilter(Fneu_list[-1],sRate = framerate,sigma = gaussian_filter_sigma/1000))
#                         else:
#                             Fneu_filt_list_now.append(Fneu_list[-1])
#                     Fneu_filt_list.append(np.asarray(Fneu_filt_list_now))
#                 except:
#                     neuropils = neuropils[:len(Fneu_list)]
#             for apgroup in apgroups:
#                 groundtruth_roi_now['ap_group_number'] = apgroup['ap_group_number']
#                 ap_group_start_time = float(apgroup['ap_group_start_time'])+sweep_timeoffset
#                 ap_group_length = float(apgroup['ap_group_end_time']-apgroup['ap_group_start_time'])
#                 ca_wave_now = groundtruth_roi_now.copy()
#                 cawave_f = np.ones(len(ophys_time))*np.nan
#                 start_idx = np.argmin(np.abs(frame_times-ap_group_start_time))
#                 idxs = np.arange(start_idx-ca_wave_step_back,start_idx+ca_wave_step_forward)
#                 idxs_valid = (idxs>=0) & (idxs<len(F))
#                 if sum(idxs_valid)<len(idxs_valid):#np.abs(time_diff)>1/framerate:
#                     #print('calcium wave outside recording, skipping {}'.format(groundtruth_roi_now))
#                     continue
#                 cawave_f[idxs_valid] = F[idxs[idxs_valid]]
#                 time_diff = frame_times[start_idx]-ap_group_start_time
#                 
#                 cawave_time = ophys_time+time_diff*1000
#                 ca_wave_now['cawave_f'] = cawave_f
#                 ca_wave_now['cawave_time'] = cawave_time
#                 calcium_wave_list.append(ca_wave_now)
#                 
#                 for neuropil,Fneu,Fneu_filt in zip(neuropils,Fneu_list,Fneu_filt_list):
#                     ca_wave_neuropil_now = groundtruth_roi_now.copy()
#                     ca_wave_neuropil_now['neuropil_number'] = neuropil['neuropil_number']
#                     cawave_fneu = np.ones(len(ophys_time))*np.nan
#                     cawave_fneu[idxs_valid] = Fneu[idxs[idxs_valid]]
#                     ca_wave_neuropil_now['cawave_fneu'] = cawave_fneu
#                     calcium_wave_neuropil_list.append(ca_wave_neuropil_now)
#                     
#                     ca_wave_properties = groundtruth_roi_now.copy()
#                     ca_wave_properties['neuropil_number'] = neuropil['neuropil_number']                    
#                     for gaussian_filter_sigma,F_filt,Fneu_filt in zip(gaussian_filter_sigmas,F_filt_list,Fneu_filt): # create filtered ROI traces
#                         for neuropil_subtraction in neuropil_subtractions:
#                             ca_wave_properties_now = ca_wave_properties.copy()
#                             ca_wave_properties_now['gaussian_filter_sigma']=gaussian_filter_sigma
#                             ca_wave_properties_now['neuropil_subtraction']=neuropil_subtraction
#                             if neuropil_subtraction == 'none':
#                                 r_neu = 0
#                             elif neuropil_subtraction == 'calculated':
#                                 try:
#                                     r_neu,r_neu_corrcoeff,r_neu_pixelnum = (imaging_gt.ROINeuropilCorrelation()&ca_wave_properties).fetch1('r_neu','r_neu_corrcoeff','r_neu_pixelnum')
#                                     if r_neu_corrcoeff<neuropil_correlation_min_corr_coeff or r_neu_pixelnum <= neuropil_correlation_min_pixel_number:
#                                         1/0 # just to get to the exception
#                                 except:
#                                     #print('no calculated r_neuropil or low correlation coefficient for {}, resetting to .7'.format(ca_wave_properties_now))
#                                     r_neu = neuropil_r_fallback
#                                     
#                             else:
#                                 r_neu = float(neuropil_subtraction)
#                             
#                             F_subtracted = F_filt-Fneu_filt*r_neu
#                             F0_gobal = np.percentile(F_subtracted[np.max([0,int(start_idx-global_f0_step/2)]):np.min([len(F_subtracted),int(start_idx+global_f0_step/2)])],10)
#                             cawave_f_now = np.ones(len(ophys_time))*np.nan
#                             cawave_f_now[idxs_valid] = F_subtracted[idxs[idxs_valid]]
#                             #%
#                             if max_rise_time+ap_group_length>cawave_time[-1]/1000:
#                                 end_idx = len(cawave_time)
#                             else:
#                                 end_idx =np.argmax(max_rise_time+ap_group_length<cawave_time/1000)
#                             peak_amplitude_idx = np.argmax(cawave_f_now[ca_wave_step_back:end_idx])+ca_wave_step_back
#                             #%
#                             ca_wave_properties_now['cawave_rneu'] =r_neu
#                             ca_wave_properties_now['cawave_baseline_f'] = np.nanmean(cawave_f_now[:ca_wave_step_back])
#                             ca_wave_properties_now['cawave_baseline_f_std'] = np.nanstd(cawave_f_now[:ca_wave_step_back])
#                             ca_wave_properties_now['cawave_baseline_dff_global'] = (ca_wave_properties_now['cawave_baseline_f']-F0_gobal)/F0_gobal
#                             ca_wave_properties_now['cawave_peak_amplitude_f'] = cawave_f_now[peak_amplitude_idx]-ca_wave_properties_now['cawave_baseline_f']
#                             ca_wave_properties_now['cawave_peak_amplitude_dff'] = ca_wave_properties_now['cawave_peak_amplitude_f']/ca_wave_properties_now['cawave_baseline_f']
#                             ca_wave_properties_now['cawave_rise_time'] = cawave_time[peak_amplitude_idx]
#                             ca_wave_properties_now['cawave_snr'] = ca_wave_properties_now['cawave_baseline_f']/ca_wave_properties_now['cawave_baseline_f_std']
#                             calcium_wave_properties_list.append(ca_wave_properties_now)
#      #%%                   
#     with dj.conn().transaction: #inserting one movie
#         #print('uploading calcium waves for {}'.format(key)) #movie['movie_name']
#         imaging_gt.CalciumWave().insert(calcium_wave_list, allow_direct_insert=True)
#         dj.conn().ping()
#         imaging_gt.CalciumWave.CalciumWaveNeuropil().insert(calcium_wave_neuropil_list, allow_direct_insert=True)
#         dj.conn().ping()
#         imaging_gt.CalciumWave.CalciumWaveProperties().insert(calcium_wave_properties_list, allow_direct_insert=True)
#             #%%
# # =============================================================================
# #     except:
# #         print('ERROR uploading calcium waves for {}'.format(key))
# # =============================================================================
# 
# =============================================================================

        
#%%
# =============================================================================
# #%%
# import scipy.ndimage as ndimage
# import scipy.signal as signal
# import os
# def gaussFilter(sig,sRate,sigma = .00005):
#     si = 1/sRate
#     #sigma = .00005
#     sig_f = ndimage.gaussian_filter(sig,sigma/si)
#     return sig_f
# ca_wave_time_back = .1
# ca_wave_time_forward = 1
# channel_number = 1
# neuropil_estimation_decay_time = 3 # - after each AP this interval is skipped for estimating the contribution of the neuropil
# neuropil_estimation_filter_sigma = .05
# neuropil_estimation_median_filter_time = 4 #seconds
# neuropil_estimation_min_filter_time = 10 #seconds
# groundtruth_rois = imaging_gt.ROISweepCorrespondance()#*lab.Surgery.VirusInjection()&'virus_id = 10257'
# for groundtruth_roi in groundtruth_rois:
#     groundtruth_roi['neuropil_number']=0
#     #%
#     #neuropil_estimation_filter_sigma_highpass = .25
#     try:
#         reponame=(imaging.RegisteredMovieFile()&groundtruth_roi).fetch('reg_movie_file_repository')[0]
#         repodir = dj.config['locations.{}'.format(reponame)]
#         moviedir=os.path.join(repodir,(imaging.RegisteredMovieFile()&groundtruth_roi).fetch('reg_movie_file_directory')[0][:-8])
#         fneu_4pix = np.load(os.path.join(moviedir,'Fneu_min_neuropil_pixels_4.npy'))
#         fneu_4pix = fneu_4pix[groundtruth_roi['roi_number'],:]
#         fneu_8pix = np.load(os.path.join(moviedir,'Fneu_min_neuropil_pixels_8.npy'))
#         fneu_8pix = fneu_8pix[groundtruth_roi['roi_number'],:]
#         fneu_16pix = np.load(os.path.join(moviedir,'Fneu_min_neuropil_pixels_16.npy'))
#         fneu_16pix = fneu_16pix[groundtruth_roi['roi_number'],:]
#     except:
#         pass
#     apgroups = ephysanal_cell_attached.APGroup()&groundtruth_roi
#     apgroups_start_time = np.asarray(apgroups.fetch('ap_group_start_time'),float)
#     session_start_time = (experiment.Session&groundtruth_roi).fetch1('session_time')
#     cell_start_time = (ephys_cell_attached.Cell&groundtruth_roi).fetch1('cell_recording_start')
#     sweep_start_time = (ephys_cell_attached.Sweep&groundtruth_roi).fetch1('sweep_start_time')
#     sweep_timeoffset = (cell_start_time -session_start_time).total_seconds()
#     sweep_start_time = float(sweep_start_time)+sweep_timeoffset
#     apmaxtimes = np.asarray((ephysanal_cell_attached.ActionPotential()&groundtruth_roi).fetch('ap_max_time'),float) +sweep_start_time
#     roi_timeoffset = (imaging.ROI()&groundtruth_roi).fetch1('roi_time_offset')
#     frame_times = (imaging.MovieFrameTimes()&groundtruth_roi).fetch1('frame_times') + roi_timeoffset
#     framerate = 1/np.median(np.diff(frame_times))
#     groundtruth_roi['channel_number'] = channel_number
#     rechannel = groundtruth_roi.copy()
#     rechannel['channel_number'] = 2
#     F = (imaging.ROITrace()&groundtruth_roi).fetch1('roi_f')
#     F_red = (imaging.ROINeuropilTrace()&rechannel).fetch1('neuropil_f')
#     Fneu = (imaging.ROINeuropilTrace()&groundtruth_roi).fetch1('neuropil_f')
#     ephys_trace,sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()&groundtruth_roi).fetch1('response_trace','sample_rate')
#     ephys_t = np.arange(len(ephys_trace))/sample_rate+sweep_start_time
#     
#     decay_step = int(neuropil_estimation_decay_time*framerate)
#     F_activity = np.zeros(len(frame_times))
#     for ap_time in apmaxtimes:
#         F_activity[np.argmax(frame_times>ap_time)] = 1
#     F_activitiy_conv= np.convolve(F_activity,np.concatenate([np.zeros(decay_step),np.ones(decay_step)]),'same')
#     F_activitiy_conv= np.convolve(F_activitiy_conv[::-1],np.concatenate([np.zeros(int((neuropil_estimation_filter_sigma*framerate*5))),np.ones(int((neuropil_estimation_filter_sigma*framerate*5)))]),'same')[::-1]
#     F_activitiy_conv[:decay_step]=1
#     needed =F_activitiy_conv==0
#     if sum(needed)> 100:
#         F_orig_filt = gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma)
#         try:
#            #F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#            F_orig_filt_ultra_low =  ndimage.minimum_filter(F_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))
#         except:
#             F_orig_filt_ultra_low = signal.medfilt(F_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(F,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#             
#         F_orig_filt_highpass = F_orig_filt-F_orig_filt_ultra_low + F_orig_filt_ultra_low[0]
#         Fneu_orig_filt = gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma)
#         try:
#             #Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#             Fneu_orig_filt_ultra_low =  ndimage.minimum_filter(Fneu_orig_filt, size=int(neuropil_estimation_min_filter_time*framerate))
#         except:
#             Fneu_orig_filt_ultra_low =signal.medfilt(Fneu_orig_filt, kernel_size=int(neuropil_estimation_median_filter_time*framerate)+1) #gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma_highpass)
#         
#         Fneu_orig_filt_highpass = Fneu_orig_filt-Fneu_orig_filt_ultra_low + Fneu_orig_filt_ultra_low[0]
#         p=np.polyfit(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],1)
#         R_neuopil_fit = np.corrcoef(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed])
#         r_neu_now =p[0]
#         neuropilvals = np.asarray([np.min(Fneu_orig_filt_highpass[needed]),np.max(Fneu_orig_filt_highpass[needed])])
#         fittedvals = np.polyval(p,neuropilvals)
#     else:
#         print('skipped')
#         continue
#     #r_neu_now = .7
#     fig = plt.figure()
#     ax_ophys = fig.add_subplot(321)
#     ax_neuropil = fig.add_subplot(322)
#     ax_ophys_subtracted = fig.add_subplot(323,sharex = ax_ophys)
#     ax_ephys = fig.add_subplot(325,sharex = ax_ophys)
#     ax_ophys.plot(frame_times,F,'g-')
#     ax_ophys.plot(frame_times,gaussFilter(F_red,framerate,sigma = neuropil_estimation_filter_sigma),'r-')
#     ax_ophys.plot(frame_times,gaussFilter(Fneu,framerate,sigma = neuropil_estimation_filter_sigma),'c-')
#     
#     try:
#         ax_ophys.plot(frame_times,gaussFilter(fneu_4pix,framerate,sigma = neuropil_estimation_filter_sigma),'b-')
#         ax_ophys.plot(frame_times,gaussFilter(fneu_8pix,framerate,sigma = neuropil_estimation_filter_sigma),'g-')
#         ax_ophys.plot(frame_times,gaussFilter(fneu_16pix,framerate,sigma = neuropil_estimation_filter_sigma),'y-')
#     except:
#         pass
#     ax_ophys.plot(apgroups_start_time+sweep_timeoffset,np.zeros(len(apgroups_start_time)),'ro')
#     ax_ophys.plot(apmaxtimes,np.zeros(len(apmaxtimes)),'k|')
#     
#     ax_ophys_subtracted.plot(frame_times,F-Fneu_orig_filt*r_neu_now,'g-')
#     ax_ophys_subtracted.plot(frame_times,gaussFilter(F-Fneu_orig_filt*r_neu_now,framerate,sigma = neuropil_estimation_filter_sigma),'k-')
#     xnow = frame_times
#     xnow[needed==False] = np.nan
#     ynow = F_orig_filt
#     ynow[needed==False] = np.nan
#     ax_ophys.plot(xnow,ynow,'k-')
#     ynow = Fneu_orig_filt
#     ynow[needed==False] = np.nan
#     ax_ophys.plot(xnow,ynow,'k-')
#     ax_neuropil.plot(Fneu_orig_filt_highpass[needed],F_orig_filt_highpass[needed],'ko')
#     ax_neuropil.plot(neuropilvals,fittedvals,'r-')
#     ax_neuropil.set_title('F = Fneu * {:01.2f} + {:02.1f} -- R = {:01.2f}'.format(p[0],p[1],R_neuopil_fit[1][0]))
#     ax_neuropil.set_ylabel('F')
#     ax_neuropil.set_xlabel('neuropil')
#     
#     ax_ephys.plot(ephys_t,ephys_trace,'k-')
#     plt.show()
#     #% break
#     for apgroup in apgroups:
#         print(apgroup)
#         break
#     
#     time.sleep(3)
# =============================================================================


