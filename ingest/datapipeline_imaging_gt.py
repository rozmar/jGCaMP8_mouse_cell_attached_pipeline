import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time as timer
import numpy as np
import datajoint as dj
from PIL import Image
import scipy
import re 
import os
import shutil
import pandas as pd
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, imaging, ephys_cell_attached, ephysanal_cell_attached, imaging_gt
from utils_pipeline import extract_scanimage_metadata, extract_files_from_dir
from suite2p.detection.masks import create_neuropil_masks, create_cell_pix
#import ray
import time

homefolder = dj.config['locations.mr_share']

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
#
