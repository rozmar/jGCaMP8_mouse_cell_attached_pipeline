from statistics import mode
import utils.utils_pipeline as utils_pipeline
import datetime
import os
import numpy as np
import datajoint as dj
import time as timer
import re
dj.conn()
from pipeline import  experiment, ephys_cell_attached, imaging, imaging_gt
#%
def populate_behavior():
    #%
    base_vis_stim_dir= '/home/rozmar/Data/Visual_stim/raw/Genie_2P_rig'#there is only a single rig..
    #dirnames = os.listdir(base_vis_stim_dir)
    for session in experiment.Session():
        #%
        subject_id = session['subject_id']
        session_date = session['session_date']
        session_time = (datetime.datetime.combine(session_date,datetime.datetime.min.time())+session['session_time'])#.time()
        dirname = session_date.strftime('%Y%m%d-anm{}'.format(subject_id))
        sessiondir = os.path.join(base_vis_stim_dir,dirname)
        runnames = os.listdir(sessiondir)
        filenames_list = list()
        runnames_list = list()
        runtimes_list = list()
        for runname in runnames:
            filenames_now = os.listdir(os.path.join(sessiondir,runname))
            if len(filenames_now) == 0:
                continue
            filename_now = np.sort(filenames_now)[-1]
            #for filename_now in filenames_now:
            runtimes_list.append(datetime.datetime.strptime(filename_now[:20],'%d-%b-%Y_%H_%M_%S'))
            filenames_list.append(filename_now)
            runnames_list.append(runname)
        
        runtimes_list = np.asarray(runtimes_list)# - datetime.timedelta(seconds=time_offset)
        filenames_list = np.asarray(filenames_list)
        runnames_list = np.asarray(runnames_list)
        
        order = np.argsort(runtimes_list)
        runtimes_list=runtimes_list[order]
        filenames_list=filenames_list[order]
        runnames_list=runnames_list[order]
        
        #cell_recording_starts,sweep_start_times,cell_numbers,sweep_numbers,sweep_end_times= (ephys_cell_attached.Cell()*ephys_cell_attached.Sweep()&session).fetch('cell_recording_start','sweep_start_time','cell_number','sweep_number','sweep_end_time')
        movie_frame_times_list,movie_names = (imaging.Movie()*imaging.MovieFrameTimes()&session).fetch('frame_times','movie_name')
        
        runname_ids = list()
        for runname_now in runnames_list:
            matches = re.findall("(\d+)", runname_now)
            if len(matches)>1:
                runname_ids.append('{}_{}'.format(int(matches[0]),int(matches[1])))
            else:
                print('could not decode {}'.format(runname_now))
                runname_ids.append('garbage')
        runname_ids = np.asarray(runname_ids)
        movie_ids = list()
        for runname_now in movie_names:
            matches = re.findall("(\d+)", runname_now)
            if len(matches)>1:
                movie_ids.append('{}_{}'.format(int(matches[0]),int(matches[1])))
            else:
                print('could not decode {}'.format(runname_now))   
                runname_ids.append('garbageeee')
          
        
        visualstim_trial_list = list()
        visual_stim_trial = 0
        dj.conn().ping()
        sweep_number= 0
        for frame_times,movie_name,movie_id in zip(movie_frame_times_list,movie_names,movie_ids): # start feeding the trials in order
            if not any(movie_id==runname_ids):
                print('no visual stim found for movie {}'.format(movie_name))
            idx = np.where(movie_id==runname_ids)[0][0]
            runname = runnames_list[idx]
            filename = filenames_list[idx]
            visualstimfile = os.path.join(sessiondir,runname,filename)
            try:
                visdata = utils_pipeline.processVisMat(visualstimfile)
            except:
                print('corrupted visual stim file, aborting')
                continue
            sweep_number+=1
            frames_back = visdata['gray_dur']*visdata['fs']     
            for amplitude,angle,cyclesperdegree,cyclespersecond,duration,frame_count,stim_init in zip(visdata['trials_list']['amplitude'],
                                                                                                      visdata['trials_list']['angle'],
                                                                                                      visdata['trials_list']['cyclesperdegree'],
                                                                                                      visdata['trials_list']['cyclespersecond'],
                                                                                                      visdata['trials_list']['duration'],
                                                                                                      visdata['trials_list']['frame_count'],
                                                                                                      visdata['stim_init']):
                
                start_idx = np.max([1,stim_init-frames_back])
                frame_count = np.min([frame_count,len(frame_times)])
                visual_stim_trial+=1
                visualstim_trial_key = {'subject_id':subject_id,
                                        'session':session['session'],
                                        'visual_stim_trial':visual_stim_trial,
                                        'visual_stim_sweep':sweep_number,
                                        'visual_stim_sweep_time':frame_times[stim_init-1]-frame_times[0],
                                        'visual_stim_trial_start':frame_times[start_idx-1],
                                        'visual_stim_trial_end':frame_times[frame_count-1],
                                        'visual_stim_grating_start':frame_times[stim_init-1],
                                        'visual_stim_grating_end':frame_times[frame_count-1],
                                        'visual_stim_grating_angle':angle,
                                        'visual_stim_grating_cyclespersecond':cyclespersecond,
                                        'visual_stim_grating_cyclesperdegree':cyclesperdegree,
                                        'visual_stim_grating_amplitude':amplitude}
                
                visualstim_trial_list.append(visualstim_trial_key)
# =============================================================================
#         print('aaa')
#         timer.sleep(1000)
# =============================================================================
        
        with dj.conn().transaction: #inserting one movie
            print('uploading visual stim for subject {}'.format(session['subject_id'])) #movie['movie_name']
            try:
                experiment.VisualStimTrial().insert(visualstim_trial_list, allow_direct_insert=True)
                dj.conn().ping()
            except dj.errors.DuplicateError:
                print('already uploaded')
        
        
        
        

        
        
       