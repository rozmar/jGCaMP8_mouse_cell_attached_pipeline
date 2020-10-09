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
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, imaging, ephys_cell_attached, ephysanal_cell_attached
from utils_pipeline import extract_scanimage_metadata, extract_files_from_dir
#import ray
import time

homefolder = dj.config['locations.mr_share']
#%%
def upload_movie_metadata_scanimage():   
#%%             
    repository = 'mr_share'
    if repository == 'mr_share':  
        repodir = dj.config['locations.mr_share']
        basedir = os.path.join(repodir, 'Data/Calcium_imaging/raw/Genie_2P_rig')
    #basedir = '/home/rozmar/Network/Voltage_imaging_rig_1p_imaging/rozsam'
    #%
    dirs = os.listdir(basedir)
    sessiondates = experiment.Session().fetch('session_date')
    #%
    for session_now in dirs:
        if not os.path.isdir(os.path.join(basedir,session_now)): # break out from session loop
                continue
        try:
            session_date = datetime.datetime.strptime(session_now[:8], '%Y%m%d').date()#.strftime('%Y-%m-%d')
            subject_id = int(session_now[-6:])
        except:
            print('couldn''t assign date to {}'.format(session_now))
            continue
        
        
        if session_date in sessiondates:
            try:
                session = (experiment.Session() & 'session_date = "'+str(session_date)+'"' & 'subject_id = {}'.format(subject_id)).fetch1('session')   
            except: # session wasn't found
                continue
            key = {'subject_id':subject_id,'session':session}
            session_dir_now = os.path.join(basedir, session_now)
            
            cells = os.listdir(session_dir_now)
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
                cell_is_dir.append(os.path.isdir(os.path.join(session_dir_now,cell)))
            cells = np.array(cells)[~np.array(np.isnan(cell_nums)) & cell_is_dir]
            cell_nums = np.array(cell_nums)[~np.array(np.isnan(cell_nums))& cell_is_dir]
            #%
            if len(cells)==0:
                continue
            cell_order = np.argsort(cell_nums)
            cell_nums = cell_nums[cell_order]
            cells = cells[cell_order]
            
            

            for moviedir in cells:
                movie_dir_now = os.path.join(session_dir_now,moviedir)
                if os.path.isdir(movie_dir_now):
# =============================================================================
#                     print('asdasdsa')
#                     time.sleep(1000)
# =============================================================================
                    upload_movie_metadata_scanimage_core(movie_dir_now,
                                                         key = key,
                                                         repository = repository,
                                                         repodir = repodir)
                        
  #%%                      
def upload_movie_metadata_scanimage_core(movie_dir_now,key,repository,repodir): # TODO It's still the script for ingesting .tif files from HCimage live
    #%%
    subject_id=key['subject_id']
    session=key['session']
    files_dict = extract_files_from_dir(movie_dir_now)
    uniquebasenames = np.unique(files_dict['basenames'][files_dict['exts']=='.tif'])
    uniquebasenames = np.sort(uniquebasenames)
    
    print(movie_dir_now)
    #%
    for basename in uniquebasenames:
        if 'zstack' in basename.lower():
            continue
        Movie_list=list()
        MovieFrameTimes_list = list()
        MovieMetaData_list = list()
        MovieFile_list = list()
        MovieChannels_list = list()
        movinamesuploaded_sofar = (imaging.Movie()&key).fetch('movie_name')
        if basename in movinamesuploaded_sofar:
            continue # movie already uploaded
        movie_idx = len(movinamesuploaded_sofar)
        print(basename)


        files_now = files_dict['filenames'][(files_dict['basenames'] == basename) &(files_dict['exts']=='.tif')]
        
        indices_now = files_dict['fileindices'][(files_dict['basenames'] == basename) &(files_dict['exts']=='.tif')]
        order = np.argsort(indices_now)
        files_now = files_now[order]
        frame_count = list()
        metadata_list = list()
        try:
            for file_now in files_now: 
                tifffile = os.path.join(movie_dir_now,file_now)
                scanimage_metadata = extract_scanimage_metadata(tifffile)            
                metadata_list.append(scanimage_metadata)
                frame_count.append(scanimage_metadata['shape'][0])
        except:
            print('could not open tiff file {} . Corrupted file?'.format(file_now))
            continue
        #break
        #%
        movie_time = metadata_list[0]['movie_start_time']
        session_datetime = (datetime.datetime.combine((experiment.Session()&key).fetch1('session_date'), datetime.time.min )+ (experiment.Session()&key).fetch1('session_time'))
        celldatetimes = (datetime.datetime.combine((experiment.Session()&key).fetch1('session_date'), datetime.time.min ) + (ephys_cell_attached.Cell()&key).fetch('cell_recording_start'))
        potential_cell_number = (ephys_cell_attached.Cell()&key).fetch('cell_number')[(celldatetimes<movie_time).argmin()-1]     
        key_cell = key.copy()
        key_cell['cell_number'] = potential_cell_number
        #%
        cell_time = (ephys_cell_attached.Cell()&key_cell).fetch1('cell_recording_start')
        cell_date =  datetime.datetime.combine((experiment.Session()&key_cell).fetch1('session_date'), datetime.time.min )
        if cell_time.total_seconds()/3600 < 4:
            cell_date = cell_date + datetime.timedelta(days=1)
        cell_datetime = cell_date+ cell_time

        sweep_nums,sweep_start_times = (ephys_cell_attached.Sweep()& key_cell).fetch('sweep_number','sweep_start_time')
        movie_start_timediff = list()
        for sweep_start_time in sweep_start_times :
            sweep_start_time_real = cell_datetime+datetime.timedelta(seconds =float(sweep_start_time))
            movie_start_timediff.append((movie_time-sweep_start_time_real).total_seconds())
        sweep_num = sweep_nums[np.argmin(np.abs(movie_start_timediff))]
        if np.min(np.abs(movie_start_timediff))>10: # 10 seconds is the max difference
            print('no sweep found for movie {}'.format(basename))
            
            continue
        
        #% calculate frame times
        scanimage_metadata = metadata_list[0]
        movie_frame_num = int(scanimage_metadata['metadata']['hStackManager']['framesPerSlice'])
        cell_start_time = (cell_datetime-session_datetime).total_seconds() # in seconds relative to session start
        if cell_start_time<0:
            print('negative time - solve it')
            time.sleep(100000)
        frametimes = (ephysanal_cell_attached.SweepFrameTimes()&key_cell&'sweep_number={}'.format(sweep_num)).fetch1('frame_time')
        frametimes = frametimes[:movie_frame_num]+cell_start_time # from session start
        if not (movie_frame_num == sum(frame_count) or movie_frame_num == sum(frame_count)/2):
            print('movie frame count not consistent with metadata, aborting {}'.format(basename))
            continue
        if movie_frame_num > len(frametimes):
            short_ephys_whitelist = [{'subject_id': 478404, 'session': 1, 'cell_number': 8, 'sweep_number': 5}]
            key_sweep = key_cell.copy()
            key_sweep['sweep_number'] = sweep_num
            if key_sweep in short_ephys_whitelist:
                print('not enough frametimes but whitelisted')
                continue
            else:
                print('not enough frametimes..WTF?!')
                #continue
                time.sleep(10000)
# =============================================================================
#             #%%
#             frames_trace = (ephys_cell_attached.SweepImagingExposure()&key_sweep).fetch1('imaging_exposure_trace')
#             frameidxes = (ephysanal_cell_attached.SweepFrameTimes()&key_cell&'sweep_number={}'.format(sweep_num)).fetch1('frame_idx')
#             plt.plot(frames_trace);plt.plot(frameidxes,frames_trace[frameidxes],'ro')
#             #%%
# =============================================================================
        movie_frametime_data = {'subject_id':subject_id,
                                'session':session,
                                'movie_number':movie_idx,
                                'frame_times':frametimes}
        MovieFrameTimes_list.append(movie_frametime_data)
        moviedata = {'subject_id':subject_id,
                     'session':session,
                     'movie_number':movie_idx,
                     'movie_name' : basename,
                     'movie_x_size' : int(scanimage_metadata['metadata']['hRoiManager']['pixelsPerLine']),
                     'movie_y_size' : int(scanimage_metadata['metadata']['hRoiManager']['linesPerFrame']),
                     'movie_frame_rate': float(scanimage_metadata['metadata']['hRoiManager']['scanVolumeRate']),
                     'movie_frame_num' : movie_frame_num,
                     'movie_start_time' : frametimes[0],
                     'movie_pixel_size': np.min(scanimage_metadata['roi_metadata'][0]['scanfields']['sizeXY'])}
        Movie_list.append(moviedata)
        
        try:
            xyzstring = scanimage_metadata['metadata']['hMotors']['samplePosition']
        except: # old file format
            xyzstring = scanimage_metadata['metadata']['hMotors']['motorPosition']
        xyz = np.asarray(xyzstring.strip(']').strip('[').split(' '),float)  
        print(xyz)
        try:
            mask = np.asarray(scanimage_metadata['metadata']['hScan2D']['mask'].strip(']').strip('[').split(';'),int)
        except: # old file format
            mask = None
        try:
            scanmode = scanimage_metadata['metadata']['hScan2D']['scanMode'].strip("'")
        except: # old file format
            scanmode = scanimage_metadata['metadata']['hScan2D']['scannerType'].strip("'").lower()
        moviemetadata = {'subject_id':subject_id,
                         'session':session,
                         'movie_number':movie_idx,
                         'movie_hbeam_power':float(scanimage_metadata['metadata']['hBeams']['powers']),
                         'movie_hmotors_sample_x':xyz[0],
                         'movie_hmotors_sample_y':xyz[1],
                         'movie_hmotors_sample_z':xyz[2],
                         'movie_hroimanager_lineperiod':float(scanimage_metadata['metadata']['hRoiManager']['linePeriod']),
                         'movie_hroimanager_scanframeperiod':float(scanimage_metadata['metadata']['hRoiManager']['scanFramePeriod']),
                         'movie_hroimanager_scanzoomfactor':float(scanimage_metadata['metadata']['hRoiManager']['scanZoomFactor']),
                         'movie_hscan2d_fillfractionspatial':float(scanimage_metadata['metadata']['hScan2D']['fillFractionSpatial']),
                         'movie_hscan2d_fillfractiontemporal':float(scanimage_metadata['metadata']['hScan2D']['fillFractionTemporal']),
                         'movie_hscan2d_mask':mask,
                         'movie_hscan2d_scanmode':scanmode}
        MovieMetaData_list.append(moviemetadata)
        #%
        savechannels = np.asarray(scanimage_metadata['metadata']['hChannels']['channelSave'].strip(']').strip('[').split(';'),int)
        savechannels_idx = savechannels -1
        offset = np.asarray(scanimage_metadata['metadata']['hChannels']['channelOffset'].strip(']').strip('[').split(' '),int)
        subtractoffset = np.asarray(scanimage_metadata['metadata']['hChannels']['channelSubtractOffset'].strip(']').strip('[').split(' '),str)=='true'
        color = np.asarray(scanimage_metadata['metadata']['hChannels']['channelMergeColor'].strip("'}").strip("{'").split("';'"),str)
        
        
        offset = offset[savechannels_idx]
        subtractoffset = np.asarray(subtractoffset[savechannels_idx],int)
        color = color[savechannels_idx]
        offset = offset[savechannels_idx]
        #%
        for channel_number,channel_color,channel_offset,channel_subtract_offset in zip(savechannels,
                                                                                       color,
                                                                                       offset,
                                                                                       subtractoffset):
            moviechannelsdata = {'subject_id':subject_id,
                             'session':session,
                             'movie_number':movie_idx,
                             'channel_number':channel_number,
                             'channel_color':channel_color,
                             'channel_offset':channel_offset,
                             'channel_subtract_offset':channel_subtract_offset}
            MovieChannels_list.append(moviechannelsdata)
        
        
        for movie_file_number, (movie_file_name,movie_frame_count) in enumerate(zip(files_now.tolist(),frame_count)):
            moviefiledata_now  = {'subject_id':subject_id,
                                  'session':session,
                                  'movie_number' : movie_idx,
                                  'movie_file_number' : movie_file_number,
                                  'movie_file_repository': repository,
                                  'movie_file_directory' : movie_dir_now[len(repodir)+1:],
                                  'movie_file_name' : movie_file_name,
                                  'movie_file_start_frame' : 0,
                                  'movie_file_end_frame' : movie_frame_count}
            MovieFile_list.append(moviefiledata_now)
            
            
            
            #%
        with dj.conn().transaction: #inserting one movie
            print('uploading to datajoint: {}'.format(movie_dir_now))
            imaging.Movie().insert(Movie_list, allow_direct_insert=True)
            dj.conn().ping()
            imaging.MovieMetaData().insert(MovieMetaData_list, allow_direct_insert=True)
            dj.conn().ping()
            imaging.MovieFile().insert(MovieFile_list, allow_direct_insert=True)
            dj.conn().ping()
            imaging.MovieFrameTimes().insert(MovieFrameTimes_list, allow_direct_insert=True)
            dj.conn().ping()
            imaging.MovieChannel().insert(MovieChannels_list, allow_direct_insert=True)
            dj.conn().ping()
