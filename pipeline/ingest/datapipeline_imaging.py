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
from utils.utils_pipeline import extract_scanimage_metadata, extract_files_from_dir
from suite2p.detection.masks import create_neuropil_masks, create_cell_pix
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
                                  'movie_file_directory' : movie_dir_now[len(repodir):],
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

#% suite2p registration
def upload_movie_registration_scanimage():
    subject_ids = np.unique(imaging.Movie().fetch('subject_id'))
    for subject_id in subject_ids:
        if len(imaging.RegisteredMovie()&'subject_id = {}'.format(subject_id)&'motion_correction_method= "Suite2P"')==0:
            print('loading registration for {}'.format(subject_id))
            upload_movie_registration_scanimage_core(subject_id)
    

def upload_movie_registration_scanimage_core(subject_id):
    #%
    movies = imaging.Movie()&'subject_id = {}'.format(subject_id)
    registered_movie_list = list()
    registered_movie_image_list = list()
    motion_correction_metrics_list = list()
    registered_movie_file_list = list()
    motion_correction_list = list()
    
    for movie in movies:
        movie_name = movie['movie_name']
        moviefiles = imaging.MovieFile()&movie
        reponame = list(moviefiles)[0]['movie_file_repository']
        repodir = dj.config['locations.{}'.format(reponame)]
        movie_file_directory = list(moviefiles)[0]['movie_file_directory']
        suite2p_movie_directory = movie_file_directory.replace('raw','suite2p')
        try:
            suite2p_dir = os.path.join(repodir,suite2p_movie_directory,movie_name)
            s2p_files=os.listdir(suite2p_dir)
        except:
            suite2p_dir = os.path.join(repodir,'D'+suite2p_movie_directory,movie_name) #HOTFIX - stupid indexing mistake in ingest script
            s2p_files=os.listdir(suite2p_dir)
        reg_metadata = np.load(os.path.join(suite2p_dir,'plane0','ops.npy'),allow_pickle = True).tolist()
        registered_movie = dict()
        for primary_key in imaging.Movie.primary_key:
            registered_movie[primary_key] = movie[primary_key]
        registered_movie['motion_correction_method'] = 'Suite2P'
        registered_movie_list.append(registered_movie)
        movie_channels = imaging.MovieChannel()&movie
        for movie_channel in movie_channels:
            registered_movie_image = registered_movie.copy()
            registered_movie_image['channel_number'] = movie_channel['channel_number']
            if movie_channel['channel_number']==reg_metadata['functional_chan']:
                registered_movie_image['registered_movie_mean_image'] = reg_metadata['meanImg']
                try:
                    registered_movie_image['registered_movie_mean_image_enhanced'] = reg_metadata['meanImgE']
                except:
                    registered_movie_image['registered_movie_mean_image_enhanced']= None
                try:
                    registered_movie_image['registered_movie_max_projection'] = reg_metadata['max_proj']
                except:
                    registered_movie_image['registered_movie_max_projection'] = None
            else:
                registered_movie_image['registered_movie_mean_image'] = reg_metadata['meanImg_chan2']
                try:
                    registered_movie_image['registered_movie_mean_image_enhanced'] = reg_metadata['meanImg_chan2_corrected']
                except:
                    registered_movie_image['registered_movie_mean_image_enhanced'] = None
                registered_movie_image['registered_movie_max_projection'] = None
            registered_movie_image_list.append(registered_movie_image)
        motion_correction_metrics = registered_movie.copy()
        motion_correction_metrics['motion_corr_residual_rigid']=  reg_metadata['regDX'][:,0]
        motion_correction_metrics['motion_corr_residual_nonrigid']=reg_metadata['regDX'][:,1]
        motion_correction_metrics['motion_corr_residual_nonrigid_max']= reg_metadata['regDX'][:,2]
        motion_correction_metrics['motion_corr_tpc']=reg_metadata['tPC']
        motion_correction_metrics_list.append(motion_correction_metrics)
        
        reg_directory_full = os.path.join(suite2p_dir,'plane0','reg_tif')
        reg_movie_files = np.sort(os.listdir(reg_directory_full))
        
        for reg_movie_file_number,reg_movie_file in enumerate(reg_movie_files):
            registered_movie_file = registered_movie.copy()
            registered_movie_file['reg_movie_file_number'] = reg_movie_file_number
            registered_movie_file['reg_movie_file_repository'] = reponame
            registered_movie_file['reg_movie_file_directory'] =reg_directory_full[len(repodir):]
            registered_movie_file['reg_movie_file_name'] = reg_movie_file
            registered_movie_file_list.append(registered_movie_file)
        
        motion_corr_descriptions = ['rigid registration','nonrigid registration']
        for motion_correction_id,motion_corr_description in enumerate(motion_corr_descriptions):
            motion_correction = registered_movie.copy()
            motion_correction['motion_correction_id'] = motion_correction_id
            motion_correction['motion_corr_description'] = motion_corr_description
            if 'nonrigid' in motion_corr_description:
                motion_correction['motion_corr_x_block']=reg_metadata['xblock']
                motion_correction['motion_corr_y_block']=reg_metadata['yblock']
                motion_correction['motion_corr_x_offset']=reg_metadata['xoff1']
                motion_correction['motion_corr_y_offset'] =reg_metadata['yoff1']
            else:
                motion_correction['motion_corr_x_block']=None
                motion_correction['motion_corr_y_block']=None
                motion_correction['motion_corr_x_offset']= reg_metadata['xoff']
                motion_correction['motion_corr_y_offset'] =reg_metadata['yoff'] 
            motion_correction_list.append(motion_correction)
            
    
    with dj.conn().transaction: #inserting one movie
        print('uploading to datajoint: subject {}'.format(movie['subject_id']))
        imaging.RegisteredMovie().insert(registered_movie_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.RegisteredMovieImage().insert(registered_movie_image_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.MotionCorrectionMetrics().insert(motion_correction_metrics_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.RegisteredMovieFile().insert(registered_movie_file_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.MotionCorrection().insert(motion_correction_list, allow_direct_insert=True)
        dj.conn().ping()
#%% upload scanimage ROIs
def upload_ROIs_scanimage():
    #%%
    subject_ids = np.unique(imaging.Movie().fetch('subject_id'))
    for subject_id in subject_ids:
        print('loading ROIs for {}'.format(subject_id))
        upload_ROIs_scanimage_core(subject_id)
       #%%     
def upload_ROIs_scanimage_core(subject_id):
    #%%
    movies = imaging.Movie()&'subject_id = {}'.format(subject_id)
    roi_list = list()
    roi_neuropil_list = list()
    roi_trace_list = list()
    roi_neuropil_trace_list = list()
    for movie in movies:
        
        if len(imaging.ROI()&movie)>0:
            continue
        #%
        registered_movie = imaging.RegisteredMovie()&movie&'motion_correction_method= "Suite2P"'
        moviemetadata = list(imaging.MovieMetaData()&movie)[0]
        channels = imaging.MovieChannel()&registered_movie
        registered_moviefiles = imaging.RegisteredMovieFile()&registered_movie
        reponame = list(imaging.RegisteredMovieFile()&registered_movie)[0]['reg_movie_file_repository']
        repodir = dj.config['locations.{}'.format(reponame)]
        movie_file_directory = list(registered_moviefiles)[0]['reg_movie_file_directory']
        roi_dir = os.path.join(repodir,movie_file_directory[:movie_file_directory.find('reg_tif')])
        try:
            #%
            iscell = np.load(os.path.join(roi_dir,'iscell.npy'))
            try:
                F =  np.load(os.path.join(roi_dir,'F_allow_orverlap.npy'))
                F_chan2 =  np.load(os.path.join(roi_dir,'F_chan2_allow_orverlap.npy'))
            except:
                print('could not load ROIs with overlaps, falling back to original: {}'.format(roi_dir))
                F =  np.load(os.path.join(roi_dir,'F.npy'))
                F_chan2 =  np.load(os.path.join(roi_dir,'F_chan2.npy'))
            Fneu =  np.load(os.path.join(roi_dir,'Fneu.npy'))
            Fneu_chan2 = np.load(os.path.join(roi_dir,'Fneu_chan2.npy'))
            roi_stat = list(np.load(os.path.join(roi_dir,'stat.npy'),allow_pickle=True))
            reg_metadata = np.load(os.path.join(roi_dir,'ops.npy'),allow_pickle = True).tolist()
            neuropil_masks = np.load(os.path.join(roi_dir,'neuropil_masks.npy'),allow_pickle = True).tolist()
            neuropil_mask_names = list(neuropil_masks.keys())
            neuropil_mask_inner_neuropil_radius = list()
            for neuropil_mask_name in neuropil_mask_names:
                if 'original' in neuropil_mask_name or 'neuropil_masks' not in neuropil_mask_name:
                    neuropil_mask_inner_neuropil_radius.append(reg_metadata['inner_neuropil_radius'])
                    continue
                #temp = 
                neuropil_mask_inner_neuropil_radius.append(list(map(int, re.findall(r'\d+', neuropil_mask_name) ))[0])
                varname_green = 'Fneu{}'.format(neuropil_mask_name[len('neuropil_masks'):])
                neuropil_masks[varname_green] = np.load(os.path.join(roi_dir,'{}.npy'.format(varname_green))) 
                varname_red = 'Fneu_chan2{}'.format(neuropil_mask_name[len('neuropil_masks'):])
                neuropil_masks[varname_red] = np.load(os.path.join(roi_dir,'{}.npy'.format(varname_red)))
            neuropil_masks['Fneu'] = Fneu
            neuropil_masks['Fneu_chan2'] = Fneu_chan2
            neuropil_mask_inner_neuropil_radius = np.sort(neuropil_mask_inner_neuropil_radius)
             #%
        except:
            print('missing or corrupt suite2p files, skipping {}'.format(movie))
            #continue
        cell_indices = np.where(iscell[:,0])[0]
        #%
# =============================================================================
#         cell_pix = create_cell_pix(roi_stat, Ly=reg_metadata['Ly'], Lx=reg_metadata['Lx'], allow_overlap=reg_metadata['allow_overlap'])
# =============================================================================
        for cell_index in cell_indices:   
            dj.conn().ping()
# =============================================================================
#             #%%
#             #TODO neuropil masks are already saved in an npy file -  just load it instead of calculating
#             neuropil_mask = create_neuropil_masks(ypixs=[roi_stat[cell_index]['ypix']],
#                                                   xpixs=[roi_stat[cell_index]['xpix']],
#                                                   cell_pix=cell_pix,
#                                                   inner_neuropil_radius=reg_metadata['inner_neuropil_radius'],
#                                                   min_neuropil_pixels=reg_metadata['min_neuropil_pixels'])
#             neuropil_mask = neuropil_mask.reshape(reg_metadata['Ly'],reg_metadata['Lx'])
#             neu_ypix,neu_xpix = np.where(neuropil_mask)
#             #%%
#             #TODO neuropil masks are already saved in an npy file -  just load it instead of calculating
# =============================================================================
            xpix = roi_stat[cell_index]['xpix']
            ypix = roi_stat[cell_index]['ypix']
            roi = list(registered_movie)[0].copy()
            roi['roi_type'] = 'Suite2P'
            roi['roi_number'] = cell_index
            roi_trace = roi.copy()
            roi_neuropil = roi.copy()
            roi_neuropil['neuropil_number'] = 0
            roi_neuropil_trace = roi_neuropil.copy()
            roi['roi_centroid_x'] = roi_stat[cell_index]['med'][1]
            roi['roi_centroid_y'] = roi_stat[cell_index]['med'][0]
            roi['roi_xpix'] = xpix
            roi['roi_ypix'] = ypix
            roi['roi_weights'] = roi_stat[cell_index]['lam']
            roi['roi_pixel_num'] = len(roi['roi_weights'])
            roi['roi_time_offset'] = moviemetadata['movie_hroimanager_lineperiod']*roi['roi_centroid_y']
            roi['roi_aspect_ratio'] = roi_stat[cell_index]['aspect_ratio']
            roi['roi_compact'] = roi_stat[cell_index]['compact']
            roi['roi_radius'] = roi_stat[cell_index]['radius']
            roi['roi_skew'] =  roi_stat[cell_index]['skew']
            roi_list.append(roi)
            
            for channel in channels:
                roi_trace['channel_number'] = channel['channel_number']
                if channel['channel_number'] == reg_metadata['functional_chan']:
                    roi_trace_green = roi_trace.copy()
                    roi_trace_green['roi_f'] = F[cell_index,:]
                    roi_trace_green['roi_f_mean'] = np.nanmean(F[cell_index,:])
                    roi_trace_green['roi_f_median'] = np.nanmedian(F[cell_index,:])
                    roi_trace_green['roi_f_min'] = np.nanmin(F[cell_index,:])
                    roi_trace_green['roi_f_max'] = np.nanmax(F[cell_index,:])
                    roi_trace_list.append(roi_trace_green)
                else:
                    try:
                        roi_trace_red = roi_trace.copy()
                        roi_trace_red['roi_f'] = F_chan2[cell_index,:]
                        roi_trace_red['roi_f_mean'] = np.nanmean(F_chan2[cell_index,:])
                        roi_trace_red['roi_f_median'] = np.nanmedian(F_chan2[cell_index,:])
                        roi_trace_red['roi_f_min'] = np.nanmin(F_chan2[cell_index,:])
                        roi_trace_red['roi_f_max'] = np.nanmax(F_chan2[cell_index,:])
                        roi_trace_list.append(roi_trace_red)
                    except: #TODO  fix this suite2p bug on the upload
                        print('red ROI not found for  {} - merging error??'.format(movie))
                        #%
                        #neuropil_mask = neuropil_mask.reshape(reg_metadata['Ly'],reg_metadata['Lx'])
                        #%
                        nframes = int(reg_metadata['nframes'])
                        nimgbatch = min(reg_metadata['batch_size'], 1000)
                        nframes = int(reg_metadata['nframes'])
                        Ly = reg_metadata['Ly']
                        Lx = reg_metadata['Lx']
                        cell_mask = np.zeros([Ly,Lx],dtype='float32')
                        
                        cell_mask[roi_stat[cell_index]['ypix'],roi_stat[cell_index]['xpix']]=roi_stat[cell_index]['lam']/sum(roi_stat[cell_index]['lam'])
                        cell_mask = cell_mask.reshape(1,np.prod(cell_mask.shape))
                        ncells = cell_mask.shape[0]
                        #neu_ypix,neu_xpix = np.where(neuropil_mask)
                        #%
                        
                        if channel['channel_number'] ==1:
                            reg_file = open(os.path.join(roi_dir,'data.bin'), 'rb')
                        elif channel['channel_number'] ==2:
                            reg_file = open(os.path.join(roi_dir,'data_chan2.bin'), 'rb')
                        
                        F_now = np.zeros((ncells, nframes),np.float32)
                        nimgbatch = int(nimgbatch)
                        block_size = Ly*Lx*nimgbatch*2
                        ix = 0
                        data = 1
                    
                        while data is not None:
                            buff = reg_file.read(block_size)
                            data = np.frombuffer(buff, dtype=np.int16, offset=0)
                            nimg = int(np.floor(data.size / (Ly*Lx)))
                            if nimg == 0:
                                break
                            data = np.reshape(data, (-1, Ly, Lx))
                            inds = ix+np.arange(0,nimg,1,int)
                            data = np.reshape(data, (nimg,-1))
                            F_now[:,inds] = np.dot(cell_mask , data.T)
                            #F_now[0,inds] = np.dot(data[:, cell_masks[n][0]], cell_masks[n][1])
                            ix += nimg
                        reg_file.close()
                        roi_trace_red = roi_trace.copy()
                        roi_trace_red['roi_f'] = F_now[0]
                        roi_trace_red['roi_f_mean'] = np.nanmean(F_now[0])
                        roi_trace_red['roi_f_median'] = np.nanmedian(F_now[0])
                        roi_trace_red['roi_f_min'] = np.nanmin(F_now[0])
                        roi_trace_red['roi_f_max'] = np.nanmax(F_now[0])
                        roi_trace_list.append(roi_trace_red)
                        
                        
                        #%
                            
            
            for neuropil_number,inner_neuropil_radius in enumerate(neuropil_mask_inner_neuropil_radius): # there are different neuropil masks
                roi_neuropil_now = roi_neuropil.copy()
                roi_neuropil_now['neuropil_number'] = neuropil_number
                if inner_neuropil_radius == reg_metadata['inner_neuropil_radius']:
                    neuropil_mask_varname = 'neuropil_masks_original'
                else:
                    neuropil_mask_varname = 'neuropil_masks_min_neuropil_pixels_{}'.format(inner_neuropil_radius)
                roi_neuropil_now['neuropil_xpix'] = neuropil_masks[neuropil_mask_varname][cell_index][1]
                roi_neuropil_now['neuropil_ypix'] = neuropil_masks[neuropil_mask_varname][cell_index][0]
                roi_neuropil_now['neuropil_pixel_num'] =  len(roi_neuropil_now['neuropil_xpix'])
                if len(roi_neuropil_now['neuropil_xpix'])<5:# too few pixels, no neuropil there
                    continue
                roi_neuropil_list.append(roi_neuropil_now)
                for channel in channels:
                    roi_neuropil_trace['channel_number'] = channel['channel_number']
                    roi_neuropil_trace['neuropil_number'] = neuropil_number
                    if channel['channel_number'] == reg_metadata['functional_chan']:
                        roi_neuropil_trace_green = roi_neuropil_trace.copy()
                        if inner_neuropil_radius == reg_metadata['inner_neuropil_radius']:
                            roi_neuropil_trace_green['neuropil_f'] = Fneu[cell_index,:]
                        else:
                            roi_neuropil_trace_green['neuropil_f'] = neuropil_masks['Fneu_min_neuropil_pixels_{}'.format(inner_neuropil_radius)][cell_index,:]
                        roi_neuropil_trace_green['neuropil_f_mean'] =np.nanmean(roi_neuropil_trace_green['neuropil_f'])
                        roi_neuropil_trace_green['neuropil_f_median']=np.nanmedian(roi_neuropil_trace_green['neuropil_f'])
                        roi_neuropil_trace_green['neuropil_f_min']=np.nanmin(roi_neuropil_trace_green['neuropil_f'])
                        roi_neuropil_trace_green['neuropil_f_max']=np.nanmax(roi_neuropil_trace_green['neuropil_f'])
                        if roi_neuropil_trace_green['neuropil_f'][0]<1:
                            pass
                            #print('neuropil was not extracted for {}, skipping'.format(roi_neuropil_trace_green))
                        else:
                            roi_neuropil_trace_list.append(roi_neuropil_trace_green)
                    else:
                        try:
                            roi_neuropil_trace_red = roi_neuropil_trace.copy()
                            
                            if inner_neuropil_radius == reg_metadata['inner_neuropil_radius']:
                                roi_neuropil_trace_red['neuropil_f'] = Fneu_chan2[cell_index,:]
                            else:
                                roi_neuropil_trace_red['neuropil_f'] = neuropil_masks['Fneu_chan2_min_neuropil_pixels_{}'.format(inner_neuropil_radius)][cell_index,:]
                            roi_neuropil_trace_red['neuropil_f_mean'] =np.nanmean(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_median']=np.nanmedian(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_min']=np.nanmin(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_max']=np.nanmax(roi_neuropil_trace_red['neuropil_f'])
                            if roi_neuropil_trace_red['neuropil_f'][0]<1:
                                pass
                                #print('neuropil was not extracted for {}, skipping'.format(roi_neuropil_trace_red))
                            else:
                                roi_neuropil_trace_list.append(roi_neuropil_trace_red)
                        except: # fix this suite2p bug on the upload
                            #%
                            print('red ROI not found for  {} - merging error - extracting neuropil again'.format(movie))
                            ncells = 1
                            nframes = int(reg_metadata['nframes'])
                            nimgbatch = min(reg_metadata['batch_size'], 1000)
                            nframes = int(reg_metadata['nframes'])
                            Ly = reg_metadata['Ly']
                            Lx = reg_metadata['Lx']
                            cell_pix = create_cell_pix(roi_stat, Ly=reg_metadata['Ly'], Lx=reg_metadata['Lx'], allow_overlap=reg_metadata['allow_overlap'])
                            neuropil_mask = create_neuropil_masks(ypixs=[roi_stat[cell_index]['ypix']],
                                                                  xpixs=[roi_stat[cell_index]['xpix']],
                                                                  cell_pix=cell_pix,
                                                                  inner_neuropil_radius=reg_metadata['inner_neuropil_radius'],
                                                                  min_neuropil_pixels=reg_metadata['min_neuropil_pixels'])
                            #neuropil_mask = neuropil_mask.reshape(reg_metadata['Ly'],reg_metadata['Lx'])
                            
                            #neu_ypix,neu_xpix = np.where(neuropil_mask)
                            
                            
                            if channel['channel_number'] ==1:
                                reg_file = open(os.path.join(roi_dir,'data.bin'), 'rb')
                            elif channel['channel_number'] ==2:
                                reg_file = open(os.path.join(roi_dir,'data_chan2.bin'), 'rb')
                            
                            Fneu_now = np.zeros((ncells, nframes),np.float32)
                            nimgbatch = int(nimgbatch)
                            block_size = Ly*Lx*nimgbatch*2
                            ix = 0
                            data = 1
                        
                            while data is not None:
                                buff = reg_file.read(block_size)
                                data = np.frombuffer(buff, dtype=np.int16, offset=0)
                                nimg = int(np.floor(data.size / (Ly*Lx)))
                                if nimg == 0:
                                    break
                                data = np.reshape(data, (-1, Ly, Lx))
                                inds = ix+np.arange(0,nimg,1,int)
                                data = np.reshape(data, (nimg,-1))
                                Fneu_now[:,inds] = np.dot(neuropil_mask , data.T)
                                ix += nimg
                            reg_file.close()
                            roi_neuropil_trace_red['neuropil_f'] = Fneu_now[0]
                            roi_neuropil_trace_red['neuropil_f_mean'] =np.nanmean(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_median']=np.nanmedian(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_min']=np.nanmin(roi_neuropil_trace_red['neuropil_f'])
                            roi_neuropil_trace_red['neuropil_f_max']=np.nanmax(roi_neuropil_trace_red['neuropil_f'])
                            if roi_neuropil_trace_red['neuropil_f'][0]<1:
                                pass
                                #print('neuropil was not extracted for {}, skipping'.format(roi_neuropil_trace_red))
                            else:
                                roi_neuropil_trace_list.append(roi_neuropil_trace_red)
                            #%%
                            
                            
                        #%%
# =============================================================================
#          #%%
#         img = np.zeros([reg_metadata['Ly'],reg_metadata['Lx']])
#         img[ypix,xpix]=1
#         img[neu_ypix,neu_xpix]=2
#         plt.imshow(img)
# =============================================================================
        #%
    with dj.conn().transaction: #inserting one movie
        print('uploading ROIs to datajoint -  subject-movie: {}-{}'.format(movie['subject_id'],'all movies')) #movie['movie_name']
        imaging.ROI().insert(roi_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROINeuropil().insert(roi_neuropil_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROITrace().insert(roi_trace_list, allow_direct_insert=True)
        dj.conn().ping()
        imaging.ROINeuropilTrace().insert(roi_neuropil_trace_list, allow_direct_insert=True)
