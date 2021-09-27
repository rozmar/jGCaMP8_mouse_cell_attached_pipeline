import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from pathlib import Path
import json
from utils import utils_plot, utils_ephys
bad_cell_blacklist = ['anm472182_cell3',
                      'anm479120_cell5',
                      'anm479571_cell3',
                      'anm479571_cell1',
                      'anm478348_cell6',
                      'anm478348_cell5',
                      'anm478342_cell4',#low snr
                      'anm478410_cell4',#low baseline f0
                      'anm478347_cell4',
                      'anm478406_cell3']
bad_movie_blacklist = ['anm479572_cell4_movie15', #wrong ROI
                       'anm479119_cell6_movie38',
                       'anm478342_cell7_movie34']
# =============================================================================
# #%%
# for downsampling_factor in [1,2,3,4,5,6,7,8]:
#     downsample_movies_and_ROIs(downsampling_factor,overwrite = True,only_metadata = False)
# =============================================================================
#%% downsample movies and ROIs for 
def downsample_movies_and_ROIs(downsampling_factor = 2,overwrite = False,only_metadata = False):
    #%%
# =============================================================================
#     downsampling_factor = 1
#     overwrite = True
#     only_metadata = False
# =============================================================================
    
    sensors_to_export = ['456','686','688','XCaMPgf','GCaMP7F']#
    sensor_names = {'686':'jGCaMP8m',
                    '688':'jGCaMP8s',
                    '456':'jGCaMP8f',
                    'XCaMPgf':'XCaMPgf',
                    'GCaMP7F':'jGCaMP7f'}
    #%
    savedir = '/home/rozmar/Mount/HDD_RAID_2_16TB/GCaMP8_downsampled/{}x_downsampled/'.format(downsampling_factor)
    roi_crit = {'channel_number':1,
                'motion_correction_method':'Suite2P',
                'roi_type':'Suite2P',
                'neuropil_number':1}
    max_sweep_ap_hw_std_per_median =  .2
    min_sweep_ap_snr_dv_median = 10
    #minimum_f0_value = 20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
    min_channel_offset = 100 # PMT error
    ephys_quality_crit_1 = 'sweep_ap_snr_dv_median>={}'.format(min_sweep_ap_snr_dv_median)
    ephys_quality_crit_2 = 'sweep_ap_hw_std_per_median<={}'.format(max_sweep_ap_hw_std_per_median)
    #baseline_f_crit = 'cawave_baseline_f > {}'.format(minimum_f0_value)
    channel_offset_crit = 'channel_offset > {}'.format(min_channel_offset)
    
    calcium_sensors = np.unique(imaging_gt.SessionCalciumSensor().fetch('session_calcium_sensor'))
    max_cells_to_export = 200 # per sensor
    max_movies_to_export = 2000 #per sensor
    #%
    for sensor in calcium_sensors:
        if sensor not in sensors_to_export:
            continue
        
        sensor_crit = 'session_calcium_sensor="{}"'.format(sensor)
        movies_all = (imaging.MovieChannel()*ephysanal_cell_attached.SweepAPQC()*imaging_gt.SessionCalciumSensor()*imaging_gt.MovieCalciumWaveSNR() & sensor_crit&ephys_quality_crit_1&ephys_quality_crit_2&channel_offset_crit)
        subject_ids = np.unique(movies_all.fetch('subject_id'))

    #%
        median_cell_snr_list = list()
        subject_id_list = list()
        cell_number_list = list()
        movie_number_list = list()
        cell_total_movie_len_list = list()
        cell_total_ap_num_list = list()
        for subject_id in subject_ids:
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_numbers = np.unique((movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit).fetch('cell_number'))
            for cell_number in cell_numbers:
                cell_crit = 'cell_number = {}'.format(cell_number)
                cell_snrs,cell_movie_total_ap_num,sweep_start_time,sweep_end_time = (movies_all*imaging_gt.ROISweepCorrespondance()*ephys_cell_attached.Sweep()&subject_crit&cell_crit).fetch('movie_median_cawave_snr_per_ap','movie_total_ap_num','sweep_start_time','sweep_end_time')
                cell_movie_len = np.asarray(sweep_end_time-sweep_start_time,float)
                #cell_firing_rate = cell_movie_total_ap_num/cell_movie_len
                median_cell_snr_list.append(np.median(cell_snrs))
                movie_number_list.append(len(cell_snrs))
                subject_id_list.append(np.median(subject_id))
                cell_number_list.append(np.median(cell_number))
                cell_total_movie_len_list.append(np.sum(cell_movie_len))
                cell_total_ap_num_list.append(np.sum(cell_movie_total_ap_num))
         #%
        cell_number_list=np.asarray(cell_number_list)
        subject_id_list=np.asarray(subject_id_list)
        median_cell_snr_list=np.asarray(median_cell_snr_list)
        movie_number_list=np.asarray(movie_number_list)
        cell_total_movie_len_list=np.asarray(cell_total_movie_len_list)
        cell_total_ap_num_list=np.asarray(cell_total_ap_num_list)
        
        order = np.argsort(median_cell_snr_list)
        
        
        median_cell_snr_list = median_cell_snr_list[order][::-1]
        subject_id_list = subject_id_list[order][::-1]
        cell_number_list = cell_number_list[order][::-1]
        movie_number_list = movie_number_list[order][::-1]
        cell_total_movie_len_list = cell_total_movie_len_list[order][::-1]
        cell_total_ap_num_list = cell_total_ap_num_list[order][::-1]
        
        cell_counter = 0
        movie_counter = 0
        #%
        for subject_id,cell_number,median_cell_snr,total_movie_len,total_ap_num in zip(subject_id_list,cell_number_list,median_cell_snr_list,cell_total_movie_len_list,cell_total_ap_num_list):
            if cell_counter>max_cells_to_export or movie_counter > max_movies_to_export:
                break
            cell_counter+=1
            #%
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_crit = 'cell_number = {}'.format(cell_number)
            celltype = (ephysanal_cell_attached.EphysCellType()& subject_crit & cell_crit).fetch1('ephys_cell_type')
            baseline_f0_value = (imaging_gt.CellBaselineFluorescence()& subject_crit & cell_crit).fetch1('cell_baseline_roi_f0_median')
            #%
            cell_movies = movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit & cell_crit# &'sweep_number = 2'
            snrs,movie_numbers = cell_movies.fetch('movie_median_cawave_snr_per_ap','movie_number')
            order = np.argsort(snrs)[::-1]
            snrs = snrs[order]
            movie_numbers = movie_numbers[order]
            
            cell_metadata_dict = {'median_cell_snr':median_cell_snr,
                                  'putative_celltype':celltype,
                                  'total_ap_num':total_ap_num,
                                  'total_movie_lenght':total_movie_len,
                                  'movie_number':len(movie_numbers),
                                  'baseline_F0':baseline_f0_value
                                  }
            savedir_cell_movie = Path(os.path.join(savedir,'movies',sensor_names[sensor],'anm{}_cell_{}'.format(int(subject_id),int(cell_number))))
            savedir_cell_metadata = Path(os.path.join(savedir,'metadata',sensor_names[sensor],'anm{}_cell_{}'.format(int(subject_id),int(cell_number))))
            savedir_cell_metadata.mkdir(parents=True, exist_ok=True)
            with open(savedir_cell_metadata/'cell_metadata.json', 'w') as outfile:
                json.dump(cell_metadata_dict, outfile, indent=4)
            #%
            for movie_number, snr in zip(movie_numbers,snrs):
                metadata_files_alread_saved = os.listdir(savedir_cell_metadata)
                dj.conn().connect()
                savedir_movie = savedir_cell_movie/'movie_{}'.format(int(movie_number))
                savedir_movie.mkdir(parents=True, exist_ok=True)
                #break
            #%
                try:
                    metadata_filename = 'movie_{}.json'.format(int(movie_number))
                    if metadata_filename in metadata_files_alread_saved and not overwrite:
                        print('{} already saved'.format(metadata_filename))
                        continue
                    else:
                        print('exporting to {}'.format(savedir_movie))
                        
                        
                        
                        
                     #%
                    movie_crit = 'movie_number = {}'.format(movie_number)
                    big_table = imaging.RegisteredMovieImage()*imaging.ROINeuropil()*imaging.Movie()*ephys_cell_attached.SweepMetadata()*imaging.ROI()*imaging.ROITrace()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()*cell_movies&movie_crit&roi_crit#*ephysanal_cell_attached.ActionPotential()
                    #%
                    roi_f,neuropil_f,roi_number,roi_centroid_x, roi_centroid_y, frame_times,roi_time_offset,movie_x_size,movie_y_size,movie_frame_num,movie_name,movie_pixel_size,roi_xpix,roi_ypix,roi_weights,neuropil_xpix,neuropil_ypix,registered_movie_mean_image = big_table.fetch1('roi_f','neuropil_f','roi_number','roi_centroid_x','roi_centroid_y','frame_times','roi_time_offset','movie_x_size','movie_y_size','movie_frame_num','movie_name','movie_pixel_size','roi_xpix','roi_ypix','roi_weights','neuropil_xpix','neuropil_ypix','registered_movie_mean_image') #ap_times
                    ap_max_times = (ephysanal_cell_attached.ActionPotential()*cell_movies&movie_crit&roi_crit).fetch('ap_max_time')
                    cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()*cell_movies&movie_crit).fetch1('cell_recording_start', 'session_time')
                    
                    sweep_start_time =  (ephys_cell_attached.Sweep()*cell_movies&movie_crit).fetch1('sweep_start_time')
                    sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()*cell_movies&movie_crit).fetch1('sample_rate')
    
                    frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))
                    #%
                    movie_files = imaging.RegisteredMovieFile()*cell_movies&movie_crit
                    reg_movie_file_repository,reg_movie_file_directory,reg_movie_file_name = movie_files.fetch('reg_movie_file_repository','reg_movie_file_directory','reg_movie_file_name')
                    
                    
                    source_dir = os.path.join(dj.config['locations.{}'.format(reg_movie_file_repository[0])],reg_movie_file_directory[0])
                    
                    
                    source_dir = source_dir[:source_dir.find('/reg_tif')]
    # =============================================================================
    #                 print('waiting')
    #                 time.sleep()
    # =============================================================================
                    #%
                    
                    Ly = int(movie_y_size)
                    Lx = int(movie_x_size)
                    downsampled_size_y = int(len(np.arange(0,Ly,downsampling_factor)))
                    downsampled_size_x = int(len(np.arange(0,Lx,downsampling_factor)))
                    data_downsampled = np.zeros((int(movie_frame_num),downsampled_size_y, downsampled_size_x),np.float)
                    #%
                    ROI_mask = np.zeros([Ly,Lx])
                    ROI_mask[roi_ypix,roi_xpix] = roi_weights/sum(roi_weights)
                    ROI_mask_down = ROI_mask[::downsampling_factor,::downsampling_factor]
                    ROI_mask_down = ROI_mask_down/sum(ROI_mask_down.flatten())
    # =============================================================================
    #                 #%%
    #                 needed = ~metadata[3]['overlap']
    #                 ROI_mask_nooverlap = np.zeros([Ly,Lx])
    #                 ROI_mask_nooverlap[roi_ypix[needed],roi_xpix[needed]] = roi_weights[needed]/sum(roi_weights[needed])
    #                 ROI_mask_down_nooverlap = ROI_mask_nooverlap[::downsampling_factor,::downsampling_factor]
    #                 ROI_mask_down_nooverlap = ROI_mask_down_nooverlap/sum(ROI_mask_down_nooverlap.flatten())
    # =============================================================================
                    #%
                    neuropil_mask = np.zeros([Ly,Lx])
                    neuropil_mask[neuropil_ypix,neuropil_xpix] = 1/len(neuropil_xpix)
                    neuropil_mask_down = neuropil_mask[::downsampling_factor,::downsampling_factor]
                    neuropil_mask_down = neuropil_mask_down/sum(neuropil_mask_down.flatten())
                    
                    
                    movie_metadata_dict = {'ephys_sample_rate':int(sample_rate),
                                           'movie_x_size':int(movie_x_size),
                                           'movie_y_size':int(movie_y_size),
                                           'movie_x_size_downsampled':int(downsampled_size_x),
                                           'movie_y_size_downsampled':int(downsampled_size_y),
                                           'movie_pixel_size':float(movie_pixel_size),
                                           'movie_pixel_size_downsampled':float(movie_pixel_size*downsampling_factor),
                                           'movie_frame_num':int(movie_frame_num),
                                           'movie_name':movie_name,
                                           'roi_time_offset':round(roi_time_offset,4),
                                           'movie_1ap_snr':round(snr,1),
                                           'downsampling_factor':downsampling_factor
                                           }
                    with open(savedir_cell_metadata/'movie_{}.json'.format(movie_number), 'w') as outfile:
                        json.dump(movie_metadata_dict, outfile, indent=4)
                    
                    if only_metadata:
                        continue
                    
                    reg_file = open(os.path.join(source_dir,'data.bin'), 'rb')
                    nimgbatch = 200
                    block_size = Ly*Lx*nimgbatch*2
                    ix = 0
                    data = 1
                    #%
                    while data is not None:
                        #%
                        buff = reg_file.read(block_size)
                        data = np.frombuffer(buff, dtype=np.int16, offset=0)
                        nimg = int(np.floor(data.size / (Ly*Lx)))
                        #%
                        if nimg == 0:
                            break
                        #%
                        data = np.reshape(data, (-1, Ly, Lx))
                        inds = ix+np.arange(0,nimg,1,int)                
                        data_down_now = data[:,::downsampling_factor,::downsampling_factor]
                        data_downsampled[inds,:,:] = data_down_now
                        
                        #data = np.reshape(data, (nimg,-1))
                        #F_down[inds] = np.dot(data_down_now,ROI_down.T)
                        ix += nimg
                    reg_file.close()
                    F_down = np.sum(data_downsampled*ROI_mask_down,(1,2))
                    Fneu_down = np.sum(data_downsampled*neuropil_mask_down,(1,2))
    
                    
                    np.savez_compressed(os.path.join(savedir_cell_metadata,'movie_{}_roi_{}.npz'.format(movie_number,roi_number)),
                                        F_original=roi_f,
                                        Fneu_original = neuropil_f,
                                        ROI_mask_original = ROI_mask,
                                        neuropil_mask_original = neuropil_mask, 
                                        mean_image_original = registered_movie_mean_image,
                                        mean_image_downsampled = np.mean(data_downsampled,0),
                                        F_downsampled=F_down,
                                        Fneu_downsampled = Fneu_down,
                                        ROI_mask_downsampled = ROI_mask_down,
                                        neuropil_mask_downsampled = neuropil_mask_down,
                                        framerate = np.median(np.diff(frame_times)),
                                        ap_max_times=np.asarray(ap_max_times,float),
                                        frame_start_times = np.asarray(frame_times,float),
                                        roi_time_offset = roi_time_offset,                                    
                                        )
                        
                                            
                                            
                        
                        
# =============================================================================
#                     print('break now')
#                     time.sleep(1000)
# =============================================================================
                            #%
                except:
                    print('error with this record')
                    cell_movies&movie_crit&roi_crit            



#%%

def export_movies_with_metadata():
    #%% export traces for s2f
    # this script locates the cells with highest snr for each sensor, then exports the first n cells or m movies, whichever comes first
    lfp_neuropil_xcorr_max_threshold = 2500 #
    overwrite = True
    sensors_to_export = ['686','688']
    sensor_names = {'686':'jGCaMP8m',
                    '688':'jGCaMP8s'}
    #%
    savedir = '/home/rozmar/Mount/HDD_RAID_2_16TB/GCaMP8_deepinterpolation_export'
    roi_crit = {'channel_number':1,
                'motion_correction_method':'Suite2P',
                'roi_type':'Suite2P',
                'neuropil_number':1}
    max_sweep_ap_hw_std_per_median =  .2
    min_sweep_ap_snr_dv_median = 10
    #minimum_f0_value = 20 # when the neuropil is bright, the F0 can go negative after neuropil subtraction
    min_channel_offset = 100 # PMT error
    ephys_quality_crit_1 = 'sweep_ap_snr_dv_median>={}'.format(min_sweep_ap_snr_dv_median)
    ephys_quality_crit_2 = 'sweep_ap_hw_std_per_median<={}'.format(max_sweep_ap_hw_std_per_median)
    #baseline_f_crit = 'cawave_baseline_f > {}'.format(minimum_f0_value)
    channel_offset_crit = 'channel_offset > {}'.format(min_channel_offset)
    
    calcium_sensors = np.unique(imaging_gt.SessionCalciumSensor().fetch('session_calcium_sensor'))
    max_cells_to_export = 200 # per sensor
    max_movies_to_export = 2000 #per sensor
    #%
    for sensor in calcium_sensors:
        if sensor not in sensors_to_export:
            continue
        
        sensor_crit = 'session_calcium_sensor="{}"'.format(sensor)
        movies_all = (imaging.MovieChannel()*ephysanal_cell_attached.SweepAPQC()*imaging_gt.SessionCalciumSensor()*imaging_gt.MovieCalciumWaveSNR() & sensor_crit&ephys_quality_crit_1&ephys_quality_crit_2&channel_offset_crit)
        subject_ids = np.unique(movies_all.fetch('subject_id'))

    #%
        median_cell_snr_list = list()
        subject_id_list = list()
        cell_number_list = list()
        movie_number_list = list()
        cell_total_movie_len_list = list()
        cell_total_ap_num_list = list()
        for subject_id in subject_ids:
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_numbers = np.unique((movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit).fetch('cell_number'))
            for cell_number in cell_numbers:
                cell_crit = 'cell_number = {}'.format(cell_number)
                cell_snrs,cell_movie_total_ap_num,sweep_start_time,sweep_end_time = (movies_all*imaging_gt.ROISweepCorrespondance()*ephys_cell_attached.Sweep()&subject_crit&cell_crit).fetch('movie_median_cawave_snr_per_ap','movie_total_ap_num','sweep_start_time','sweep_end_time')
                cell_movie_len = np.asarray(sweep_end_time-sweep_start_time,float)
                #cell_firing_rate = cell_movie_total_ap_num/cell_movie_len
                median_cell_snr_list.append(np.median(cell_snrs))
                movie_number_list.append(len(cell_snrs))
                subject_id_list.append(np.median(subject_id))
                cell_number_list.append(np.median(cell_number))
                cell_total_movie_len_list.append(np.sum(cell_movie_len))
                cell_total_ap_num_list.append(np.sum(cell_movie_total_ap_num))
         #%
        cell_number_list=np.asarray(cell_number_list)
        subject_id_list=np.asarray(subject_id_list)
        median_cell_snr_list=np.asarray(median_cell_snr_list)
        movie_number_list=np.asarray(movie_number_list)
        cell_total_movie_len_list=np.asarray(cell_total_movie_len_list)
        cell_total_ap_num_list=np.asarray(cell_total_ap_num_list)
        
        order = np.argsort(median_cell_snr_list)
        
        
        median_cell_snr_list = median_cell_snr_list[order][::-1]
        subject_id_list = subject_id_list[order][::-1]
        cell_number_list = cell_number_list[order][::-1]
        movie_number_list = movie_number_list[order][::-1]
        cell_total_movie_len_list = cell_total_movie_len_list[order][::-1]
        cell_total_ap_num_list = cell_total_ap_num_list[order][::-1]
        
        cell_counter = 0
        movie_counter = 0
        #%
        for subject_id,cell_number,median_cell_snr,total_movie_len,total_ap_num in zip(subject_id_list,cell_number_list,median_cell_snr_list,cell_total_movie_len_list,cell_total_ap_num_list):
            if cell_counter>max_cells_to_export or movie_counter > max_movies_to_export:
                break
            cell_counter+=1
            #%
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_crit = 'cell_number = {}'.format(cell_number)
            celltype = (ephysanal_cell_attached.EphysCellType()& subject_crit & cell_crit).fetch1('ephys_cell_type')
            
            #%
            cell_movies = movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit & cell_crit# &'sweep_number = 2'
            snrs,movie_numbers = cell_movies.fetch('movie_median_cawave_snr_per_ap','movie_number')
            order = np.argsort(snrs)[::-1]
            snrs = snrs[order]
            movie_numbers = movie_numbers[order]
            
            cell_metadata_dict = {'median_cell_snr':median_cell_snr,
                                  'putative_celltype':celltype,
                                  'total_ap_num':total_ap_num,
                                  'total_movie_lenght':total_movie_len,
                                  'movie_number':len(movie_numbers)
                                  }
            #%
            savedir_cell_movie = Path(os.path.join(savedir,'movies',sensor_names[sensor],'anm{}_cell_{}'.format(int(subject_id),int(cell_number))))
            savedir_cell_metadata = Path(os.path.join(savedir,'metadata',sensor_names[sensor],'anm{}_cell_{}'.format(int(subject_id),int(cell_number))))
            savedir_cell_ephys = Path(os.path.join(savedir,'ephys',sensor_names[sensor],'anm{}_cell_{}'.format(int(subject_id),int(cell_number))))
            savedir_cell_metadata.mkdir(parents=True, exist_ok=True)
            with open(savedir_cell_metadata/'cell_metadata.json', 'w') as outfile:
                json.dump(cell_metadata_dict, outfile, indent=4)
            #%
            for movie_number, snr in zip(movie_numbers,snrs):
                metadata_files_alread_saved = os.listdir(savedir_cell_metadata)
                dj.conn().connect()
                savedir_movie = savedir_cell_movie/'movie_{}'.format(int(movie_number))
                savedir_movie.mkdir(parents=True, exist_ok=True)
                #break
            #%
                try:
                    metadata_filename = 'movie_{}.json'.format(int(movie_number))
                    if metadata_filename in metadata_files_alread_saved and not overwrite:
                        print('{} already saved'.format(metadata_filename))
                        continue
                    else:
                        print('exporting to {}'.format(savedir_movie))
                        
                        
                    
                        
                     #%
                    movie_crit = 'movie_number = {}'.format(movie_number)
                    big_table = imaging.Movie()*ephys_cell_attached.SweepMetadata()*imaging.ROI()*imaging.ROITrace()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()*cell_movies&movie_crit&roi_crit#*ephysanal_cell_attached.ActionPotential()
                    #%
                    roi_centroid_x, roi_centroid_y, frame_times,roi_time_offset,movie_x_size,movie_y_size,movie_frame_num,movie_name,movie_pixel_size = big_table.fetch1('roi_centroid_x','roi_centroid_y','frame_times','roi_time_offset','movie_x_size','movie_y_size','movie_frame_num','movie_name','movie_pixel_size') #ap_times
                    ap_max_times = (ephysanal_cell_attached.ActionPotential()*cell_movies&movie_crit&roi_crit).fetch('ap_max_time')
                    cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()*cell_movies&movie_crit).fetch1('cell_recording_start', 'session_time')
                    
                    sweep_start_time =  (ephys_cell_attached.Sweep()*cell_movies&movie_crit).fetch1('sweep_start_time')
                    sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()*cell_movies&movie_crit).fetch1('sample_rate')
    
                    frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))
                    #%
                    movie_files = imaging.RegisteredMovieFile()*cell_movies&movie_crit
                    reg_movie_file_repository,reg_movie_file_directory,reg_movie_file_name = movie_files.fetch('reg_movie_file_repository','reg_movie_file_directory','reg_movie_file_name')
                    for reg_movie_file_repository_now,reg_movie_file_directory_now,reg_movie_file_name_now in zip(reg_movie_file_repository,reg_movie_file_directory,reg_movie_file_name):
                        source_dir = os.path.join(dj.config['locations.{}'.format(reg_movie_file_repository_now)],reg_movie_file_directory_now)
                        destination_dir = savedir_movie
                        os.system('rsync -rv --size-only '+ '\'' + str(source_dir) +'/'+ '\'' +' '+ '\''+str(destination_dir)+ '\''+'/')
                       
                        break
                        source_file = os.path.join(dj.config['locations.{}'.format(reg_movie_file_repository_now)],reg_movie_file_directory_now,reg_movie_file_name_now)
                        destination_file = os.path.join(savedir_movie,reg_movie_file_name_now)
                        shutil.copyfile(source_file, destination_file)
                        
                    #%
                    movie_metadata_dict = {'roi_centroid_x':float(roi_centroid_x),
                                           'roi_centroid_y':float(roi_centroid_y),
                                           'ephys_sample_rate':int(sample_rate),
                                           'movie_x_size':int(movie_x_size),
                                           'movie_y_size':int(movie_y_size),
                                           'movie_pixel_size':float(movie_pixel_size),
                                           'movie_frame_num':int(movie_frame_num),
                                           'movie_name':movie_name,
                                           'roi_time_offset':round(roi_time_offset,4),
                                           'movie_1ap_snr':round(snr,1),
                                           'ap_max_times':list(np.asarray(ap_max_times,float)),
                                           'frame_start_times':list(np.round(np.asarray(frame_times,float),4)),
                                           }
                    
                    with open(savedir_cell_metadata/'movie_{}.json'.format(movie_number), 'w') as outfile:
                        json.dump(movie_metadata_dict, outfile, indent=4)
    # =============================================================================
    #                 print('waiting')
    #                 time.sleep(1000)    
    # =============================================================================
                    if len(imaging_gt.LFPNeuropilCorrelation()*cell_movies&movie_crit&roi_crit)>0:
                        lfp_neuropil_xcorr_max = (imaging_gt.LFPNeuropilCorrelation()*cell_movies&movie_crit&roi_crit).fetch1('lfp_neuropil_xcorr_max')
                        if lfp_neuropil_xcorr_max>=lfp_neuropil_xcorr_max_threshold:
                            savedir_cell_ephys.mkdir(parents=True, exist_ok=True)
                            response_trace = (ephys_cell_attached.SweepResponse()*cell_movies&movie_crit&roi_crit).fetch1('response_trace')
                            np.save(os.path.join(savedir_cell_ephys,'movie_{}_ephys.npy'.format(movie_number)),response_trace)
                            
                        
                    #%
                except:
                    print('error with this record')
                    cell_movies&movie_crit&roi_crit
   
#%%




def export_s2f_data_from_datajoint(): # for Ziquiang
    #%% export traces for s2f
    # this script locates the cells with highest snr for each sensor, then exports the first n cells or m movies, whichever comes first
    fig = plt.figure(figsize = [15,5])
    ax_ophys=fig.add_subplot(2,1,1)
    ax_ephys=fig.add_subplot(2,1,2,sharex = ax_ophys)
    #%
    savedir = '/home/rozmar/Data/Calcium_imaging/GCaMP8_exported_ROIs_s2f'
    files_alread_saved = os.listdir(savedir)
    #savedir = '/home/rozmar/Network/rozsam_google_drive/Data_to_share/GCaMP8_exported_ROIs_s2f'
    roi_crit = {'channel_number':1,
                'motion_correction_method':'Suite2P',
                'roi_type':'Suite2P',
                'neuropil_number':1}
    max_sweep_ap_hw_std_per_median =  .2
    min_sweep_ap_snr_dv_median = 10
    min_channel_offset = 100 # PMT error
    ephys_quality_crit_1 = 'sweep_ap_snr_dv_median>={}'.format(min_sweep_ap_snr_dv_median)
    ephys_quality_crit_2 = 'sweep_ap_hw_std_per_median<={}'.format(max_sweep_ap_hw_std_per_median)
    channel_offset_crit = 'channel_offset > {}'.format(min_channel_offset)
    calcium_sensors = np.unique(imaging_gt.SessionCalciumSensor().fetch('session_calcium_sensor'))
    max_cells_to_export = 200 # per sensor
    max_movies_to_export = 2000 #per sensor
    for sensor in calcium_sensors:
        sensor_crit = 'session_calcium_sensor="{}"'.format(sensor)
        movies_all = (imaging.MovieChannel()*ephysanal_cell_attached.SweepAPQC()*imaging_gt.SessionCalciumSensor()*imaging_gt.MovieCalciumWaveSNR() & sensor_crit&ephys_quality_crit_1&ephys_quality_crit_2&channel_offset_crit)
        subject_ids = np.unique(movies_all.fetch('subject_id'))
        median_cell_snr_list = list()
        subject_id_list = list()
        cell_number_list = list()
        movie_number_list = list()
        cell_types_list = list()
        for subject_id in subject_ids:
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_numbers = np.unique((movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit).fetch('cell_number'))
            for cell_number in cell_numbers:
                cell_crit = 'cell_number = {}'.format(cell_number)
                cell_snrs, cell_types = (ephysanal_cell_attached.EphysCellType()*movies_all*imaging_gt.ROISweepCorrespondance()&subject_crit&cell_crit).fetch('movie_median_cawave_snr_per_ap','ephys_cell_type')
                median_cell_snr_list.append(np.median(cell_snrs))
                cell_types_list.append(cell_types[0])
                movie_number_list.append(len(cell_snrs))
                subject_id_list.append(np.median(subject_id))
                cell_number_list.append(np.median(cell_number))
         #%
        cell_number_list=np.asarray(cell_number_list)
        subject_id_list=np.asarray(subject_id_list)
        median_cell_snr_list=np.asarray(median_cell_snr_list)
        movie_number_list=np.asarray(movie_number_list)
        cell_types_list=np.asarray(cell_types_list)
        
        order = np.argsort(median_cell_snr_list)
        median_cell_snr_list = median_cell_snr_list[order][::-1]
        subject_id_list = subject_id_list[order][::-1]
        cell_number_list = cell_number_list[order][::-1]
        movie_number_list = movie_number_list[order][::-1]
        cell_types_list = cell_types_list[order][::-1]
        
        cell_counter = 0
        movie_counter = 0
        for subject_id,cell_number,median_cell_snr,cell_type in zip(subject_id_list,cell_number_list,median_cell_snr_list,cell_types_list):
# =============================================================================
#             if median_cell_snr<np.median(median_cell_snr_list):
#                 break
# =============================================================================
            if cell_counter>max_cells_to_export or movie_counter > max_movies_to_export:
                break
            cell_counter+=1
            #%
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_crit = 'cell_number = {}'.format(cell_number)
            cell_movies = movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit & cell_crit# &'sweep_number = 2'
            snrs,movie_numbers = cell_movies.fetch('movie_median_cawave_snr_per_ap','movie_number')
            order = np.argsort(snrs)[::-1]
            snrs = snrs[order]
            movie_numbers = movie_numbers[order]
            for movie_number, snr in zip(movie_numbers,snrs):
                dj.conn().connect()
                try:
                    filename = 'sensor_{}_subject_{}_cell_{}_movie_{}_snr_{:.2f}_putativecelltype_{}'.format(sensor,int(subject_id),int(cell_number),int(movie_number),snr,cell_type)
                    if filename+'.npz' in files_alread_saved:
                        print('{} already saved'.format(filename))
                        continue
                    else:
                        print('exporting {}'.format(filename))
                    movie_crit = 'movie_number = {}'.format(movie_number)
                    big_table = ephys_cell_attached.SweepMetadata()*imaging.ROI()*imaging.ROITrace()*imaging.ROINeuropilTrace()*imaging.MovieFrameTimes()*cell_movies&movie_crit&roi_crit#*ephysanal_cell_attached.ActionPotential()
                    
                    roi_f,neuropil_f,frame_times,roi_time_offset = big_table.fetch1('roi_f','neuropil_f','frame_times','roi_time_offset') #ap_times
                    ap_max_times = (ephysanal_cell_attached.ActionPotential()*cell_movies&movie_crit&roi_crit).fetch('ap_max_time')
                    cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()*cell_movies&movie_crit).fetch1('cell_recording_start', 'session_time')
                    
                    sweep_start_time =  (ephys_cell_attached.Sweep()*cell_movies&movie_crit).fetch1('sweep_start_time')
                    response_trace, sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()*cell_movies&movie_crit).fetch1('response_trace','sample_rate')
    
                    ephys_time = np.arange(len(response_trace))/sample_rate
                    frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time)) +roi_time_offset
    
                    roi_f_corr = roi_f-.8*neuropil_f
                    dff = (roi_f_corr-np.percentile(roi_f_corr,10))/np.percentile(roi_f_corr,10)
                    
                    np.savez_compressed(os.path.join(savedir,filename+'.npz'),
                                                                 F=roi_f,
                                                                 Fneu = neuropil_f,
                                                                 framerate = np.median(np.diff(frame_times)),
                                                                 ap_times=ap_max_times,
                                                                 frame_times = frame_times,
                                                                 ephys_time = ephys_time,
                                                                 ephys_raw = response_trace)
                    ax_ophys.cla()
                    ax_ephys.cla()
                    ax_ophys.plot(frame_times,dff,'g-')
                    ax_ophys.plot(np.asarray(ap_max_times,'float'),np.zeros(len(ap_max_times))-.5,'r|')
                    ax_ephys.plot(ephys_time,response_trace,'k-')
                    ax_ephys.set_xlabel('Time (s)')
                    ax_ephys.set_ylabel('ephys')
                    ax_ophys.set_ylabel('dF/F')
                    ax_ephys.set_xlim([0,np.max(ephys_time)])
                    fig.savefig(os.path.join(savedir,filename+'.png'), bbox_inches='tight')
                    movie_counter+=1
                except:
                    print('error with this record')
                    cell_movies&movie_crit&roi_crit
                    
#%%
                    

def plot_movie(key,title = '',filter_sigma = 5):
    cell_recording_start, session_time = (ephys_cell_attached.Cell()*experiment.Session()&key).fetch1('cell_recording_start', 'session_time')

    imaging_table = imaging.MovieFrameTimes()*imaging.ROI()*imaging.ROINeuropilTrace()*imaging.ROITrace()*imaging_gt.ROISweepCorrespondance()
    roi_f,neuropil_f,roi_time_offset,frame_times = (imaging_table&key).fetch1('roi_f','neuropil_f','roi_time_offset','frame_times')
    frame_times = frame_times+ roi_time_offset      
    frame_rate = np.median(np.diff(frame_times))
    if filter_sigma>0:
        roi_f = utils_plot.gaussFilter(roi_f,frame_rate,filter_sigma)
        neuropil_f = utils_plot.gaussFilter(neuropil_f,frame_rate,filter_sigma) 
        
    roi_f_corr = roi_f-neuropil_f*0.8
    f0 = np.percentile(roi_f_corr,10)
    roi_dff = (roi_f_corr-f0)/f0

    sweep_start_time =  (ephys_cell_attached.Sweep()&key).fetch1('sweep_start_time')
    response_trace, sample_rate = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()&key).fetch1('response_trace','sample_rate')
    response_trace,stim_idxs,stim_amplitudes = utils_ephys.remove_stim_artefacts_without_stim(response_trace, sample_rate)
    response_trace_filt = utils_plot.hpFilter(response_trace, 50, 1, sample_rate, padding = True)
    response_trace_filt = utils_plot.gaussFilter(response_trace_filt,sample_rate,sigma = .0001)
    
    ephys_time = np.arange(len(response_trace))/sample_rate
    frame_times = frame_times - ((cell_recording_start-session_time).total_seconds()+float(sweep_start_time))

    ap_max_times,ap_max_indices=(ephysanal_cell_attached.ActionPotential()&key).fetch('ap_max_time','ap_max_index')
    
    fig = plt.figure(figsize = [10,10])
    ax_ophys = fig.add_subplot(2,1,1)
    ax_ephys = fig.add_subplot(2,1,2,sharex = ax_ophys)
    ax_ophys.plot(frame_times,roi_dff,'g-')
    ax_ophys.plot(np.asarray(ap_max_times,'float'),np.zeros(len(ap_max_times))-.5,'r|')
    ax_ephys.plot(ephys_time,response_trace_filt,'k-')
    ax_ophys.set_title(title)
    data_dict = {'dff':roi_f_corr,
                'frame_times':frame_times,
                'ephys_trace':response_trace_filt,
                'ephys_time':ephys_time,
                'ap_max_times':np.asarray(ap_max_times,float)}
    return data_dict, fig

def export_high_snr_movies_from_datajoint():
    save_dir = '/home/rozmar/Data/Calcium_imaging/good_snr_movies_for_Ilya'
    
    
    sensors = np.unique(imaging_gt.SessionCalciumSensor().fetch('session_calcium_sensor'))
    for sensor in sensors:
        sensor_df = pd.DataFrame(imaging_gt.ROISweepCorrespondance()*imaging_gt.SessionCalciumSensor()*imaging_gt.MovieCalciumWaveSNR()&'session_calcium_sensor = "{}"'.format(sensor))
        sensor_df = sensor_df.sort_values('movie_median_cawave_snr_per_ap',ascending=False)
        for movie_row in sensor_df[:5].iterrows():
            movie_row = movie_row[1]
            dest_dir = os.path.join(save_dir,
                                    '{sensor}_{subject}_cell{cell}_movie{movie}_snr{snr:.2f}'.format(sensor = sensor,
                                                                                                subject = movie_row['subject_id'],
                                                                                                cell = movie_row['cell_number'],
                                                                                                movie = movie_row['movie_number'],
                                                                                                snr = movie_row['movie_median_cawave_snr_per_ap'] ))
            if os.path.isdir(dest_dir):
                continue
            key = {'subject_id': movie_row['subject_id'],
                   'session':movie_row['session'],
                   'cell_number':movie_row['cell_number'],
                   'sweep_number':movie_row['sweep_number'],  
                   'movie_number':movie_row['movie_number'],
                   'motion_correction_method':"Suite2P",# the second four entries restrict the imaging data (see above)
                   'roi_type':"Suite2P",
                   'channel_number':1,
                   'neuropil_number':1}
            #print(movie_row)
            data_dict,fig = plot_movie(key,title = imaging.Movie()&key)
            reg_movie_file_repository,reg_movie_file_directory,reg_movie_file_name = (imaging.RegisteredMovieFile()&key).fetch('reg_movie_file_repository','reg_movie_file_directory','reg_movie_file_name')
            source_dir = os.path.join(dj.config['locations.{}'.format(reg_movie_file_repository[0])],reg_movie_file_directory[0])
            
            regtiff_dest_dir = os.path.join(dest_dir,'regtiff')
            Path(regtiff_dest_dir).mkdir(parents=True, exist_ok=True)
            for fname in reg_movie_file_name:
                shutil.copy(os.path.join(source_dir,fname),os.path.join(regtiff_dest_dir,fname))
                
            np.savez_compressed(os.path.join(dest_dir,'data.npz'),
                                dff=data_dict['dff'],
                                frame_times = data_dict['frame_times'],
                                ephys_trace = data_dict['ephys_trace'],
                                ephys_time = data_dict['ephys_time'],
                                ap_max_times = data_dict['ap_max_times'])
            fig.savefig(os.path.join(dest_dir,'movie.png'), bbox_inches='tight')