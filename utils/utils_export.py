import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
import numpy as np
def export_s2f_data_from_datajoint():
    #%% export traces for s2f
    # this script locates the cells with highest snr for each sensor, then exports the first n cells or m movies, whichever comes first
    roi_crit = {'channel_number':1,
                'motion_correction_method':'Suite2P',
                'roi_type':'Suite2P',
                'neuropil_number':1}
    max_sweep_ap_hw_std_per_median =  .2
    min_sweep_ap_snr_dv_median = 10
    ephys_quality_crit_1 = 'sweep_ap_snr_dv_median>={}'.format(min_sweep_ap_snr_dv_median)
    ephys_quality_crit_2 = 'sweep_ap_hw_std_per_median<={}'.format(max_sweep_ap_hw_std_per_median)
    calcium_sensors = np.unique(imaging_gt.SessionCalciumSensor().fetch('session_calcium_sensor'))
    max_cells_to_export = 10 # per sensor
    max_movies_to_export = 50 #per sensor
    for sensor in calcium_sensors:
        sensor_crit = 'session_calcium_sensor="{}"'.format(sensor)
        movies_all = (ephysanal_cell_attached.SweepAPQC()*imaging_gt.SessionCalciumSensor()*imaging_gt.MovieCalciumWaveSNR() & sensor_crit&ephys_quality_crit_1&ephys_quality_crit_2)
        subject_ids = np.unique(movies_all.fetch('subject_id'))
        median_cell_snr_list = list()
        subject_id_list = list()
        cell_number_list = list()
        movie_number_list = list()
        for subject_id in subject_ids:
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_numbers = np.unique((movies_all*imaging_gt.ROISweepCorrespondance() & subject_crit).fetch('cell_number'))
            for cell_number in cell_numbers:
                cell_crit = 'cell_number = {}'.format(cell_number)
                cell_snrs = (movies_all*imaging_gt.ROISweepCorrespondance()&subject_crit&cell_crit).fetch('movie_median_cawave_snr_per_ap')
                median_cell_snr_list.append(np.median(cell_snrs))
                movie_number_list.append(len(cell_snrs))
                subject_id_list.append(np.median(subject_id))
                cell_number_list.append(np.median(cell_number))
         #%
        cell_number_list=np.asarray(cell_number_list)
        subject_id_list=np.asarray(subject_id_list)
        median_cell_snr_list=np.asarray(median_cell_snr_list)
        movie_number_list=np.asarray(movie_number_list)
        order = np.argsort(median_cell_snr_list)
        median_cell_snr_list = median_cell_snr_list[order][::-1]
        subject_id_list = subject_id_list[order][::-1]
        cell_number_list = cell_number_list[order][::-1]
        movie_number_list = movie_number_list[order][::-1]
        
        cell_counter = 0
        movie_counter = 0
        for subject_id,cell_number,median_cell_snr in zip(subject_id_list,cell_number_list,median_cell_snr_list):
            if cell_counter>max_cells_to_export:
                break
            subject_crit = 'subject_id = {}'.format(subject_id)
            cell_crit = 'cell_number = {}'.format(cell_number)
            cell_movies = movies_all*imaging_gt.ROISweepCorrespondance()&subject_crit&cell_crit
            snrs,movie_numbers = cell_movies.fetch('movie_median_cawave_snr_per_ap','movie_number')
            order = np.argsort(snrs)[::-1]
            snrs = snrs[order]
            movie_numbers = movie_numbers[order]
            for movie_number, snr in zip(movie_numbers,snrs):
                movie_crit = 'movie_number = {}'.format(movie_number)
                big_table = ephys_cell_attached.SweepMetadata()*imaging.ROITrace()*imaging.MovieFrameTimes()*ephysanal_cell_attached.ActionPotential()*imaging_gt.ROISweepCorrespondance()*cell_movies&movie_crit&roi_crit
                break
                #F,Fneu,framerate,ap_times,frame_times,neuropil_r = 
            break
        
        
        break