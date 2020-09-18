import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())
from PIL import Image
import numpy as np
import os
#%%
@schema
class Movie(dj.Imported):
    definition = """
    -> experiment.Session
    movie_number                : smallint
    ---
    movie_name                  : varchar(200)          # movie name
    movie_x_size                : double                # (pixels)
    movie_y_size                : double                # (pixels)
    movie_frame_rate            : double                # (Hz)             
    movie_frame_num             : int                   # number of frames        
    movie_start_time            : decimal(12, 6)        # (s) from session start # maybe end_time would also be useful
    movie_pixel_size            : decimal(5,2)          # in microns
    """ 
@schema
class MovieChannel(dj.Imported):
    definition = """
    -> Movie
    channel_number              : smallint
    ---
    channel_color               : varchar (20) #green/red
    """ 
@schema
class MovieMetaData(dj.Imported):
    definition = """
    -> Movie
    ---
    movie_hbeam_power                     : decimal(5,2)         # power percent
    movie_hmotors_sample_x                : decimal(6,2)          # sample location in microns
    movie_hmotors_sample_y                : decimal(6,2)          # sample location in microns
    movie_hmotors_sample_z                : decimal(6,2)          # sample location in microns
    movie_hroimanager_lineperiod          : double                # 
    movie_hroimanager_scanframeperiod     : double                # 
    movie_hroimanager_scanzoomfactor      : double                # 
    movie_hscan2d_fillfractionspatial     : double                #
    movie_hscan2d_fillfractiontemporal    : double                #
    movie_hscan2d_mask                    : blob                  #
    movie_hscan2d_scanmode                : varchar(50)           # resonant/...
    """ 

@schema
class MoviePowerPostObjective(dj.Imported):
    definition = """
    -> Movie
    ---
    movie_power                           : decimal(6,2)          # power after objective in mW - calculated
    """ 
    
@schema
class MovieFrameTimes(dj.Imported):
    definition = """
    -> Movie
    ---
    frame_times                : longblob              # timing of each frame relative to Session start - frame start time
    """

@schema
class MovieFile(dj.Imported): #MovieFile
    definition = """
    -> Movie 
    movie_file_number         : smallint
    ---
    movie_file_repository     : varchar(200)          # name of the repository where the data are
    movie_file_directory      : varchar(200)          # location of the files  
    movie_file_name           : varchar(100)          # file name
    movie_file_start_frame    : int                   # first frame of this file that belongs to the movie
    movie_file_end_frame      : int                   # last frame of this file that belongs to the movie
    """


@schema
class MotionCorrectionMethod(dj.Lookup): 
    definition = """
    #
    motion_correction_method  :  varchar(30)
    """
    contents = zip(['Matlab','VolPy','Suite2P','VolPy2x'])

@schema
class RegisteredMovie(dj.Imported): #MovieFile
    definition = """
    -> Movie
    -> MotionCorrectionMethod
    ---
    registered_movie_mean_image : longblob
    """
@schema
class MotionCorrectionMetrics(dj.Imported): 
    definition = """
    -> RegisteredMovie
    ---
    motion_corr_parameters      : longblob              # probably a dict?  ##
    motion_corr_metrics         : longblob              # ??
    """
@schema
class RegisteredMovieFile(dj.Imported): #MovieFile
    definition = """
    -> RegisteredMovie 
    movie_file_number         : smallint
    ---
    reg_movie_file_repository     : varchar(200)          # name of the repository where the data are
    reg_movie_file_directory      : varchar(200)          # location of the files  
    reg_movie_file_name           : varchar(100)          # file name
    reg_movie_file_start_frame    : int                   # first frame of this file that belongs to the movie
    reg_movie_file_end_frame      : int                   # last frame of this file that belongs to the movie
    """

@schema
class MotionCorrection(dj.Imported): 
    definition = """
    -> RegisteredMovie
    motion_correction_id    : smallint             # id of motion correction in case of multiple motion corrections
    ---
    motion_corr_description     : varchar(300)         #description of the motion correction
    motion_corr_vectors         : longblob             # registration vectors   
    """


@schema
class ROIType(dj.Lookup): 
    definition = """
    #
    roi_type  :  varchar(30)
    """
    contents = zip(['Suite2P',
                    'VolPy'])

@schema
class ROI(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> RegisteredMovie
    -> ROIType   
    -> MovieChannel
    roi_number                      : int           # roi number (restarts for every registered movie, same number in different channels means the same ROI)
    ---
    roi_f                           : longblob      # spikepursuit
    roi_centroid_x                  : double        # ROI centroid  x, pixels
    roi_centroid_y                  : double        # ROI centroid  y, pixels
    roi_mask                        : longblob      # pixel mask 
    roi_time_offset                 : double        # time offset of ROI relative to frame time
    """
    
@schema
class ROINeuropil(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    ---
    neuropil_f                      : longblob      # spikepursuit
    neuropil_mask                   : longblob      # pixel mask 
    """
