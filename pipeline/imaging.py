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
    channel_offset = NULL       : int #
    channel_subtract_offset     : tinyint # 1 or 0
    """ 
@schema
class MovieMetaData(dj.Imported):
    definition = """
    -> Movie
    ---
    movie_hbeam_power                     : decimal(5,2)         # power percent
    movie_hmotors_sample_x = NULL         : decimal(8,2)          # sample location in microns
    movie_hmotors_sample_y = NULL         : decimal(8,2)          # sample location in microns
    movie_hmotors_sample_z = NULL         : decimal(8,2)          # sample location in microns
    movie_hroimanager_lineperiod          : double                # 
    movie_hroimanager_scanframeperiod     : double                # 
    movie_hroimanager_scanzoomfactor      : double                # 
    movie_hscan2d_fillfractionspatial     : double                #
    movie_hscan2d_fillfractiontemporal    : double                #
    movie_hscan2d_mask = NULL             : blob                  # not saved in earlier scanimage version
    movie_hscan2d_scanmode                : varchar(50)           # resonant/...
    """ 

@schema 
class MovieNoteType(dj.Lookup):
    definition = """
    movie_note_type : varchar(20)
    """
    contents = zip(('ephys', 'imaging', 'analysis', 'cell', 'quality','beam expander'))
                    
@schema
class MovieNote(dj.Imported):
    definition = """
    -> Movie
    -> MovieNoteType
    ---
    movie_note  : varchar(255) 
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
class RegisteredMovie(dj.Imported):
    definition = """
    -> Movie
    -> MotionCorrectionMethod
    """
@schema
class RegisteredMovieImage(dj.Imported):
    definition = """
    -> RegisteredMovie
    -> MovieChannel
    ---
    registered_movie_mean_image               : longblob
    registered_movie_mean_image_enhanced=null : longblob
    registered_movie_max_projection=null      : longblob
    """
@schema
class MotionCorrectionMetrics(dj.Imported): 
    definition = """
    -> RegisteredMovie
    ---
    motion_corr_residual_rigid        : longblob #regDX[:,0]
    motion_corr_residual_nonrigid     : longblob  #regDX[:,1]
    motion_corr_residual_nonrigid_max : longblob  #regDX[:,2]
    motion_corr_tpc                   : longblob #tPC
    """
@schema
class RegisteredMovieFile(dj.Imported): #MovieFile
    definition = """
    -> RegisteredMovie 
    reg_movie_file_number         : smallint
    ---
    reg_movie_file_repository     : varchar(200)          # name of the repository where the data are
    reg_movie_file_directory      : varchar(200)          # location of the files  
    reg_movie_file_name           : varchar(100)          # file name
    """

@schema
class MotionCorrection(dj.Imported): 
    definition = """
    -> RegisteredMovie
    motion_correction_id    : smallint             # id of motion correction in case of multiple motion corrections
    ---
    motion_corr_description     : varchar(300)         #description of the motion correction - rigid/nonrigid
    motion_corr_x_block=null    : longblob            
    motion_corr_y_block=null    : longblob
    motion_corr_x_offset        : longblob         # registration vectors   
    motion_corr_y_offset        : longblob         # registration vectors   
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
    roi_number                      : int           # roi number (restarts for every registered movie, same number in different channels means the same ROI)
    ---
    roi_centroid_x                  : double        # ROI centroid  x, pixels
    roi_centroid_y                  : double        # ROI centroid  y, pixels
    roi_xpix                        : longblob      # pixel mask 
    roi_ypix                        : longblob      # pixel mask 
    roi_time_offset                 : double        # time offset of ROI relative to frame time
    roi_aspect_ratio                : double
    roi_compact                     : double
    roi_radius                      : double
    roi_skew                        : double
    """
@schema
class ROINeuropil(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    neuropil_number                      : int  # in case there are different neuropils
    ---
    neuropil_xpix                        : longblob      # pixel mask 
    neuropil_ypix                        : longblob      # pixel mask 
    """
    
@schema
class ROITrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    -> MovieChannel
    ---
    roi_f                           : longblob
    """    


@schema
class ROINeuropilTrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROINeuropil
    -> MovieChannel
    ---
    neuropil_f                      : longblob      # spikepursuit
    """
