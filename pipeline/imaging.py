import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())
from PIL import Image
import numpy as np
import os
import pandas as pd
import datetime
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
    definition = """#beam expander info needed, populate after movienote
    -> Movie
    ---
    movie_power                           : decimal(6,2)          # power after objective in mW - calculated
    movie_power_days_to_calibration       : int
    """ 
    def make(self, key):
        #%%
        #key = {'subject_id': 478348, 'session': 1, 'movie_number': 35}
        
        if len(MovieNote()&key) == 0:#imaging.
            print('no notes for {}'.format(key))
        else:
            print(key)
            movie_hbeam_power = float((MovieMetaData()&key).fetch1('movie_hbeam_power'))#imaging.
            try:
                beamexp = (MovieNote()&key&'movie_note_type = "beam expander"').fetch1('movie_note')#imaging
            except:
                beamexp = 'nan' 
            session_date = (experiment.Session()&key).fetch1('session_date')
            calibration_curve_names =os.listdir(dj.config['locations.metadata_genie_2p_rig_powers'])
            calibration_times = list()
            calibration_files = list()
            for calibration_curve_name in calibration_curve_names:
                try:
                    calibration_times.append(datetime.datetime.strptime(calibration_curve_name[:-4],'%Y-%m-%d').date())
                    calibration_files.append(calibration_curve_name)
                except:
                    pass#print(calibration_curve_name)
            needed_idx = np.argmin(np.abs(session_date-np.asarray(calibration_times)))
            time_difi = session_date-np.asarray(calibration_times)[needed_idx]
            key['movie_power_days_to_calibration'] = time_difi.days
            df = pd.read_csv(os.path.join(dj.config['locations.metadata_genie_2p_rig_powers'],np.asarray(calibration_files)[needed_idx]))
            x_col = None
            y_col = None
            for column_name in df.keys():
                if (beamexp == 'nan'or beamexp =='none') and 'power' in column_name and 'scanning' in column_name and 'exp' not in column_name:
                    y_col = column_name
                elif '2' in beamexp and 'power' in column_name and 'scanning' in column_name and 'exp' in column_name and '2' in column_name:
                    y_col = column_name
                elif 'command' in column_name:
                    x_col = column_name
            xvals = df[x_col].values
            yvals = df[y_col].values
            xvals_real = list()
            yvals_real = list()
            for x,y in zip(xvals,yvals):
                try:
                    x = float(x)
                    y = float(y)
                    xvals_real.append(x)
                    yvals_real.append(y)
                except:
                    pass
                    
            p = np.polyfit(xvals_real,yvals_real,1)
            key['movie_power']=np.polyval(p,movie_hbeam_power)
            #%%
            self.insert1(key,skip_duplicates=True)
                #break
            
            
        
        
            
    #%%
    
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
    roi_pixel_num                   : int
    roi_weights                     : longblob      # weight of each pixel
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
    neuropil_pixel_num                   : int
    """
    
@schema
class ROITrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROI
    -> MovieChannel
    ---
    roi_f                           : longblob
    roi_f_mean                      : double
    roi_f_median                    : double
    roi_f_min                       : double
    roi_f_max                       : double
    """    


@schema
class ROINeuropilTrace(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> ROINeuropil
    -> MovieChannel
    ---
    neuropil_f                           : longblob      # spikepursuit
    neuropil_f_mean                      : double
    neuropil_f_median                    : double
    neuropil_f_min                       : double
    neuropil_f_max                       : double
    """
