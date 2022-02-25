import numpy as np
import utils.utils_pipeline as utils_pipeline
import notebook_google.notebook_main as online_notebook
from pipeline.ingest import datapipeline_metadata,datapipeline_elphys,datapipeline_imaging,datapipeline_imaging_gt, datapipeline_behavior
#% set parameters

s2p_params = {'cell_soma_size':20, # microns - obsolete
              'max_reg_shift':20, # microns
              'max_reg_shift_NR': 10, # microns
              'block_size': 60, #microns
              'smooth_sigma':.5, # microns
              'smooth_sigma_time':.001, #seconds,
              'basedir':'/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig', # folder where the raw ephys and movies are found
              'basedir_out':'/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig', #'/home/rozmar/Data/Calcium_imaging/suite2p_fast/Genie_2P_rig'
              'basedir_out_slow':'/home/rozmar/Data/Calcium_imaging/suite2p/686_688',
              'overwrite': False} # folder where the suite2p output is saved

metadata_params = {'notebook_name':"Genie Calcium Imaging",
                   'metadata_dir':"/home/rozmar/Data/Calcium_imaging/Metadata",
                   'genie_2p_power_notebook_name':"genie2p rig power",
                   'genie_2p_power_metadata_dir':"/home/rozmar/Data/Calcium_imaging/genie_2p_rig_powers"}
anatomy_metadata_params = {'notebook_name':"Genie anti-GFP immunostaining",
                           'metadata_dir':"/home/rozmar/Data/Anatomy/Metadata"}

ephys_params = {'basedir':'/home/rozmar/Network/rozsam-nas-01-smb/Calcium_imaging/raw/Genie_2P_rig'}#'/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig'

visual_stim_params = {'basedir': '/home/rozmar/Data/Visual_stim/raw/Genie_2P_rig'}


#%% registration and ROI detection
utils_pipeline.batch_run_suite2p(s2p_params)
utils_pipeline.suite2p_extract_altered_neuropil(s2p_params,neuropil_pixel_nums = [4,8,16])
#%% update metadata from google spreadsheet
online_notebook.update_metadata(metadata_params['notebook_name'],metadata_params['metadata_dir'])
online_notebook.update_metadata(anatomy_metadata_params['notebook_name'],anatomy_metadata_params['metadata_dir'])
online_notebook.update_metadata(metadata_params['genie_2p_power_notebook_name'],metadata_params['genie_2p_power_metadata_dir'],transposed = True)

#%% Populate LAB schema
datapipeline_metadata.populatemetadata() #LAB schema
#% # ephys goes first
datapipeline_elphys.populateelphys_wavesurfer(cores = 3) #ephys_cell_attached -  only 3 fits in the memory at the same time
datapipeline_elphys.populatemytables(paralel = True, cores = 12)
#% basic imaging data
datapipeline_imaging.upload_movie_metadata_scanimage()
datapipeline_imaging.upload_movie_registration_scanimage()
#% ROIs and corresponding metadata
datapipeline_imaging.upload_ROIs_scanimage() # using only scanimage files
datapipeline_imaging_gt.upload_ground_truth_ROIs_and_notes() # using the metadata from google drive in csv files
#%%
datapipeline_imaging_gt.populatemytables(paralel = True, cores = 12)
#% 
datapipeline_elphys.populatemytables(paralel = True, cores = 12)
datapipeline_imaging_gt.populatemytables(paralel = True, cores = 12)







    
#%%
datapipeline_behavior.populate_behavior() # this one needs frame times







