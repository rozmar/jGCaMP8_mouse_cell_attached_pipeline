import datajoint as dj

# =============================================================================
# import numpy as np
# 
import pipeline.lab as lab#, ccf
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 
schema = dj.schema(get_schema_name('experiment'),locals())
#schema = dj.schema('rozmar_foraging_experiment',locals())
# =============================================================================
#schema = dj.schema('rozmar_tutorial', locals())


@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session : smallint 		# session number
    ---
    session_date   : date
    session_time   : time
    -> lab.Person
    -> lab.Rig
    """
@schema
class SessionComment(dj.Imported):
    definition = """
    -> Session
    session_comment : varchar(512)
    """

@schema
class VisualStimTrial(dj.Imported):
    definition = """
    -> Session
    visual_stim_trial : smallint 		# trial number
    ---
    visual_stim_trial_start              : decimal(10, 4)  # (s) relative to session beginning 
    visual_stim_trial_end                : decimal(10, 4)  # (s) relative to session beginning 
    visual_stim_grating_start            : decimal(10, 4)  # (s) relative to session beginning 
    visual_stim_grating_end              : decimal(10, 4)  # (s) relative to session beginning 
    visual_stim_grating_angle            : int  # orientation of drifting grating
    visual_stim_grating_cyclespersecond  : double # 
    visual_stim_grating_cyclesperdegree  : double #
    visual_stim_grating_amplitude        : double #
    """

