import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_cell_attached as ephys_cell_attached
import pipeline.ephysanal as ephysanal
import pipeline.imaging as imaging
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging_gt'),locals()) # TODO ez el van baszva

import numpy as np
import scipy
#%%


@schema
class ROISweepCorrespondance(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    -> imaging.ROI
    """

    