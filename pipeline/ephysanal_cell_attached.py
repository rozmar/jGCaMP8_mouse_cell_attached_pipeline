import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy import interpolate
from pipeline import pipeline_tools, lab, experiment, ephys_cell_attached#, ephysanal
from utils import utils_ephys
import time
#dj.conn()
#%%
schema = dj.schema(pipeline_tools.get_schema_name('ephysanal_cell_attached'),locals())

#%%
@schema
class EphysCellType(dj.Computed):
    definition = """
    -> ephys_cell_attached.Cell
    ---
    ephys_cell_type : varchar(20) # pyr / int / unidentified
    """
    def make(self, key):
        #%
# =============================================================================
#         key = {'subject_id':471991,
#                'session':1,
#                'cell_number':1}
# =============================================================================
        
        cellspikeparameters = CellSpikeParameters()&key#ephysanal_cell_attached
        if len(cellspikeparameters) ==0:
            key['ephys_cell_type'] = 'unidentified'
        else:
            if len(cellspikeparameters)>1:
                cellspikeparameters = cellspikeparameters&'recording_mode = "current clamp"'
            ap_peak_to_trough_time_median,ap_ahp_amplitude_median = cellspikeparameters.fetch1('ap_peak_to_trough_time_median','ap_ahp_amplitude_median')
            if ap_peak_to_trough_time_median < 0.65 and ap_ahp_amplitude_median>.1:
                key['ephys_cell_type'] = 'int'
            else:
                key['ephys_cell_type'] = 'pyr'

        self.insert1(key,skip_duplicates=True)
        
        
    #%
@schema
class ActionPotential(dj.Computed):
    definition = """
    -> ephys_cell_attached.Sweep
    ap_num : smallint unsigned # action potential number in sweep
    ---
    ap_max_index=null            : int unsigned # index of AP max on sweep
    ap_max_time=null             : decimal(8,4) # time of the AP max relative to sweep start
    ap_snr_v=null                : double       #
    ap_snr_dv=null               : double       #
    ap_halfwidth=null            : double       # hw in ms
    ap_full_width=null           : double       # in ms           
    ap_rise_time=null            : double       # in ms
    ap_integral=null             : double       # A*s or V*s depending on recording mode
    ap_amplitude=null            : double       # in A or V depending on recording mode
    ap_ahp_amplitude=null        : double       # in A or V depending on recording mode
    ap_isi=null                  : double       # s between AP before and this AP - null if first AP in sweep
    ap_peak_to_trough_time=null  : double       # ms
    ap_baseline_current=null     : double       # A
    ap_baseline_voltage=null     : double       # V
    """
    def make(self, key):
        #%%
        #key = {'subject_id': 478342, 'session': 1, 'cell_number': 6, 'sweep_number': 2}
        #print('starting - {}'.format(np.random.randint(1000)))
        sample_rate,recording_mode = (ephys_cell_attached.SweepMetadata()&key).fetch1('sample_rate','recording_mode')
        trace = (ephys_cell_attached.SweepResponse()&key).fetch1('response_trace')
        trace_original = trace.copy()
        squarepulse_indices = (SquarePulse()&key&'square_pulse_num>0').fetch('square_pulse_start_idx')#ephysanal_cell_attached.
        if len(squarepulse_indices)>0:
            #print(key)
            trace = utils_ephys.remove_stim_artefacts_with_stim(trace,sample_rate,squarepulse_indices)
        try:
            stimulus = (ephys_cell_attached.SweepStimulus()&key).fetch1('stimulus_trace')
        except:
            stimulus = None
            #%%
        try:       
            ap_dict = utils_ephys.findAPs(trace, sample_rate,recording_mode, SN_min = 5,method = 'diff') 
        except:
            print('error in AP detection')
            print(key)
            return None
        apnum = len(ap_dict['peak_idx'])
        if apnum>0:
            dj_dict = {'subject_id':[key['subject_id']]*apnum,
                       'session': [key['session']]*apnum,
                       'cell_number': [key['cell_number']]*apnum, 
                       'sweep_number': [key['sweep_number']]*apnum,
                       'ap_num': np.arange(apnum)+1,
                       'ap_max_index':ap_dict['peak_idx'],
                       'ap_max_time':ap_dict['peak_times'],
                       'ap_snr_v':ap_dict['peak_snr_v'],
                       'ap_snr_dv':ap_dict['peak_snr_dv'],
                       'ap_halfwidth':ap_dict['half_width'],
                       'ap_full_width':ap_dict['full_width'],
                       'ap_rise_time':ap_dict['rise_time'],
                       'ap_integral':ap_dict['ap_integral'],
                       'ap_amplitude':ap_dict['peak_amplitude_left'],
                       'ap_ahp_amplitude':ap_dict['ahp_amplitude'],
                       'ap_isi':ap_dict['isi_left'],
                       'ap_peak_to_trough_time':ap_dict['peak_to_through']}
            dj_list = list()
            for idx in range(apnum):
                dictnow = dict()
                for key in dj_dict.keys():
                    dictnow[key] = dj_dict[key][idx]
                baseline_idxs = ap_dict['baseline_idxs'][idx]
                
                if recording_mode == 'current clamp':
                    try:
                        dictnow['ap_baseline_current'] = np.nanmedian(stimulus[baseline_idxs])
                    except:
                        dictnow['ap_baseline_current'] = None
                    dictnow['ap_baseline_voltage'] = np.nanmedian(trace_original[baseline_idxs])
                elif recording_mode == 'voltage clamp':
                    try:
                        dictnow['ap_baseline_voltage'] = np.nanmedian(stimulus[baseline_idxs])
                    except:
                        dictnow['ap_baseline_voltage'] = None
                    dictnow['ap_baseline_current'] = np.nanmedian(trace_original[baseline_idxs])
                else:
                    print('unknown recording mode')
                    return None
                
                dj_list.append(dictnow)
# =============================================================================
#             t = np.arange(len(trace))/sample_rate
#             plt.plot(t,trace)
#             plt.plot(t[ap_dict['peak_idx']],trace[ap_dict['peak_idx']],'ro')
# =============================================================================
          #%%
            self.insert(dj_list,skip_duplicates=True)
        else:
            print('no AP found')
            print(key)
            key['ap_num']=0
            self.insert1(key,skip_duplicates=True)

@schema
class SweepAPQC(dj.Computed):
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    sweep_ap_snr_dv_min=null       : double
    sweep_ap_snr_dv_median=null       : double
    sweep_ap_snr_dv_percentiles=null  : blob #10,20,30,40..90th percentiles
    sweep_ap_snr_v_min=null       : double
    sweep_ap_snr_v_median=null       : double
    sweep_ap_snr_v_percentiles=null  : blob #10,20,30,40..90th percentiles
    sweep_ap_hw_median=null           : double 
    sweep_ap_hw_percentiles=null      : blob #10,20,30,40..90th percentiles
    sweep_ap_hw_max=null           : double 
    sweep_ap_hw_std=null           : double 
    sweep_ap_hw_std_per_median=null           : double 
    sweep_ap_fw_median=null           : double 
    sweep_ap_fw_percentiles=null      : blob #10,20,30,40..90th percentiles
    sweep_ap_fw_max=null           : double 
    sweep_ap_fw_std=null           : double 
    sweep_ap_fw_std_per_median=null           : double 
    """
    def make(self, key):
        percentiles = [10,20,30,40,50,60,70,80,90]
        hws,snr_dvs,snr_vs,fws = (ActionPotential()&key).fetch('ap_halfwidth','ap_snr_dv','ap_snr_v','ap_full_width')
        hws = hws[np.isnan(hws)==False]
        fws = fws[np.isnan(fws)==False]
        snr_dvs = snr_dvs[np.isnan(snr_dvs)==False]
        snr_vs = snr_vs[np.isnan(snr_vs)==False]
        if len(hws)>=10:
            key['sweep_ap_snr_v_percentiles'] = np.percentile(snr_vs,percentiles)
            key['sweep_ap_snr_dv_percentiles'] = np.percentile(snr_dvs,percentiles)
            key['sweep_ap_hw_percentiles'] = np.percentile(hws,percentiles)
            key['sweep_ap_fw_percentiles'] = np.percentile(fws,percentiles)
        if len(hws)>=1:
            key['sweep_ap_snr_v_min'] = np.nanmin(snr_vs)
            key['sweep_ap_snr_v_median'] = np.nanmedian(snr_vs)
            key['sweep_ap_snr_dv_min'] = np.nanmin(snr_dvs)
            key['sweep_ap_snr_dv_median'] = np.nanmedian(snr_dvs)            
            
            key['sweep_ap_hw_max'] = np.nanmax(hws)
            key['sweep_ap_hw_median'] = np.nanmedian(hws)
            key['sweep_ap_hw_std'] = np.nanstd(hws)
            
            key['sweep_ap_fw_max'] = np.nanmax(fws)
            key['sweep_ap_fw_median'] = np.nanmedian(fws)
            key['sweep_ap_fw_std'] = np.nanstd(fws)
            
            key['sweep_ap_fw_std_per_median'] = key['sweep_ap_fw_std']/key['sweep_ap_fw_median']
            key['sweep_ap_hw_std_per_median'] = key['sweep_ap_hw_std']/key['sweep_ap_hw_median']
            
        self.insert1(key,skip_duplicates=True)
        
    

            
@schema
class APGroup(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    ap_group_number   : int # unique within cell
    ---
    ap_group_ap_num     : int
    ap_group_start_time : decimal(8,4) #- time of first AP in group from recording start (cell start time)
    ap_group_end_time   : decimal(8,4) #- time of last AP in group from recording start (cell start time)
    ap_group_min_snr_v  : double
    ap_group_min_snr_dv : double
    ap_group_pre_isi    : double
    ap_group_post_isi   : double
    """
    class APGroupMember(dj.Part):
        definition = """
        -> master
        -> ActionPotential
        """

@schema
class IngestAPGroup(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    ap_group_complete : int
    """
    def make(self, key):
        key_original = key.copy()
        key_original['ap_group_complete'] = 1
        min_baseline_time = .1
        max_integration_time = .02 #.1 for pyramidal cells
        
        sweep_numbers = (ephys_cell_attached.Sweep()&key).fetch('sweep_number')
        apgroup_list = list()
        groupmember_list = list()
        ap_group_number = 0
        for sweep_number in sweep_numbers:
            dj.conn().ping()
            key['sweep_number'] = sweep_number
            sweep_start_time,sweep_end_time = (ephys_cell_attached.Sweep()&key).fetch1('sweep_start_time','sweep_end_time')
            sweep_end_time = float(sweep_end_time-sweep_start_time)
            sweep_start_time = float(sweep_start_time)
            ap_num,ap_max_time,ap_snr_v,ap_snr_dv = (ephys_cell_attached.Sweep()*ActionPotential()&key).fetch('ap_num','ap_max_time','ap_snr_v','ap_snr_dv')#ephysanal_cell_attached
            ap_max_time = np.asarray(ap_max_time,float)
            
            isis_pre = np.diff(np.concatenate([[0],ap_max_time]))
            isis_post = np.diff(np.concatenate([ap_max_time,[sweep_end_time]]))
            
            group_start_idxes = np.where(isis_pre>min_baseline_time)[0]
            group_end_idxes = np.where(isis_post>max_integration_time)[0]
            for group_start_idx in group_start_idxes:
                if any(group_end_idxes>=group_start_idx):
                    apgroup_key = key.copy()
                    ap_group_number += 1
                    group_end_idx = group_end_idxes[np.argmax(group_end_idxes>=group_start_idx)]
                    ap_indices_now = np.arange(group_start_idx,group_end_idx+1)
                    
                    apgroup_key['ap_group_number'] = ap_group_number
                    apgroup_key['ap_group_ap_num'] = len(ap_indices_now)
                    apgroup_key['ap_group_start_time'] = ap_max_time[group_start_idx] + sweep_start_time
                    apgroup_key['ap_group_end_time'] = ap_max_time[group_end_idx] + sweep_start_time
                    apgroup_key['ap_group_min_snr_v'] = np.min(ap_snr_v[ap_indices_now])
                    apgroup_key['ap_group_min_snr_dv'] = np.min(ap_snr_dv[ap_indices_now])
                    apgroup_key['ap_group_pre_isi'] = isis_pre[group_start_idx]
                    apgroup_key['ap_group_post_isi'] = isis_post[group_end_idx]            
                    apgroup_list.append(apgroup_key)
                    for ap_idx in ap_indices_now:
                        groupmember_key = key.copy()
                        groupmember_key['ap_group_number'] = ap_group_number
                        groupmember_key['ap_num'] = ap_num[ap_idx]
                        groupmember_list.append(groupmember_key)
                        
        #with dj.conn().transaction: #inserting one cell
        print('uploading AP groups to datajoint: {}'.format(key))
        APGroup().insert(apgroup_list, allow_direct_insert=True)#ephysanal_cell_attached
        dj.conn().ping()
        APGroup.APGroupMember().insert(groupmember_list, allow_direct_insert=True)      #ephysanal_cell_attached
        self.insert1(key_original,skip_duplicates=True)



@schema
class APGroupTrace(dj.Imported):
    definition = """
    -> APGroup
    ---
    ap_group_trace            : longblob
    ap_group_trace_offset     : double
    ap_group_trace_multiplier : double
    ap_group_trace_peak_idx   : int
    """  
@schema
class IngestAPGroupTrace(dj.Imported):  
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    ap_group_trace_complete       : int #1 if done
    """     
    def make(self, key):
        #%%
# =============================================================================
#         key = {'subject_id': 471991,
#                'session': 1,
#                'cell_number': 1,
#                'sweep_number': 0}
# =============================================================================
        
        time_back = .5
        time_forward = 1.5
        
        apgroups = APGroup()&key#ephysanal_cell_attached.
        if len(apgroups)==0:
            #return None
            pass
        response_trace = (ephys_cell_attached.SweepResponse()&key).fetch1('response_trace')
        sample_rate,recording_mode = (ephys_cell_attached.SweepMetadata()&key).fetch1('sample_rate','recording_mode')
        step_back = int(time_back*sample_rate)
        step_forward = int(time_forward*sample_rate)
        #%
        apgrouptrace_list = list()
        for apgroup in apgroups:
            
            apgroup_members = APGroup.APGroupMember()&apgroup#ephysanal_cell_attached.
            ap_nums = apgroup_members.fetch('ap_num')
            first_ap = ActionPotential()&key&'ap_num = {}'.format(np.min(ap_nums))#ephysanal_cell_attached.
            ap_max_index,ap_amplitude,ap_baseline_current,ap_baseline_voltage = first_ap.fetch1('ap_max_index','ap_amplitude','ap_baseline_current','ap_baseline_voltage')
            actual_step_back = np.min([step_back,ap_max_index])
            actual_step_forward = np.min([step_forward,len(response_trace)-ap_max_index])
            
            if recording_mode == 'voltage clamp':
                baseline_value = ap_baseline_current
                multiplier = -1
            elif recording_mode == 'current clamp':
                multiplier = 1
                baseline_value = ap_baseline_voltage
            trace = multiplier*(response_trace[ap_max_index-actual_step_back:actual_step_forward+ap_max_index]-baseline_value)/ap_amplitude
            key_apgrouptrace = key.copy()
            key_apgrouptrace['ap_group_number'] = apgroup['ap_group_number']
            key_apgrouptrace['ap_group_trace_offset']=baseline_value
            key_apgrouptrace['ap_group_trace_multiplier']=multiplier/ap_amplitude
            key_apgrouptrace['ap_group_trace_peak_idx']=actual_step_back
            key_apgrouptrace['ap_group_trace']=trace
            apgrouptrace_list.append(key_apgrouptrace)
            #trace_time = np.arange(-actual_step_back,actual_step_forward)/sample_rate
            #plt.plot(trace_time,trace)
        APGroupTrace().insert(apgrouptrace_list, allow_direct_insert=True)#imaging_gt.
        key['ap_group_trace_complete'] = 1
        self.insert1(key,skip_duplicates=True)
        
        #%%
@schema
class APDoublet(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    ap_doublet_number   : int # unique within cell
    ---
    ap_doublet_start_time : decimal(8,4) #- time of first AP in group from recording start (cell start time)
    ap_doublet_length   : double #- length in seconds
    ap_doublet_min_snr_v  : double
    ap_doublet_min_snr_dv : double
    ap_doublet_pre_isi    : double
    ap_doublet_post_isi   : double
    """
    class APDoubletMember(dj.Part):
        definition = """
        -> master
        -> ActionPotential
        """

@schema
class IngestAPDoublet(dj.Imported):
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    ap_doublet_complete : int
    """
    def make(self, key):
        key_original = key.copy()
        key_original['ap_doublet_complete'] = 1
        min_baseline_time = .1 
        max_doublet_isi = .1
        sweep_numbers = (ephys_cell_attached.Sweep()&key).fetch('sweep_number')
        apdoublet_list = list()
        doubletmember_list = list()
        ap_doublet_number = 0
        for sweep_number in sweep_numbers:
            dj.conn().ping()
            key['sweep_number'] = sweep_number
            sweep_start_time,sweep_end_time = (ephys_cell_attached.Sweep()&key).fetch1('sweep_start_time','sweep_end_time')
            sweep_end_time = float(sweep_end_time-sweep_start_time)
            sweep_start_time = float(sweep_start_time)
            ap_num,ap_max_time,ap_snr_v,ap_snr_dv = (ephys_cell_attached.Sweep()*ActionPotential()&key).fetch('ap_num','ap_max_time','ap_snr_v','ap_snr_dv')#ephysanal_cell_attached
            ap_max_time = np.asarray(ap_max_time,float)
            
            isis_pre = np.diff(np.concatenate([[0],ap_max_time]))
            isis_post = np.diff(np.concatenate([ap_max_time,[sweep_end_time]]))
            
            doublet_start_idxes = np.where(isis_pre>min_baseline_time)[0]
            doublet_end_idxes = doublet_start_idxes+1
            try:
                if doublet_end_idxes[-1]>=len(ap_max_time):
                    doublet_end_idxes = doublet_end_idxes[:-1]
            except:
                pass
            for doublet_start_idx in doublet_start_idxes:
                if any(doublet_end_idxes>=doublet_start_idx):
                    apdoublet_key = key.copy()
                    ap_doublet_number += 1
                    doublet_end_idx = doublet_end_idxes[np.argmax(doublet_end_idxes>=doublet_start_idx)]
                    ap_indices_now = np.arange(doublet_start_idx,doublet_end_idx+1)
                    apdoublet_key['ap_doublet_start_time'] = ap_max_time[doublet_start_idx] + sweep_start_time
                    apdoublet_key['ap_doublet_length'] = ap_max_time[doublet_end_idx] - ap_max_time[doublet_start_idx]
                    apdoublet_key['ap_doublet_min_snr_v'] = np.min(ap_snr_v[ap_indices_now])
                    apdoublet_key['ap_doublet_min_snr_dv'] = np.min(ap_snr_dv[ap_indices_now])
                    apdoublet_key['ap_doublet_pre_isi'] = isis_pre[doublet_start_idx]
                    apdoublet_key['ap_doublet_post_isi'] = isis_post[doublet_end_idx]    
                    apdoublet_key['ap_doublet_number'] = ap_doublet_number
                    if apdoublet_key['ap_doublet_length']<max_doublet_isi and apdoublet_key['ap_doublet_length']>0:
                        apdoublet_list.append(apdoublet_key)
                        for ap_idx in ap_indices_now:
                            doubletmember_key = key.copy()
                            doubletmember_key['ap_doublet_number'] = ap_doublet_number
                            doubletmember_key['ap_num'] = ap_num[ap_idx]
                            doubletmember_list.append(doubletmember_key)
                    else:
                        ap_doublet_number -= 1
                        
                    
         #%%               
        #with dj.conn().transaction: #inserting one cell
        #print('uploading AP doublets to datajoint: {}'.format(key))
        APDoublet().insert(apdoublet_list, allow_direct_insert=True)#ephysanal_cell_attached
        dj.conn().ping()
        APDoublet.APDoubletMember().insert(doubletmember_list, allow_direct_insert=True)      #ephysanal_cell_attached
        self.insert1(key_original,skip_duplicates=True)
 
        
        
        
        
#%%        
@schema
class CellSpikeParameters(dj.Computed):
    definition = """
    -> ephys_cell_attached.Cell
    -> ephys_cell_attached.RecordingMode
    ---
    ap_halfwidth_mean                  : double
    ap_halfwidth_median                : double
    ap_halfwidth_std                   : double
    ap_full_width_mean              : double
    ap_full_width_median            : double
    ap_full_width_std               : double
    ap_rise_time_mean               : double
    ap_rise_time_median             : double
    ap_rise_time_std                : double
    ap_peak_to_trough_time_mean        : double
    ap_peak_to_trough_time_median      : double
    ap_peak_to_trough_time_std         : double
    ap_isi_mean                        : double
    ap_isi_median                      : double
    ap_isi_std                         : double
    ap_snr_mean                        : double
    ap_snr_median                      : double
    ap_snr_std                         : double    
    ap_amplitude_mean                  : double
    ap_amplitude_median                : double
    ap_amplitude_std                   : double
    ap_integral_mean                   : double
    ap_integral_median                 : double
    ap_integral_std                    : double
    ap_ahp_amplitude_mean              : double
    ap_ahp_amplitude_median            : double
    ap_ahp_amplitude_std               : double    
    ap_quantity                        : smallint # ap num used for this analysis
    average_ap_wave                    : longblob
    average_ap_wave_time               : longblob
    ap_isi_percentiles                 : longblob
    total_ap_num                       : longblob
    ahp_freq_dependence=null           : double
    """
    def make(self, key):
        #print(key)
        min_isi = 0.1
        if key['recording_mode'] == 'current clamp': #for string purposes
            recmode = 'cc'
            max_amplitude = .005
        else:
            recmode = 'vc'
            max_amplitude = 1
        time_back = .001
        time_forward = .002
        APnum_to_use = 50
        minimum_ap_num = 5
        absolute_min_snr = 10
        
        
        snrs =(ActionPotential()&key).fetch('ap_snr_dv')#ephysanal_cell_attached.
        if len(snrs)<minimum_ap_num:
            #print('not enough aps')
            return None
        snrs_sorted = np.sort(snrs)[::-1]
        min_snr = np.max([snrs_sorted[np.min([len(snrs_sorted)-1,APnum_to_use])],absolute_min_snr])
        isi_all,ap_amplitude_all,ap_ahp_amplitude_all = (ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(absolute_min_snr)).fetch('ap_isi','ap_amplitude','ap_ahp_amplitude')
        
        ap_amplitude_all = ap_amplitude_all[~np.isnan(isi_all)]
        ap_ahp_amplitude_all = ap_ahp_amplitude_all[~np.isnan(isi_all)]
        isi_all = isi_all[~np.isnan(isi_all)]
        if len(isi_all)>minimum_ap_num:
            key['ap_isi_percentiles'] = np.percentile(isi_all,list(range(1,101,1)))
            if len(isi_all)>100:
                ahp_ratios = ap_ahp_amplitude_all/ap_amplitude_all
                try:
                    highfr_ahp_rato = np.nanmedian(ahp_ratios[(isi_all>.005)&(isi_all<.01)])
                    lowfr_ahp_rato = np.nanmedian(ahp_ratios[(isi_all>.5)&(isi_all<10)])
                    ahp_freq_dependence = lowfr_ahp_rato-highfr_ahp_rato
                    key['ahp_freq_dependence']=ahp_freq_dependence 
                except:
                    key['ahp_freq_dependence']= None
            else:
                key['ahp_freq_dependence']= None
        else:
            key['ap_isi_percentiles'] = None
            key['ahp_freq_dependence']= None
        key['total_ap_num'] = len(isi_all)
        aps_needed = (ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(min_snr)&'ap_amplitude < {}'.format(max_amplitude)&'ap_isi > {}'.format(min_isi))#ephysanal_cell_attached.
        if len(aps_needed)<minimum_ap_num:
            amplitudes_sorted = np.sort((ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(min_snr)&'ap_isi > {}'.format(min_isi)).fetch('ap_amplitude'))
            try:
                max_amplitude = amplitudes_sorted[minimum_ap_num-1]*1.01
                aps_needed = (ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(min_snr)&'ap_amplitude < {}'.format(max_amplitude)&'ap_isi > {}'.format(min_isi))#ephysanal_cell_attached.
            except: #try without the ISI restriction
                amplitudes_sorted = np.sort((ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(min_snr)).fetch('ap_amplitude'))
                try:
                    max_amplitude = amplitudes_sorted[minimum_ap_num-1]*1.01
                    aps_needed = (ephys_cell_attached.SweepMetadata()*ActionPotential()&key&'ap_snr_dv > {}'.format(min_snr)&'ap_amplitude < {}'.format(max_amplitude)&'ap_isi > {}'.format(min_isi))#ephysanal_cell_attached.
                except:
                    print('only {} ap left'.format(len(amplitudes_sorted)))
                    pass
            if len(aps_needed)<minimum_ap_num:
                #print('not enough aps')
                return None
            else:
                print('max amplitude changed to {}'.format(max_amplitude))
        ap_properties_needed = ['ap_halfwidth',
                                'ap_full_width',
                                'ap_rise_time',
                                'ap_integral',
                                'ap_amplitude',
                                'ap_ahp_amplitude',
                                'ap_isi',
                                'ap_peak_to_trough_time',
                                'ap_snr_dv']
        ap_properties_list=aps_needed.fetch(*ap_properties_needed)
        ap_data_now = dict()
        for data,data_name in zip(ap_properties_list,ap_properties_needed):
            ap_data_now[data_name] = data  
        key['ap_quantity'] = len(aps_needed)
        key['ap_halfwidth_mean'] = np.nanmean(ap_data_now['ap_halfwidth'])
        key['ap_halfwidth_median'] = np.nanmedian(ap_data_now['ap_halfwidth'])
        key['ap_halfwidth_std'] = np.nanstd(ap_data_now['ap_halfwidth'])
        key['ap_full_width_mean'] = np.nanmean(ap_data_now['ap_full_width'])
        key['ap_full_width_median'] = np.nanmedian(ap_data_now['ap_full_width'])
        key['ap_full_width_std'] = np.nanstd(ap_data_now['ap_full_width'])
        key['ap_rise_time_mean'] = np.nanmean(ap_data_now['ap_rise_time'])
        key['ap_rise_time_median'] = np.nanmedian(ap_data_now['ap_rise_time'])
        key['ap_rise_time_std'] = np.nanstd(ap_data_now['ap_rise_time'])
        key['ap_peak_to_trough_time_mean'] = np.nanmean(ap_data_now['ap_peak_to_trough_time'])
        key['ap_peak_to_trough_time_median'] = np.nanmedian(ap_data_now['ap_peak_to_trough_time'])
        key['ap_peak_to_trough_time_std'] = np.nanstd(ap_data_now['ap_peak_to_trough_time'])
        key['ap_isi_mean'] = np.nanmean(ap_data_now['ap_isi'])
        key['ap_isi_median'] = np.nanmedian(ap_data_now['ap_isi'])
        key['ap_isi_std'] = np.nanstd(ap_data_now['ap_isi'])
        key['ap_snr_mean'] = np.nanmean(ap_data_now['ap_snr_dv'])
        key['ap_snr_median'] = np.nanmedian(ap_data_now['ap_snr_dv'])
        key['ap_snr_std'] = np.nanstd(ap_data_now['ap_snr_dv'])
        key['ap_amplitude_mean'] = np.nanmean(ap_data_now['ap_amplitude'])
        key['ap_amplitude_median'] = np.nanmedian(ap_data_now['ap_amplitude'])
        key['ap_amplitude_std'] = np.nanstd(ap_data_now['ap_amplitude'])
        ap_integrals_normalized = ap_data_now['ap_integral']/ap_data_now['ap_amplitude']
        key['ap_integral_mean'] = np.nanmean(ap_integrals_normalized)
        key['ap_integral_median'] = np.nanmedian(ap_integrals_normalized)
        key['ap_integral_std'] = np.nanstd(ap_integrals_normalized)
        
        ap_ahp_amplitudes_normalized = ap_data_now['ap_ahp_amplitude']/ap_data_now['ap_amplitude']
        key['ap_ahp_amplitude_mean'] = np.nanmean(ap_ahp_amplitudes_normalized)
        key['ap_ahp_amplitude_median'] = np.nanmedian(ap_ahp_amplitudes_normalized)
        key['ap_ahp_amplitude_std'] = np.nanstd(ap_ahp_amplitudes_normalized)
        
        #% mean AP wave
        sweepneeded = ephys_cell_attached.Sweep()&aps_needed
        sample_rates,trace_sweep_numbers,traces  = (ephys_cell_attached.SweepMetadata()*ephys_cell_attached.SweepResponse()&sweepneeded).fetch('sample_rate','sweep_number','response_trace')
        if recmode == 'cc':
            sweep_numbers,ap_max_indexs,ap_amplitudes,ap_baseline_values = aps_needed.fetch('sweep_number','ap_max_index','ap_amplitude','ap_baseline_voltage')
        elif recmode == 'vc':
            sweep_numbers,ap_max_indexs,ap_amplitudes,ap_baseline_values = aps_needed.fetch('sweep_number','ap_max_index','ap_amplitude','ap_baseline_current')
            
        ap_waves = list()
        ap_wave_times = list()
        for sweep_number,ap_max_index,ap_amplitude,ap_baseline_value in zip(sweep_numbers,ap_max_indexs,ap_amplitudes,ap_baseline_values):
            trace_index = np.where(trace_sweep_numbers==sweep_number)[0][0]
            sample_rate = sample_rates[trace_index]
            step_back = int(time_back*sample_rate)
            step_forward = int(time_forward*sample_rate)
            start_idx = ap_max_index-step_back
            end_idx=ap_max_index+step_forward
            if start_idx >=0 and end_idx<len(traces[trace_index]):
                ap_waves.append((traces[trace_index][start_idx:end_idx]-ap_baseline_value)/ap_amplitude)
                ap_wave_times.append(np.arange(-step_back,step_forward,1)/sample_rate*1000)
                #break
        ap_waves = np.asarray(ap_waves)
        ap_wave_times = np.asarray(ap_wave_times)
        ap_wave_mean = np.nanmean(ap_waves,0)
        ap_wave_mean_time = np.nanmean(ap_wave_times,0)
        key['average_ap_wave'] = ap_wave_mean
        key['average_ap_wave_time'] = ap_wave_mean_time
        #key['ap_waves'] = ap_waves
        self.insert1(key,skip_duplicates=True)   


@schema
class SweepFrameTimes(dj.Computed):
    definition = """
    -> ephys_cell_attached.Sweep
    ---
    frame_idx : longblob # index of positive square pulses
    frame_sweep_time : longblob  # time of exposure relative to sweep start in seconds
    frame_time : longblob  # time of exposure relative to recording start in seconds
    """    
    def make(self, key):
        #%
        #key = {'subject_id': 471997, 'session': 1, 'cell_number': 2,'sweep_number':1}# {'subject_id': 471997, 'session': 1, 'cell_number': 1,'sweep_number':0}#{'subject_id':456462,'session':1,'cell_number':3,'sweep_number':24}
        exposure = (ephys_cell_attached.SweepImagingExposure()&key).fetch('imaging_exposure_trace')
        if len(exposure)>0:
            si = 1/(ephys_cell_attached.SweepMetadata()&key).fetch('sample_rate')[0]
            sweeptime = float((ephys_cell_attached.Sweep()&key).fetch('sweep_start_time')[0])
            exposure = np.concatenate([[0],np.diff(exposure[0])])
            peaks = signal.find_peaks(exposure,height = 1)
            peaks_idx = peaks[0]
            #%
            difi = np.diff(peaks_idx)
            #difi[-1]=1500
            median_dif = np.median(difi)
            if sum(difi>10*median_dif)>0:
                needed_idx_end = np.argmax(np.concatenate([[0],difi])>2*median_dif)
                peaks_idx=peaks_idx[:needed_idx_end]
            key['frame_idx']= peaks_idx 
            key['frame_sweep_time']= peaks_idx*si
            key['frame_time']= peaks_idx*si + sweeptime
            #%
            self.insert1(key,skip_duplicates=True)
    
    
@schema
class SquarePulse(dj.Computed):
    definition = """
    -> ephys_cell_attached.Sweep
    square_pulse_num : smallint unsigned # action potential number in sweep
    ---
    square_pulse_start_idx=null             : int unsigned # index of sq pulse start
    square_pulse_end_idx=null               : int unsigned # index of sq pulse end
    square_pulse_start_time=null            : decimal(8,4) # time of the sq pulse start relative to recording start
    square_pulse_length=null                : decimal(8,4) # length of the square pulse in seconds
    square_pulse_baseline_current=null      : float
    square_pulse_amplitude_current=null     : float #amplitude of square pulse
    square_pulse_baseline_voltage=null      : float
    square_pulse_amplitude_voltage=null     : float #amplitude of square pulse
    """
    def make(self, key):
        #%%
        #key ={'subject_id': 479571, 'session': 1, 'cell_number': 4, 'sweep_number': 4}
        sample_rate,recording_mode = (ephys_cell_attached.SweepMetadata()&key).fetch1('sample_rate','recording_mode')
        trace = (ephys_cell_attached.SweepResponse()&key).fetch1('response_trace')
        onestep = int(sample_rate*.002)
        maxlen = len(trace)-1
        keylist = list()
        #plt.plot(trace)
        #%
        print(key)
        #%%
        try:
            #%%
            #keylist = list()
            try:
                stimulus = (ephys_cell_attached.SweepStimulus()&key).fetch1('stimulus_trace')
            except: # stimulus not found
                stimulus = None
                
            if not type(stimulus) == type(None): # - there is a stimulus
                maxstimdiff = np.max(np.abs(np.diff(stimulus)))
                if recording_mode=='current clamp':
                    minstimamplitude = 25e-12
                else:
                    minstimamplitude = 2e-3
                if maxstimdiff>minstimamplitude:
                    
                    stim_filt = utils_ephys.gaussFilter(stimulus,sample_rate,sigma = .0001)
                    trace_filt = utils_ephys.gaussFilter(trace,sample_rate,sigma = .0001)
                    stim_filt_diff = np.abs(np.diff(stim_filt))
                    stim_start_idxs = list()
                    stim_end_idxs = list()
                    threshold = np.median(np.abs((stim_filt_diff))/.6745)*5#minstimamplitude/10
                    while any(stim_filt_diff>threshold)>0:
                        stim_start_idx=np.argmax(stim_filt_diff)
                        stim_start_idx_0 = stim_start_idx
                        stim_start_idx_1 = stim_start_idx
                        while stim_start_idx_0>0 and stim_filt_diff[stim_start_idx_0-1]<stim_filt_diff[stim_start_idx_0]:
                            stim_start_idx_0 -=1
                        while stim_start_idx_1<maxlen and stim_filt_diff[stim_start_idx_1+1]<stim_filt_diff[stim_start_idx_1]:
                            stim_start_idx_1 +=1
                        stim_filt_diff[stim_start_idx_0:stim_start_idx_1+1]=0
                        if stim_filt[stim_start_idx_1]-stim_filt[stim_start_idx_0]>minstimamplitude:
                            stim_start_idxs.append(stim_start_idx)
                        elif stim_filt[stim_start_idx_1]-stim_filt[stim_start_idx_0]< -1*minstimamplitude:
                            stim_end_idxs.append(stim_start_idx)                            
                    stim_start_idxs = np.sort(stim_start_idxs)
                    stim_end_idxs = np.sort(stim_end_idxs)
                    all_stim_idxs = np.asarray(np.sort(np.concatenate([stim_start_idxs,stim_end_idxs])),int)
                    for i,stim_start_idx in enumerate(all_stim_idxs):
                        if stim_start_idx in stim_start_idxs:
                            if any(stim_filt[stim_start_idx+onestep:]<=stim_filt[np.max([0,stim_start_idx-onestep])]):
                                stim_end_idx = stim_start_idx+onestep+np.argmax(stim_filt[stim_start_idx+onestep:]<=stim_filt[np.max([0,stim_start_idx-onestep])])
                                real_stim_end_idx = stim_end_idxs[np.argmin(np.abs(stim_end_idxs-stim_end_idx))]
                            else:
                                real_stim_end_idx = maxlen
                        else:
                            if any(stim_filt[stim_start_idx+onestep:]>=stim_filt[np.max([0,stim_start_idx-onestep])]):
                                stim_end_idx = stim_start_idx+onestep+np.argmax(stim_filt[stim_start_idx+onestep:]>=stim_filt[np.max([0,stim_start_idx-onestep])])
                                real_stim_end_idx = stim_start_idxs[np.argmin(np.abs(stim_start_idxs-stim_end_idx))]
                            else:
                                real_stim_end_idx = maxlen
                        
                        keynow = key.copy()
                        keynow['square_pulse_num']=i+1
                        keynow['square_pulse_start_idx']=stim_start_idx
                        keynow['square_pulse_end_idx']=real_stim_end_idx
                        keynow['square_pulse_start_time']=stim_start_idx/sample_rate
                        keynow['square_pulse_length']=(real_stim_end_idx-stim_start_idx)/sample_rate
                        baseline_idx = np.max([0,stim_start_idx-onestep])
                        pulseon_idx = np.min([maxlen,stim_start_idx+onestep])
                        if recording_mode=='current clamp':
                            b_current = stim_filt[baseline_idx]
                            d_current = stim_filt[pulseon_idx]-stim_filt[baseline_idx]
                            b_voltage = trace_filt[baseline_idx]
                            d_voltage = trace_filt[pulseon_idx]-trace_filt[baseline_idx]
                        else:
                            b_voltage = stim_filt[baseline_idx]
                            d_voltage = stim_filt[pulseon_idx]-stim_filt[baseline_idx]
                            b_current = trace_filt[baseline_idx]
                            d_current = trace_filt[pulseon_idx]-trace_filt[baseline_idx]   
                        keynow['square_pulse_amplitude_current']=d_current
                        keynow['square_pulse_amplitude_voltage']=d_voltage
                        keynow['square_pulse_baseline_current']=b_current
                        keynow['square_pulse_baseline_voltage']=b_voltage
                        keylist.append(keynow)
            elif recording_mode=='voltage clamp' and type(stimulus) == type(None): #there is no stimulus but we try to extract square pulses
                try:
                    trace_corrected,stim_idxs,stim_amplitudes = utils_ephys.remove_stim_artefacts_without_stim(trace, sample_rate)
                except:
                    print('could not extract squarepulses')
                    print(key)
                    return None
                if len(stim_idxs)>0:
                    print('stim found without stimulus trace')
                    print(key)
                    trace_filt = utils_ephys.gaussFilter(trace,sample_rate,sigma = .0001)
                    for idx, (stim_idx,stim_amplitude) in enumerate(zip(stim_idxs,stim_amplitudes)):
                        keynow = key.copy()
                        if stim_amplitude>0:
                            if any(trace_filt[stim_idx+onestep:]<=trace_filt[stim_idx-onestep]):
                                pulse_end_idx = np.argmax(trace_filt[stim_idx+onestep:]<=trace_filt[stim_idx-onestep])
                            else:
                                pulse_end_idx = maxlen
                        else:
                            if any(trace_filt[stim_idx+onestep:]>=trace_filt[stim_idx-onestep]):
                                pulse_end_idx = np.argmax(trace_filt[stim_idx+onestep:]>=trace_filt[stim_idx-onestep])
                            else:
                                pulse_end_idx = maxlen
                        keynow['square_pulse_num']=idx+1
                        keynow['square_pulse_start_idx']=stim_idx
                        keynow['square_pulse_end_idx']=pulse_end_idx
                        keynow['square_pulse_start_time']=stim_idx/sample_rate
                        keynow['square_pulse_length']=(pulse_end_idx-stim_idx)/sample_rate
                        baseline_idx = np.max([0,stim_idx-onestep])
                        pulseon_idx = np.min([maxlen,stim_idx+onestep])
                        b_voltage = None
                        d_voltage = None
                        b_current = trace_filt[baseline_idx]
                        d_current = trace_filt[pulseon_idx]-trace_filt[baseline_idx]   
                        keynow['square_pulse_amplitude_current']=d_current
                        keynow['square_pulse_amplitude_voltage']=d_voltage
                        keynow['square_pulse_baseline_current']=b_current
                        keynow['square_pulse_baseline_voltage']=b_voltage
                        keylist.append(keynow)
                        #%%
            if len(keylist)>0:
                self.insert(keylist,skip_duplicates=True)   
            else:
                key['square_pulse_num'] = 0
                key['square_pulse_start_idx'] = None
                key['square_pulse_end_idx'] = None
                key['square_pulse_start_time'] = None
                key['square_pulse_length'] = None
                key['square_pulse_amplitude_current'] = None
                key['square_pulse_amplitude_voltage'] = None
                key['square_pulse_baseline_current'] = None
                key['square_pulse_baseline_voltage'] = None     
                #%
                self.insert1(key,skip_duplicates=True)   
        except:
            print('ERROR duing insertion of squarepulse:')
            print(key)
    #
# =============================================================================
#     step = 100
#     key = {
#            'subject_id' : 453476,
#            'session':29,
#            'cell_number':1,
#            'sweep_number':3,
#            'square_pulse':0}
#     sqstart = trace[start_idx-step:start_idx+step]
#     sqend = trace[end_idx-step:end_idx+step]
#     time = np.arange(-step,step)/sr*1000
#     fig=plt.figure()
#     ax_v=fig.add_axes([0,0,.8,.8])
#     ax_v.plot(time,sqstart)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_start,vrs_start],'o')    
#     ax_v.plot(time,sqend)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_end,vrs_end],'o')    
# =============================================================================

