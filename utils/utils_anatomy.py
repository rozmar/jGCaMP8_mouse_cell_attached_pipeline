import datajoint as dj
dj.conn()
from pipeline import pipeline_tools,lab, experiment, ephys_cell_attached,ephysanal_cell_attached, imaging, imaging_gt
import os
from skimage.io import imread
import scipy.ndimage as ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tifffile as tf
from pathlib import Path
#%% merge RFP and GFP channels for ilastik pipeline
def merge_rfp_gfp_for_ilastik():
    anatomy_base_dir = '/home/rozmar/Data/Anatomy/cropped_rotated'
    output_base_dir = '/home/rozmar/Data/Anatomy/ilastik/raw'
    for dirname in os.listdir(anatomy_base_dir):
        if '.' in dirname or 'anm' not in dirname or '20x' not in dirname:
            continue
        dest_dir = os.path.join(output_base_dir,dirname)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        rfp_files = np.sort(os.listdir(os.path.join(anatomy_base_dir,dirname,'RFP')))
        fitc_files = np.sort(os.listdir(os.path.join(anatomy_base_dir,dirname,'FITC')))
        for rfp_file,fitc_file in zip(rfp_files,fitc_files):
            R = tf.imread(os.path.join(anatomy_base_dir,dirname,'RFP',rfp_file))
            G = tf.imread(os.path.join(anatomy_base_dir,dirname,'FITC',fitc_file))
            I = np.stack([R,G])
            fname = rfp_file[:rfp_file.find('RFP')]+'RFP_FITC'+rfp_file[rfp_file.find('RFP')+3:rfp_file.find('.')]+'.tif'
            tf.imsave(os.path.join(dest_dir,fname),I,imagej=True,metadata={'Composite mode': 'composite'})
            #break
        #break

#%%

def filter_anatomy_images_find_injection_sites(magnification = '10x'):
    #%%
    anatomy_base_dir = '/home/rozmar/Data/Anatomy/cropped_rotated'
    save_base_dir = '/home/rozmar/Data/Anatomy/filtered'
    anatomy_metadata = pd.read_csv("/home/rozmar/Data/Anatomy/Metadata/experiments.csv")
    sigma_gauss_microns = 50 # microns 
    minimum_horizontal_separation_microns = 250#500 for 20x
    jsonfilename = magnification+'_{}sigma.json'
    #orverwrite = True
    dirs_ = os.listdir(anatomy_base_dir)
    subject_ids = list()
    dirs =  list()
    for dir_now in dirs_: 
        if dir_now[3:9].isnumeric():
            dirs.append(dir_now)
            subject_ids.append(dir_now[3:9])
    peak_data = dict()
    fig = plt.figure(figsize = [20,20])
    for subject_id,subject_dir in zip(subject_ids,dirs):
        metadata_idx = np.argmax(anatomy_metadata['Spreadsheet ID'] == subject_dir)
        pixel_size = anatomy_metadata['pixel size of export'][metadata_idx]
        sigma_gauss = sigma_gauss_microns/pixel_size
        minimum_horizontal_separation = minimum_horizontal_separation_microns/pixel_size
        if magnification not in subject_dir:
            continue # only 20x images
    # =============================================================================
    #     if os.path.exists(os.path.join(save_base_dir,basename+str(z)+'.jpg')) and not orverwrite:
    #         continue
    # =============================================================================
        peak_data[subject_dir] = dict()
        peak_data[subject_dir]['peak_values_rfp_1'] = list()
        peak_data[subject_dir]['peak_values_fitc_1'] = list()
        peak_data[subject_dir]['peak_values_rfp_2'] = list()
        peak_data[subject_dir]['peak_values_fitc_2'] = list()
        peak_data[subject_dir]['peak_idx_x_1'] = list()
        peak_data[subject_dir]['peak_idx_y_1'] = list()
        peak_data[subject_dir]['peak_idx_x_2'] = list()
        peak_data[subject_dir]['peak_idx_y_2'] = list()
        peak_data[subject_dir]['z'] = list()
        peak_data[subject_dir]['filename'] = list()
        subject_dir_full = os.path.join(anatomy_base_dir,subject_dir)
        #channels = os.listdir(subject_dir_full)
        rfp_dir = os.path.join(subject_dir_full,'RFP')
        fitc_dir = os.path.join(subject_dir_full,'FITC')
        rfp_files = np.sort(os.listdir(rfp_dir))
        for rfp_file in rfp_files:
            print(rfp_file)
            basename = rfp_file[:rfp_file.find('RFP')]
            z = int(rfp_file[rfp_file.find('_Z_')+3:-5])
            im_rfp = imread(os.path.join(rfp_dir,rfp_file))
            im_rfp_filt = ndimage.filters.gaussian_filter(im_rfp,sigma_gauss)
            fitc_file = rfp_file[:rfp_file.find('RFP')]+'FITC'+rfp_file[rfp_file.find('RFP')+3:]
            im_fitc = imread(os.path.join(fitc_dir,fitc_file))
            im_fitc_filt = ndimage.filters.gaussian_filter(im_fitc,sigma_gauss)
            
            #%
            max_val_rfp = np.max(im_rfp_filt)
            num_objects_separated = 0
            denominator = 1.1
            while num_objects_separated<2 and denominator<5:
                threshold = max_val_rfp/denominator        
                thresholded_image = im_rfp_filt>threshold    
                labeled, num_objects = ndimage.label(thresholded_image)
                denominator += .1
                #%
                max_vals = list()
                max_idxs = list()
                max_idxs_x = list()
                max_idxs_y = list()
                for peak_num in range(1,num_objects+1):
                    max_val = np.max(im_rfp_filt[labeled==peak_num])
                    max_vals.append(max_val)
                    slice_x,slice_y = ndimage.find_objects(labeled==peak_num)[0]
                    maxidxes = np.where(im_rfp_filt[slice_x,slice_y]==np.max(im_rfp_filt[slice_x,slice_y]))
                    max_idx = np.asarray(np.mean(maxidxes,1),int)+np.asarray([slice_x.start,slice_y.start])
                    max_idxs.append(max_idx)
                    max_idxs_x.append(max_idx[0])
                    max_idxs_y.append(max_idx[1])
                num_objects_separated = num_objects
                if np.max(max_idxs_y)-np.min(max_idxs_y)<minimum_horizontal_separation:
                    num_objects_separated = 1
    
                    #%
            max_vals = list()
            max_idxs = list()
            max_idxs_x = list()
            max_idxs_y = list()
            for peak_num in range(1,num_objects+1):
                max_val = np.max(im_rfp_filt[labeled==peak_num])
                
                slice_x,slice_y = ndimage.find_objects(labeled==peak_num)[0]
                maxidxes = np.where(im_rfp_filt[slice_x,slice_y]==np.max(im_rfp_filt[slice_x,slice_y]))
                max_idx = np.asarray(np.mean(maxidxes,1),int)+np.asarray([slice_x.start,slice_y.start])
                
                if len(max_idxs_y)==0 or np.abs(max_idxs_y[0]-max_idx[1])>minimum_horizontal_separation:
                    max_vals.append(max_val)
                    max_idxs.append(max_idx)
                    max_idxs_x.append(max_idx[0])
                    max_idxs_y.append(max_idx[1])
            order = np.argsort(max_vals)[::-1]
            max_idxs = np.asarray(max_idxs)[order]
            max_vals = np.asarray(max_vals)[order]
            for peak_num, (max_idx, max_val) in enumerate(zip(max_idxs,max_vals)):
                peak_num+=1
                if peak_num>2:
                    break
                peak_data[subject_dir]['peak_values_rfp_{}'.format(peak_num)].append(float(im_rfp_filt[max_idx[0],max_idx[1]]))
                peak_data[subject_dir]['peak_values_fitc_{}'.format(peak_num)].append(float(im_fitc_filt[max_idx[0],max_idx[1]]))
                peak_data[subject_dir]['peak_idx_x_{}'.format(peak_num)].append(float(max_idx[0]))
                peak_data[subject_dir]['peak_idx_y_{}'.format(peak_num)].append(float(max_idx[1]))
                if peak_num ==1:
                    peak_data[subject_dir]['z'].append(z)
                    peak_data[subject_dir]['filename'].append(str(rfp_file))
                print('peak_value: {}'.format(max_val))
                #break
            if peak_num==1:
                peak_data[subject_dir]['peak_values_rfp_{}'.format(2)].append(np.nan)
                peak_data[subject_dir]['peak_values_fitc_{}'.format(2)].append(np.nan)
                peak_data[subject_dir]['peak_idx_x_{}'.format(2)].append(np.nan)
                peak_data[subject_dir]['peak_idx_y_{}'.format(2)].append(np.nan)
    
            #%
            fig.clear()
            ax_rfp=fig.add_subplot(2,2,1)
            ax_fitc=fig.add_subplot(2,2,2)
            ax_rfp_filt=fig.add_subplot(2,2,3)
            ax_fitc_filt=fig.add_subplot(2,2,4)
            ax_rfp.set_title('anti-GFP')
            ax_fitc.set_title('native fluorescence')
            
            
            ax_rfp.imshow(im_rfp)
            ax_rfp_filt.imshow(im_rfp_filt)
            ax_fitc.imshow(im_fitc)
            ax_fitc_filt.imshow(im_fitc_filt)   
            for max_idx in max_idxs:
                ax_rfp_filt.plot(max_idx[1],max_idx[0],'ro')
                ax_fitc_filt.plot(max_idx[1],max_idx[0],'ro')
            fig.savefig(os.path.join(save_base_dir,basename+str(z)+'.jpg'))
    #%
    #%
    files = peak_data.keys()
    for file in files:
        variables = peak_data[file].keys()
        for variable in variables:
            try:
                float(peak_data[file][variable][0])
                peak_data[file][variable] = np.asarray(peak_data[file][variable],float)
            except:
                peak_data[file][variable] = np.asarray(peak_data[file][variable],str)
            peak_data[file][variable] = list(peak_data[file][variable])
        
    
    #%  5  
            
    import json        
    jsonfile = os.path.join(save_base_dir,jsonfilename.format(sigma_gauss_microns))
    with open(jsonfile, 'w') as outfile:
        json.dump(peak_data, outfile, indent=4)        
    #%% 
        
def digest_anatomy_json_file(magnification = '10x',plotit = True):
    #%%
    jsonfile = os.path.join(dj.config['locations.mr_share'],'Data/Anatomy/filtered/{}_50sigma.json'.format(magnification))
    with open(jsonfile, "r") as read_file:
        peak_data = json.load(read_file)
    
    minimum_horizontal_separation_microns = 400
    if plotit:
        fig_peaks = plt.figure()
    
    injection_site_dict = {'sensor':list(),
                           'subject':list(),
                           'X':list(),
                           'Y':list(),
                           'Z':list(),
                           'filename':list(),
                           'RFP':list(),
                           'FITC':list()}
    for sample_i,sample in enumerate(peak_data.keys()):
        
        subject = sample[3:9]
        if subject =='479410':
            subject = '478410' # stupid typo
        sensor = (imaging_gt.SessionCalciumSensor()&'subject_id = {}'.format(subject)).fetch1('session_calcium_sensor')
        ml,ap = (lab.Surgery.VirusInjection()&'subject_id = {}'.format(subject)).fetch('ml_location','ap_location')
        injection_number = np.unique(np.asarray([ml,ap],float),axis = 1).shape[1]
        minimum_horizontal_separation_microns = float(np.mean(np.diff(np.unique(ap))))*.6
        #%
        peak_data_now = peak_data[sample]
        
        horizontal_diff = np.asarray(peak_data_now['peak_idx_y_1']) - np.asarray(peak_data_now['peak_idx_y_2'])
        inverse_order = horizontal_diff<0 # first order the points based on their relative location
    
        peak_data_new = {}
        for k in peak_data_now.keys(): peak_data_new[k] = list()
        
        for to_invert,peak_values_rfp_1, peak_values_fitc_1, peak_values_rfp_2, peak_values_fitc_2, peak_idx_x_1, peak_idx_y_1, peak_idx_x_2, peak_idx_y_2, z, filename in zip(inverse_order,peak_data_now['peak_values_rfp_1'], peak_data_now['peak_values_fitc_1'], peak_data_now['peak_values_rfp_2'], peak_data_now['peak_values_fitc_2'], peak_data_now['peak_idx_x_1'], peak_data_now['peak_idx_y_1'], peak_data_now['peak_idx_x_2'], peak_data_now['peak_idx_y_2'], peak_data_now['z'], peak_data_now['filename']):
            if to_invert:
                peak_data_new['peak_values_rfp_1'].append(peak_values_rfp_2)
                peak_data_new['peak_values_fitc_1'].append(peak_values_fitc_2)            
                peak_data_new['peak_idx_x_1'].append(peak_idx_x_2)
                peak_data_new['peak_idx_y_1'].append(peak_idx_y_2)
                peak_data_new['peak_values_rfp_2'].append(peak_values_rfp_1)
                peak_data_new['peak_values_fitc_2'].append(peak_values_fitc_1)            
                peak_data_new['peak_idx_x_2'].append(peak_idx_x_1)
                peak_data_new['peak_idx_y_2'].append(peak_idx_y_1)
    
            else:
                peak_data_new['peak_values_rfp_1'].append(peak_values_rfp_1)
                peak_data_new['peak_values_fitc_1'].append(peak_values_fitc_1)            
                peak_data_new['peak_idx_x_1'].append(peak_idx_x_1)
                peak_data_new['peak_idx_y_1'].append(peak_idx_y_1)
                peak_data_new['peak_values_rfp_2'].append(peak_values_rfp_2)
                peak_data_new['peak_values_fitc_2'].append(peak_values_fitc_2)            
                peak_data_new['peak_idx_x_2'].append(peak_idx_x_2)
                peak_data_new['peak_idx_y_2'].append(peak_idx_y_2)
            peak_data_new['filename'].append(filename)
            peak_data_new['z'].append(z)
        for k in peak_data_new.keys():
            peak_data_new[k] = np.asarray(peak_data_new[k])
        #
        if plotit:
            if sample_i ==0:
                ax_now = fig_peaks.add_subplot(3,4,sample_i+1)
                ax_now_twin=ax_now.twinx()
                ax_first = ax_now
                #ax_first_twin = ax_now_twin
            else:
                ax_now = fig_peaks.add_subplot(3,4,sample_i+1,sharey = ax_first)
                ax_now_twin=ax_now.twinx()
                #plt.axes(ax_now_twin, sharey = ax_first_twin)
            ax_now.plot( peak_data_new['z'],peak_data_new['peak_values_rfp_1'],'r-')
            ax_now.plot( peak_data_new['z'],peak_data_new['peak_values_rfp_2'],'k-')
            ax_now_twin.plot( peak_data_new['z'],peak_data_new['peak_values_fitc_1'],'g-')
            ax_now_twin.plot( peak_data_new['z'],peak_data_new['peak_values_fitc_2'],'b-')
        
        
        rfp_concatenated = np.concatenate([peak_data_new['peak_values_rfp_1'],peak_data_new['peak_values_rfp_2']])
        z_concatenated = np.concatenate([peak_data_new['z'],peak_data_new['z']+20000])
        side_concatenated = np.asarray(np.concatenate([np.zeros(len(peak_data_new['peak_values_rfp_1']))+1,np.zeros(len(peak_data_new['peak_values_rfp_2']))+2]),int)
        
        for peak_num in range(injection_number):
            #maxval=np.nanmax(rfp_concatenated)
            max_idx = np.nanargmax(rfp_concatenated)
            
            side = side_concatenated[max_idx]
            if side == 1:
                idx_real = max_idx
            else:
                idx_real = max_idx-len(peak_data_new['z'])
            
            injection_site_dict['sensor'].append(sensor)
            injection_site_dict['subject'].append(subject)
            injection_site_dict['X'].append(peak_data_new['peak_idx_x_{}'.format(side)][idx_real])
            injection_site_dict['Y'].append(peak_data_new['peak_idx_y_{}'.format(side)][idx_real])
            injection_site_dict['Z'].append(peak_data_new['z'][idx_real])
            injection_site_dict['RFP'].append(peak_data_new['peak_values_rfp_{}'.format(side)][idx_real])
            injection_site_dict['FITC'].append(peak_data_new['peak_values_fitc_{}'.format(side)][idx_real])
            injection_site_dict['filename'].append(peak_data_new['filename'][idx_real])
            if plotit:
                ax_now.plot( peak_data_new['z'][idx_real],peak_data_new['peak_values_rfp_{}'.format(side)][idx_real],'ro')
            
            
            todel = (z_concatenated>=z_concatenated[max_idx]-minimum_horizontal_separation_microns) & (z_concatenated<=z_concatenated[max_idx]+minimum_horizontal_separation_microns)
            rfp_concatenated[todel]=0
            
        
        if plotit:    
            ax_now.set_title(sample)       
    
        print('{}- {} injections, minimum separation enforced: {} microns'.format(subject,injection_number,minimum_horizontal_separation_microns))
      #%%   
    return injection_site_dict
    
def export_in_vivo_brightnesses():
#%% ROI fluorescence for ALL ROIs
    sensors = ['GCaMP7F','XCaMPgf','456','686','688']
    channel = 1
    neuropil_number = 1
    roi_name = 'roi_f_min'#'neuropil_f_min'#
    correct_depth = False
    f_corrs = list()
    f_corrs_norm = list()
    depths = list()
    sensor_f_corrs = dict()
    sensor_f_corrs_norm = dict()
    sensor_neuropil_to_roi = dict()
    sensor_expression_time = dict()
    subject_median_roi_values = dict()
    subject_expression_times = dict()
    sensr_depths = dict()
    for sensor in sensors:
        #sensor = '686'
        sensr_depths[sensor]=list()
        sensor_f_corrs[sensor]=list()
        sensor_f_corrs_norm[sensor]=list()
        subject_median_roi_values[sensor] = list()
        subject_expression_times[sensor] = list()
        sensor_neuropil_to_roi[sensor]=list()
        sensor_expression_time[sensor]=list()
        sessions = experiment.Session()*imaging_gt.SessionCalciumSensor()&'session_calcium_sensor = "{}"'.format(sensor)
        for i,session in enumerate(sessions):
            rois = imaging.ROINeuropilTrace()*imaging_gt.SessionCalciumSensor*imaging_gt.MovieDepth()*imaging.MovieMetaData*imaging.MoviePowerPostObjective()*imaging.ROITrace()&session&'channel_number = {}'.format(channel) & 'neuropil_number = {}'.format(neuropil_number)
            f,power,depth,session_sensor_expression_time,roi_f_min,neuropil_f_min = rois.fetch(roi_name,'movie_power','movie_depth','session_sensor_expression_time','roi_f_min','neuropil_f_min')#
            power = np.asarray(power,float)
            f = np.asarray(f,float)
            depth = np.asarray(depth,float)
            depth[depth<0]=100
            roi_f_min = np.asarray(roi_f_min,float)
            neuropil_f_min = np.asarray(neuropil_f_min,float)
            #f = roi_f_min-0.8*neuropil_f_min#roi_f_min/neuropil_f_min# 
            if correct_depth:
                f_corr = f/(power**2)*40**2+.004*f*depth#*np.exp(depth/200)
                neuropil_f_min_corr = neuropil_f_min/(power**2)*40**2+.004*neuropil_f_min*depth
            else:
                f_corr = f/(power**2)*40**2
                neuropil_f_min_corr = neuropil_f_min/(power**2)*40**2#+.004*neuropil_f_min_corr*depth
            f_corr_norm = (f_corr-np.nanmedian(f_corr))/np.diff(np.percentile(f_corr,[25,75]))#np.nanstd(f_corr)#
            if min(depth)>0:
                sensr_depths[sensor].append(depth)
                depths.append(depth)
                f_corrs.append(f_corr)
                sensor_f_corrs[sensor].append(f_corr)
                f_corrs_norm.append(f_corr_norm)
                sensor_f_corrs_norm[sensor].append(f_corr_norm)
                subject_median_roi_values[sensor].append(np.nanmedian(f_corr))
                subject_expression_times[sensor].append(np.nanmedian(session_sensor_expression_time))
                sensor_neuropil_to_roi[sensor].append(neuropil_f_min_corr/f_corr)      
                sensor_expression_time[sensor].append(session_sensor_expression_time)
    return sensor_f_corrs

def export_ilastik_data():
    sensor_translator = {'XCaMP':'XCaMPgf', 
                         'UF456':'456', 
                         'G7F':'GCaMP7F', 
                         'UF686':'686', 
                         'UF688':'688'}
    ilastik_batch_dir = '/home/rozmar/Data/Anatomy/ilastik/raw/batch'
    anatomy_metadata = pd.read_csv("/home/rozmar/Data/Anatomy/Metadata/experiments.csv")
    injection_site_dict = digest_anatomy_json_file(plotit = False)
    ilastik_data_all = []
    slice_id = 0
    slice_data_all = []
    for brain in anatomy_metadata.iterrows():
        brain = brain[1]
        if '20x' not in brain['Spreadsheet ID']:
            continue
        brain_metadata = pd.read_csv("/home/rozmar/Data/Anatomy/Metadata/{}.csv".format(brain['Spreadsheet ID']))
        injection_idx = (np.asarray(injection_site_dict['subject'],int)==brain['ID'])
        injection_Z = np.asarray(injection_site_dict['Z'])[injection_idx]
        for section in brain_metadata.iterrows():
            section = section[1]
            
            #%
            unique_inj_zs = np.unique(injection_Z)
            while any(np.diff(unique_inj_zs)<150):
                idx_now = np.argmax(np.diff(unique_inj_zs)<150)
                unique_inj_zs[idx_now:idx_now+2] = int((np.mean(unique_inj_zs[idx_now:idx_now+2]))/50)*50
                unique_inj_zs = np.unique(unique_inj_zs)
            z_to_injection = np.floor((np.min(np.abs(unique_inj_zs - section['Z location'])))/100)*100+50
            object_filename = brain['Spreadsheet ID'] +'_RFP_FITC_Z_{}_table.csv'.format(str(int(section['Z location'])).zfill(6))
            ilastik_data = pd.read_csv(os.path.join(ilastik_batch_dir,'objects',object_filename))    
            slice_id+=1
            if len(ilastik_data)>0:
                ilastik_data['sensor'] = sensor_translator[brain['sensor']]
                ilastik_data['z_to_injection'] = z_to_injection
                ilastik_data['section_z'] = str(int(section['Z location'])).zfill(6)
                ilastik_data['exp_name'] = brain['Spreadsheet ID']
                ilastik_data['slice_id'] = slice_id
                if len(ilastik_data_all) == 0:
                    ilastik_data_all = ilastik_data
                    
                else:
                    ilastik_data_all = pd.concat([ilastik_data_all,ilastik_data], ignore_index=True)
                    #break
                good_cell_num = (ilastik_data['Predicted Class']=='Good Cell').sum()+(ilastik_data['Predicted Class']=='2 good cells').sum()*2+(ilastik_data['Predicted Class']=='2 mixed cells').sum() + (ilastik_data['Predicted Class']=='3+ good cells').sum()*3
                filled_cell_num = (ilastik_data['Predicted Class']=='Filled').sum()+(ilastik_data['Predicted Class']=='2 filled cells').sum()*2+(ilastik_data['Predicted Class']=='2 mixed cells').sum()
                dict_now = {'sensor':sensor_translator[brain['sensor']],
                            'z_to_injection': z_to_injection,
                            'good_cell_num':good_cell_num,
                            'filled_cell_num':filled_cell_num,
                            'cell_num':filled_cell_num+good_cell_num,
                            'filled_cell_ratio':filled_cell_num/(filled_cell_num+good_cell_num),
                            'brain_name':brain['Spreadsheet ID'],
                            'section_z':str(int(section['Z location'])).zfill(6),
                            }
                slice_data_now = pd.Series(dict_now)
                if len(slice_data_all) == 0:
                    slice_data_all = slice_data_now
                else:
                    slice_data_all = pd.concat([slice_data_all,slice_data_now], axis=1, ignore_index=True)
                    
    slice_data_all = slice_data_all.transpose() 
    return ilastik_data_all,slice_data_all