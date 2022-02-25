import os 
from ScanImageTiffReader import ScanImageTiffReader
from pathlib import Path
import json
import numpy as np
from suite2p import default_ops as s2p_default_ops
from suite2p import run_s2p
from suite2p import classification
try:
    from suite2p.detection.masks import create_neuropil_masks, create_cell_pix, create_cell_mask
    from suite2p.extraction.extract import extract_traces
except:
    print('wrong s2p version, skipping some functions')
import shutil
import time
import datetime
from scipy.io import loadmat
#%%

def processVisMat(vis_mat_path):
    #%%
    """
    processVisMat
        load visual stim file and get relevant parameters
    
    @input:
        vis_mat_path: path to vis stim mat file
    
    @outputs:
        vis
            ncondition: 8
            ntrial: 5
            fs: 122
            stim_dur: 2
            gray_dur: 2
            total_frames: 19520
            total_time: 160
            stim_init: [1×40 double]
            numStim: 40
            angle: [1×40 double]
            trials: {1×40 cell}
    """
    mat = loadmat(vis_mat_path)['vis'][0][0]
    
    vis = {}
    vis['ncondition'] = mat[0][0][0]
    vis['ntrial'] = mat[1][0][0]
    vis['fs'] = mat[2][0][0]
    vis['stim_dur'] = mat[3][0][0]
    vis['gray_dur'] = mat[4][0][0]
    vis['total_frames'] = mat[5][0][0]
    vis['total_time'] = mat[6][0][0]
    vis['stim_init'] = mat[7][0]
    vis['numStim'] = mat[8][0][0]
    vis['angle'] = mat[9][0]
    vis['trials'] = mat[10]
    vis['trials_list'] = dict()
    keys = vis['trials'][0][0].dtype.fields.keys()
    for key in keys:
        vis['trials_list'][key]=list()
        for trial in vis['trials'][0]:
            vis['trials_list'][key].append(trial[key][0][0][0][0])
            
        
        
    #%
    return vis

def extract_scanimage_metadata(file):
    #%%
    image = ScanImageTiffReader(file)
    metadata_raw = image.metadata()
    description_first_image = image.description(0)
    description_first_image_dict = dict(item.split(' = ') for item in description_first_image.rstrip(r'\n ').rstrip('\n').split('\n'))
    metadata_str = metadata_raw.split('\n\n')[0]
    metadata_json = metadata_raw.split('\n\n')[1]
    metadata_dict = dict(item.split('=') for item in metadata_str.split('\n') if 'SI.' in item)
    metadata = {k.strip().replace('SI.','') : v.strip() for k, v in metadata_dict.items()}
    for k in list(metadata.keys()):
        if '.' in k:
            ks = k.split('.')
            # TODO just recursively create dict from .-containing values
            if k.count('.') == 1:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') == 2:
                if not ks[0] in metadata.keys():
                    metadata[ks[0]] = {}
                    metadata[ks[0]][ks[1]] = {}
                elif not ks[1] in metadata[ks[0]].keys():
                    metadata[ks[0]][ks[1]] = {}
                metadata[ks[0]][ks[1]] = metadata[k]
            elif k.count('.') > 2:
                print('skipped metadata key ' + k + ' to minimize recursion in dict')
            metadata.pop(k)
    metadata['json'] = json.loads(metadata_json)
    frame_rate = metadata['hRoiManager']['scanVolumeRate']
    try:
        z_collection = metadata['hFastZ']['userZs']
        num_planes = len(z_collection)
    except: # new scanimage version
        if metadata['hFastZ']['hasFastZ'] == 'true':
            print('multiple planes not handled in metadata collection.. HANDLE ME!!!')
            time.sleep(1000)
        else:
            num_planes = 1
    
    roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
    
    
    if type(roi_metadata) == dict:
        roi_metadata = [roi_metadata]
    num_rois = len(roi_metadata)
    roi = {}
    w_px = []
    h_px = []
    cXY = []
    szXY = []
    for r in range(num_rois):
        roi[r] = {}
        roi[r]['w_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][0]
        w_px.append(roi[r]['w_px'])
        roi[r]['h_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][1]
        h_px.append(roi[r]['h_px'])
        roi[r]['center'] = roi_metadata[r]['scanfields']['centerXY']
        cXY.append(roi[r]['center'])
        roi[r]['size'] = roi_metadata[r]['scanfields']['sizeXY']
        szXY.append(roi[r]['size'])
        #print('{} {} {}'.format(roi[r]['w_px'], roi[r]['h_px'], roi[r]['size']))
    
    w_px = np.asarray(w_px)
    h_px = np.asarray(h_px)
    szXY = np.asarray(szXY)
    cXY = np.asarray(cXY)
    cXY = cXY - szXY / 2
    cXY = cXY - np.amin(cXY, axis=0)
    mu = np.median(np.transpose(np.asarray([w_px, h_px])) / szXY, axis=0)
    imin = cXY * mu
    
    n_rows_sum = np.sum(h_px)
    n_flyback = (image.shape()[1] - n_rows_sum) / np.max([1, num_rois - 1])
    
    irow = np.insert(np.cumsum(np.transpose(h_px) + n_flyback), 0, 0)
    irow = np.delete(irow, -1)
    irow = np.vstack((irow, irow + np.transpose(h_px)))
    
    data = {}
    data['fs'] = frame_rate
    data['nplanes'] = num_planes
    data['nrois'] = num_rois #or irow.shape[1]?
    if data['nrois'] == 1:
        data['mesoscan'] = 0
    else:
        data['mesoscan'] = 1
    
    if data['mesoscan']:
        #data['nrois'] = num_rois #or irow.shape[1]?
        data['dx'] = []
        data['dy'] = []
        data['lines'] = []
        for i in range(num_rois):
            data['dx'] = np.hstack((data['dx'], imin[i,1]))
            data['dy'] = np.hstack((data['dy'], imin[i,0]))
            data['lines'] = list(range(irow[0,i].astype('int32'), irow[1,i].astype('int32') - 1)) ### TODO NOT QUITE RIGHT YET
        data['dx'] = data['dx'].astype('int32')
        data['dy'] = data['dy'].astype('int32')
        print(data['dx'])
        print(data['dy'])
        print(data['lines'])
            #data['lines']{i} = 
            #data.dx(i) = int32(imin(i,2));
            #data.dy(i) = int32(imin(i,1));
            #data.lines{i} = irow(1,i):(irow(2,i)-1)
    movie_start_time = datetime.datetime.strptime(description_first_image_dict['epoch'].rstrip(']').lstrip('['), '%Y %m %d %H %M %S.%f')
    out = {'metadata':metadata,
           'roidata':data,
           'roi_metadata':roi_metadata,
           'frame_rate':frame_rate,
           'num_planes':num_planes,
           'shape':image.shape(),
           'description_first_frame':description_first_image_dict,
           'movie_start_time': movie_start_time}
    #%%
    return out
#%%
def extract_files_from_dir(basedir):
    files = os.listdir(basedir)
    exts = list()
    basenames = list()
    fileindexs = list()
    dirs = list()
    for file in files:
        if Path(os.path.join(basedir,file)).is_dir():
            exts.append('')
            basenames.append('')
            fileindexs.append(np.nan)
            dirs.append(file)
        else:
            if '_' in file and ('cell' in file.lower() or 'stim' in file.lower()):
                basenames.append(file[:-1*file[::-1].find('_')-1])
                try:
                    fileindexs.append(int(file[-1*file[::-1].find('_'):file.find('.')]))
                except:
                    print(file)
                    fileindexs.append(int(file[-1*file[::-1].find('_'):file.find('.')]))
            else:
                basenames.append(file[:file.find('.')])
                fileindexs.append(-1)
            exts.append(file[file.find('.'):])
    tokeep = np.asarray(exts) != ''
    files = np.asarray(files)[tokeep]
    exts = np.asarray(exts)[tokeep]
    basenames = np.asarray(basenames)[tokeep]
    fileindexs = np.asarray(fileindexs)[tokeep]
    out = {'dir':basedir,
           'filenames':files,
           'exts':exts,
           'basenames':basenames,
           'fileindices':fileindexs,
           'dirs':dirs
           }
    return out
#%%
def batch_run_suite2p(s2p_params):
    #%%
    sessions = os.listdir(s2p_params['basedir'])
    sessions.sort()
    for session in sessions:
        #%
       # session = '2021-07-01-anm483859-interneurons'
        session_dir = os.path.join(s2p_params['basedir'],session)
        if not os.path.isdir(session_dir):
            continue

        files_dict = extract_files_from_dir(session_dir)
        for dir_now in files_dict['dirs']:
            if 'stack' not in dir_now and 'suite2p' not in dir_now:
                cell_dir = os.path.join(session_dir,dir_now)
                #%%
# =============================================================================
#             for cell_dir in ['/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/20200731-anm472181',#179 for visualstimonly experiments
#                              '/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/20200704-anm478411',
#                              '/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/20200623-anm478404',
#                              '/home/rozmar/Data/Calcium_imaging/raw/Genie_2P_rig/20200623-anm478343']:
# =============================================================================
                files_dict = extract_files_from_dir(cell_dir)
                savedir = s2p_params['basedir_out'] + files_dict['dir'][len(s2p_params['basedir']):]
                savedir_slow = s2p_params['basedir_out_slow'] + files_dict['dir'][len(s2p_params['basedir']):]
                uniquebasenames = np.unique(files_dict['basenames'][files_dict['exts']=='.tif'])
                for uniquebasename in uniquebasenames:
                    print(uniquebasename)
                    #%
                    save_dir_now = os.path.join(savedir,uniquebasename)
                    save_dir_slow_now = os.path.join(savedir_slow,uniquebasename)
                    os.makedirs(save_dir_now, exist_ok=True)
                    os.makedirs(save_dir_slow_now, exist_ok=True)
                    files_already_there = os.listdir(save_dir_now)
                    files_already_there_slow = os.listdir(save_dir_slow_now)
                    if len(files_already_there)>1 or len(files_already_there_slow)>1 or s2p_params['overwrite']:
                        print('{} - {} - {} already done'.format(session,dir_now, uniquebasename))
                        continue
                    if 'stack' in uniquebasename.lower():
                        print('{} is a stacks, skipped'.format(uniquebasename))
                        continue
                        #time.sleep(10000)
                        
                    file_idxs = (files_dict['basenames'] == uniquebasename) & (files_dict['exts']=='.tif')
                    fnames = files_dict['filenames'][file_idxs]
                    file_indices = files_dict['fileindices'][file_idxs]
                    order  = np.argsort(file_indices)
                    fnames = fnames[order]
                    file_indices = file_indices[order]
                    metadata = extract_scanimage_metadata(os.path.join(files_dict['dir'],fnames[0]))
                    pixelsize = metadata['roi_metadata'][0]['scanfields']['sizeXY']
                    movie_dims = metadata['roi_metadata'][0]['scanfields']['pixelResolutionXY']
                    zoomfactor = int(metadata['metadata']['hRoiManager']['scanZoomFactor'])
                    
                    
                    XFOV = 1500*np.exp(-zoomfactor/11.5)+88 # HOTFIX for 16x objective
                    #pixelsize_real =  XFOV/movie_dims[0]
                    
                    FOV = np.min(pixelsize)*np.asarray(movie_dims)
                    ops = s2p_default_ops()#run_s2p.default_ops()
                    
                    print('zoom: {}x, pixelsizes: {} micron, dims: {} pixel, FOV: {} microns, estimated XFOV: {}'.format(zoomfactor, pixelsize,movie_dims,FOV,XFOV))
                    ops['reg_tif'] = True # save registered movie as tif files
                    #ops['num_workers'] = 8
                    ops['delete_bin'] = 0 
                    ops['keep_movie_raw'] = 0
                    ops['fs'] = float(metadata['frame_rate'])
                    #ops['nchannels'] = int(metadata['metadata']['hChannels']['channelsAvailable']) #TODO this is not good! doesn't recognize single channel movies
                    if '[' in metadata['metadata']['hChannels']['channelSave']:
                        ops['nchannels'] = 2
                    else:
                        ops['nchannels'] = 1
                    ops['save_path'] = save_dir_now
                    ops['fast_disk'] = save_dir_now
                    ops['diameter'] = np.ceil(s2p_params['cell_soma_size']/np.min(pixelsize))#pixelsize_real
                    ops['tau'] = .6
                    ops['maxregshift'] =  s2p_params['max_reg_shift']/np.max(FOV)
                    ops['nimg_init'] = 1200
                    ops['nonrigid'] = True
                    ops['maxregshiftNR'] = int(s2p_params['max_reg_shift_NR']/np.min(pixelsize)) # this one is in pixels...
                    block_size_optimal = np.round((s2p_params['block_size']/np.min(pixelsize)))
                    potential_bases = np.asarray([2**np.floor(np.log(block_size_optimal)/np.log(2)),2**np.ceil(np.log(block_size_optimal)/np.log(2)),3**np.floor(np.log(block_size_optimal)/np.log(3)),3**np.ceil(np.log(block_size_optimal)/np.log(3))])
                    block_size = int(potential_bases[np.argmin(np.abs(potential_bases-block_size_optimal))])
                    ops['block_size'] = np.ones(2,int)*block_size
                    ops['smooth_sigma'] = s2p_params['smooth_sigma']/np.min(pixelsize)#pixelsize_real #ops['diameter']/10 #
                    ops['smooth_sigma_time'] = s2p_params['smooth_sigma_time']*float(metadata['frame_rate']) # ops['tau']*ops['fs']#
                    ops['data_path'] = [files_dict['dir']]
                    ops['tiff_list'] = fnames
                    
     
                    print(ops['data_path'])
                    
                    #%
                    try:
                        run_s2p(ops)
                    except:
                        print('error in suite2p')
                    #%
                    files_to_move = os.listdir(os.path.join(ops['data_path'][0],'suite2p'))
                    for file_now in files_to_move:
                        shutil.move(os.path.join(ops['data_path'][0],'suite2p',file_now),os.path.join(savedir,uniquebasename,file_now))

                    planes = os.listdir(os.path.join(savedir,uniquebasename,'suite2p'))
                    for plane in planes:
                        rawfiles = os.listdir(os.path.join(savedir,uniquebasename,'suite2p',plane))
                        for rawfile in rawfiles:
                            shutil.move(os.path.join(savedir,uniquebasename,'suite2p',plane,rawfile),os.path.join(savedir,uniquebasename,plane,rawfile))
                    os.rmdir(os.path.join(ops['data_path'][0],'suite2p'))
                    
#%%
                    
                    
                    
                    
                    
#%%
def suite2p_export_ROIs_with_overlap(s2p_params):
    #%%
    
    sessions = os.listdir(s2p_params['basedir_out'])
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(s2p_params['basedir_out'],session)
        if not os.path.isdir(session_dir):
            continue
        
        cells = os.listdir(session_dir)
        for cell in cells:
            cell_dir = os.path.join(session_dir,cell)
            movies = os.listdir(cell_dir)
            for movie in movies:
                #t0 = time.time()
                movie_dir = os.path.join(cell_dir,movie,'plane0')
                #%
                print('extracting roi from {}'.format(movie_dir))
                try:
                    iscell = np.load(os.path.join(movie_dir,'iscell.npy'))
                    roi_stat = list(np.load(os.path.join(movie_dir,'stat.npy'),allow_pickle=True))
                    reg_metadata = np.load(os.path.join(movie_dir,'ops.npy'),allow_pickle = True).tolist()
                except:
                    print('missing or corrupt suite2p files, skipping {}'.format(movie))
                    continue
                if reg_metadata['allow_overlap']:
                    print('overlap is already enabled')
                    continue
                cell_masks = list()
                for roi_stat_now in roi_stat:
                    cell_mask = create_cell_mask(roi_stat_now,Ly=reg_metadata['Ly'], Lx=reg_metadata['Lx'], allow_overlap=True)
                    #break
                    cell_masks.append(cell_mask)
                cell_pix = create_cell_pix(roi_stat, Ly=reg_metadata['Ly'], Lx=reg_metadata['Lx'], allow_overlap=reg_metadata['allow_overlap'])
                neuropil_mask = create_neuropil_masks(ypixs=[stat['ypix'] for stat in roi_stat],
                                                      xpixs=[stat['xpix'] for stat in roi_stat],
                                                      cell_pix=cell_pix,
                                                      inner_neuropil_radius=reg_metadata['inner_neuropil_radius'],
                                                      min_neuropil_pixels=reg_metadata['min_neuropil_pixels'])
                
                reg_file  = os.path.join(movie_dir,'data.bin')
                reg_metadata['allow_overlap'] = True
                F,Fneu,ops = extract_traces(reg_metadata, cell_masks, neuropil_mask, reg_file)  
                #%
                np.save(os.path.join(movie_dir,'F_allow_orverlap.npy'),F,allow_pickle = True)
                reg_file_chan2  = os.path.join(movie_dir,'data_chan2.bin')
                F_chan2,Fneu_chan2,ops = extract_traces(reg_metadata, cell_masks, neuropil_mask, reg_file_chan2)    
                np.save(os.path.join(movie_dir,'F_chan2_allow_orverlap.npy'),F_chan2,allow_pickle = True)
# =============================================================================
#                 break
#             break
#         break
# =============================================================================
    
              
 #%%                   
                    
                    
                    
def suite2p_extract_altered_neuropil(s2p_params,neuropil_pixel_nums = [4,8,16]):
    #%
    
    sessions = os.listdir(s2p_params['basedir_out'])
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(s2p_params['basedir_out'],session)
        if not os.path.isdir(session_dir):
            continue
        
        cells = os.listdir(session_dir)
        for cell in cells:
            cell_dir = os.path.join(session_dir,cell)
            movies = os.listdir(cell_dir)
            for movie in movies:
                t0 = time.time()
                movie_dir = os.path.join(cell_dir,movie,'plane0')
                print('extracting roi from {}'.format(movie_dir))
                try:
                    iscell = np.load(os.path.join(movie_dir,'iscell.npy'))
                    roi_stat = list(np.load(os.path.join(movie_dir,'stat.npy'),allow_pickle=True))
                    reg_metadata = np.load(os.path.join(movie_dir,'ops.npy'),allow_pickle = True).tolist()
                except:
                    print('missing or corrupt suite2p files, skipping {}'.format(movie))
                    continue
                cell_indices = np.where(iscell[:,0])[0]
                cell_pix = create_cell_pix(roi_stat, Ly=reg_metadata['Ly'], Lx=reg_metadata['Lx'], allow_overlap=reg_metadata['allow_overlap'])
                #%
                neuropil_masks_dict = dict()
                neuropil_mask = create_neuropil_masks(ypixs=[stat['ypix'] for stat in roi_stat],
                                                      xpixs=[stat['xpix'] for stat in roi_stat],
                                                      cell_pix=cell_pix,
                                                      inner_neuropil_radius=reg_metadata['inner_neuropil_radius'],
                                                      min_neuropil_pixels=reg_metadata['min_neuropil_pixels'])
                original_neuropil_mask = neuropil_mask.reshape(len(roi_stat),reg_metadata['Ly'],reg_metadata['Lx'])
                maskindices=list()
                for i in range(original_neuropil_mask.shape[0]):
                    maskindices.append(np.where(original_neuropil_mask[i,:,:]))#)
                neuropil_masks_dict['neuropil_masks_original']=maskindices    #%%
                neuropil_masks = list()
                nimgbatch = min(reg_metadata['batch_size'], 1000)
                nframes = int(reg_metadata['nframes'])
                Ly = reg_metadata['Ly']
                Lx = reg_metadata['Lx']
                ncells = neuropil_mask.shape[0]
                
                
                for neuropil_pixel_num in neuropil_pixel_nums:
                    neuropil_mask = create_neuropil_masks(ypixs=[stat['ypix'] for stat in roi_stat],
                                                      xpixs=[stat['xpix'] for stat in roi_stat],
                                                      cell_pix=cell_pix,
                                                      inner_neuropil_radius=neuropil_pixel_num,
                                                      min_neuropil_pixels=reg_metadata['min_neuropil_pixels'])
                    neuropil_masks.append(neuropil_mask)
                    neuropil_mask_now = neuropil_mask.reshape(len(roi_stat),reg_metadata['Ly'],reg_metadata['Lx'])
                    maskindices=list()
                    for i in range(neuropil_mask_now.shape[0]):
                        maskindices.append(np.where(neuropil_mask_now[i,:,:]))#)
                    neuropil_masks_dict['neuropil_masks_min_neuropil_pixels_{}'.format(neuropil_pixel_num)]=maskindices    #%%
                
                np.save(os.path.join(movie_dir,'neuropil_masks.npy'),neuropil_masks_dict)
                
                
                for channel in [1,2]:
                    if channel == 1:
                        reg_file = open(os.path.join(movie_dir,'data.bin'), 'rb')
                    else:
                        reg_file = open(os.path.join(movie_dir,'data_chan2.bin'), 'rb')
                    for neuropil_pixel_num,neuropil_mask in zip(neuropil_pixel_nums,neuropil_masks):
                        Fneu = np.zeros((ncells, nframes),np.float32)
                        nimgbatch = int(nimgbatch)
                        block_size = Ly*Lx*nimgbatch*2
                        ix = 0
                        data = 1
                    
                        while data is not None:
                            buff = reg_file.read(block_size)
                            data = np.frombuffer(buff, dtype=np.int16, offset=0)
                            nimg = int(np.floor(data.size / (Ly*Lx)))
                            if nimg == 0:
                                break
                            data = np.reshape(data, (-1, Ly, Lx))
                            inds = ix+np.arange(0,nimg,1,int)
                            data = np.reshape(data, (nimg,-1))
                            Fneu[:,inds] = np.dot(neuropil_mask , data.T)
                            ix += nimg
                        if channel == 1:
                            np.save(os.path.join(movie_dir,'Fneu_min_neuropil_pixels_{}.npy'.format(neuropil_pixel_num)),Fneu)
                        else:
                            np.save(os.path.join(movie_dir,'Fneu_chan2_min_neuropil_pixels_{}.npy'.format(neuropil_pixel_num)),Fneu)
                    reg_file.close()
                print('Extracted fluorescence from %d ROIs in %d frames, %0.2f sec.'%(ncells, reg_metadata['nframes']*6, time.time()-t0))
                    #%%
# =============================================================================
#                     break
#                     
#                 #%%
#                 
#                 if len(cell_indices) <2:
#                     continue
#                 break
#             if len(cell_indices) <2:
#                 continue
#             break
#         if len(cell_indices) <2:
#             continue
#         break
# =============================================================================
    
                    
                    
#%% copy  iscell.npys
def suite2p_save_iscell_file_locations(basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig_old_registration',destdir = '/home/rozmar/Data/Calcium_imaging/suite2p/iscell_files'):
    filepaths = list()
    sessions = os.listdir(basedir)
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(basedir,session)
        if not os.path.isdir(session_dir):
            continue
        files_dict = extract_files_from_dir(session_dir)
        for dir_now in files_dict['dirs']:
            if 'stack' not in dir_now and 'suite2p' not in dir_now:
                cell_dir = os.path.join(session_dir,dir_now)
                movies_dict = extract_files_from_dir(cell_dir)
                for movie_dir_now in movies_dict['dirs']:
                    print(movie_dir_now)
                    sourcefile = os.path.join(cell_dir,movie_dir_now,'plane0/iscell.npy')
                    if os.path.exists(sourcefile):
                        
                        filepaths.append(sourcefile)
    # =============================================================================
    #                 try:
    #                     shutil.copy(sourcefile,os.path.join(destdir,'{}_{}_{}.npy'.format(session,dir_now,movie_dir_now)))
    #                 except:
    #                     print('not found')
    # =============================================================================
    with open(os.path.join(destdir,'listfile.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % path for path in filepaths)
        
def suite2p_backup_iscell_files(basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig'):
    #%%
    #basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig'
    filepaths = list()
    sessions = os.listdir(basedir)
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(basedir,session)
        if not os.path.isdir(session_dir):
            continue
        files_dict = extract_files_from_dir(session_dir)
        for dir_now in files_dict['dirs']:
            if 'stack' not in dir_now and 'suite2p' not in dir_now:
                cell_dir = os.path.join(session_dir,dir_now)
                movies_dict = extract_files_from_dir(cell_dir)
                for movie_dir_now in movies_dict['dirs']:
                    print(movie_dir_now)
                    sourcefile = os.path.join(cell_dir,movie_dir_now,'plane0/iscell.npy')
                    if os.path.exists(sourcefile):
                        
                        filepaths.append(sourcefile)
                    try:
                        shutil.copy(sourcefile,os.path.join(cell_dir,movie_dir_now,'plane0/iscell_backup.npy'))
                    except:
                        print('not found')
# =============================================================================
#     with open(os.path.join(destdir,'listfile.txt'), 'w') as filehandle:
#         filehandle.writelines("%s\n" % path for path in filepaths)
# =============================================================================
def suite2p_rerun_classifier(basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig',classifierfile='/home/rozmar/Data/Calcium_imaging/suite2p/iscell_files/bigclassifier.npy'):
    #%%
    
# =============================================================================
#     basedir = '/home/rozmar/Data/Calcium_imaging/suite2p/Genie_2P_rig'
#     classifierfile='/home/rozmar/Data/Calcium_imaging/suite2p/iscell_files/bigclassifier.npy'
# =============================================================================
    sessions = os.listdir(basedir)
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(basedir,session)
        if not os.path.isdir(session_dir):
            continue
        files_dict = extract_files_from_dir(session_dir)
        for dir_now in files_dict['dirs']:
            if 'stack' not in dir_now and 'suite2p' not in dir_now:
                cell_dir = os.path.join(session_dir,dir_now)
                movies_dict = extract_files_from_dir(cell_dir)
                for movie_dir_now in movies_dict['dirs']:
                    print(movie_dir_now)
                    iscellfile = os.path.join(cell_dir,movie_dir_now,'plane0/iscell.npy')
                    opsfile = os.path.join(cell_dir,movie_dir_now,'plane0/ops.npy')
                    statfile = os.path.join(cell_dir,movie_dir_now,'plane0/stat.npy')
                    if os.path.exists(iscellfile):
# =============================================================================
#                         try:
# =============================================================================
                        ops = np.load(opsfile, allow_pickle=True).item()
                        stat = np.load(statfile, allow_pickle=True)
                        ops['save_path'] = os.path.join(cell_dir,movie_dir_now,'plane0')
                        iscell = classification.classify(ops, stat, classfile=classifierfile)
                        #np.save(iscellfile,iscell)
# =============================================================================
#                         except:
#                             print('error..')
# =============================================================================
# =============================================================================
#     with open(os.path.join(destdir,'listfile.txt'), 'w') as filehandle:
#         filehandle.writelines("%s\n" % path for path in filepaths)
# =============================================================================
                        
