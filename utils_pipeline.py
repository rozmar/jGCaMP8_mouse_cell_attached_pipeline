import os 
from ScanImageTiffReader import ScanImageTiffReader
from pathlib import Path
import json
import numpy as np
from suite2p import run_s2p
import shutil
#%%
def extract_scanimage_metadata(file):
    image = ScanImageTiffReader(file)
    metadata_raw = image.metadata()
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
    z_collection = metadata['hFastZ']['userZs']
    num_planes = len(z_collection)
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
    out = {'metadata':metadata,
           'roidata':data,
           'roi_metadata':roi_metadata,
           'frame_rate':frame_rate,
           'num_planes':num_planes}
    return out

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

def batch_run_suite2p(s2p_params):
    #%%
    sessions = os.listdir(s2p_params['basedir'])
    sessions.sort()
    for session in sessions:
        session_dir = os.path.join(s2p_params['basedir'],session)
        if not os.path.isdir(session_dir):
            continue
        files_dict = extract_files_from_dir(session_dir)
        for dir_now in files_dict['dirs']:
            if 'stack' not in dir_now and 'suite2p' not in dir_now:
                cell_dir = os.path.join(session_dir,dir_now)
                files_dict = extract_files_from_dir(cell_dir)
                savedir = s2p_params['basedir_out'] + files_dict['dir'][len(s2p_params['basedir']):]
                savedir_slow = s2p_params['basedir_out_slow'] + files_dict['dir'][len(s2p_params['basedir']):]
                uniquebasenames = np.unique(files_dict['basenames'][files_dict['exts']=='.tif'])
                for uniquebasename in uniquebasenames:
                    #%
                    save_dir_now = os.path.join(savedir,uniquebasename)
                    save_dir_slow_now = os.path.join(savedir_slow,uniquebasename)
                    file_idxs = files_dict['basenames'] == uniquebasename
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
                    ops = run_s2p.default_ops()
                    
                    print('zoom: {}x, pixelsizes: {} micron, dims: {} pixel, FOV: {} microns, estimated XFOV: {}'.format(zoomfactor, pixelsize,movie_dims,FOV,XFOV))
                    ops['reg_tif'] = True # save registered movie as tif files
                    ops['num_workers'] = 10
                    ops['delete_bin'] = 0 
                    ops['keep_movie_raw'] = 0
                    ops['fs'] = float(metadata['frame_rate'])
                    ops['nchannels'] = int(metadata['metadata']['hChannels']['channelsAvailable'])
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
                    os.makedirs(save_dir_now, exist_ok=True)
                    os.makedirs(save_dir_slow_now, exist_ok=True)
                    files_already_there = os.listdir(save_dir_now)
                    files_already_there_slow = os.listdir(save_dir_slow_now)
                    if len(files_already_there)>1 or len(files_already_there_slow)>1:
                        print('{} already done'.format(uniquebasename))
                        #time.sleep(10000)
                    else:
                        print(ops['data_path'])
                        
                        #%%
                        run_s2p.run_s2p(ops)
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