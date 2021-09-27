#%% you need the napari environment to run this script
import napari
from magicgui import magicgui
import vispy
from enum import Enum
import pickle as pkl
import numpy as np
import os
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("gui qt5")  
import os
from skimage.io import imread
from dask import delayed
import matplotlib.pyplot as plt

#%% metadata
import pandas as pd
metadata_dir = '/home/rozmar/Data/Anatomy/Metadata'
img_dir_base = '/home/rozmar/Data/Anatomy/exported'
metadata_files = os.listdir(metadata_dir)
anatomy_metadata=dict()

for metadata_file in metadata_files:
    if 'anm' in metadata_file and '??' not in metadata_file: #and '20x' not in metadata_file and 
        anatomy_metadata[metadata_file[:-4]]=pd.read_csv(os.path.join(metadata_dir,metadata_file))
sample_names = anatomy_metadata.keys()    
channels = ['DAPI','FITC','RFP']
#%%
channels_to_calculate_max = ['RFP']
import scipy.ndimage as ndimage
import time
filter_sigma = 50 # pixels ~ 6 microns
lazy_imread = delayed(imread)

metadata_all=dict()
for sample_name in sample_names:
    print(sample_name)
    metadata_all[sample_name]=dict()
    metadata_sample = anatomy_metadata[sample_name]
    for channel in channels:
        metadata_all[sample_name][channel] = dict()
        z_plane = '0'
        fnames= list()
        z_positions= list()
        readers=list()
        max_position_rfp =list()
        max_value_rfp = list()
        rotation_rfp = list()
        for metadata_sample_row in metadata_sample.iterrows():
            metadata_sample_idx = metadata_sample_row[0]
            metadata_sample_row = metadata_sample_row[1]
            filename = 'Slide {:d} from cassette {:d}-Region {:03d}_{}_{:d}.tiff'.format(int(metadata_sample_row['Slide']),
                                                                                 int(metadata_sample_row['Casette']),
                                                                                 int(metadata_sample_row['Region']),
                                                                                 channel,
                                                                                 int(z_plane))
            path = os.path.join(img_dir_base,sample_name,'Slides','Slide {:d} of {:d}'.format(int(metadata_sample_row['Slide']),
                                                                                              int(metadata_sample_row['Casette'])))
            z_positions.append(metadata_sample_row['Z location'])
            fnames.append(os.path.join(path,filename))
            readers.append(lazy_imread(os.path.join(path,filename)))  # doesn't actually read the file
            if 'center_x' not in metadata_sample.keys() and channel in channels_to_calculate_max:
                data = readers[-1].compute()
                data_filt=ndimage.filters.gaussian_filter(data,filter_sigma)
                max_value = np.max(data_filt)
                max_position =np.unravel_index(np.argmax(data_filt), data_filt.shape)
                max_position_rfp.append(max_position)
                max_value_rfp.append(max_value)
                rotation_rfp.append(np.nan)
# =============================================================================
#                 print('done')
#                 time.sleep(1000)
# =============================================================================
            elif channel in channels_to_calculate_max:
                if '20x' in sample_name:
                    pixel_num_to_cut = 2000
                else:
                    pixel_num_to_cut = 1000
                max_position =np.asarray([metadata_sample_row['center_x'],metadata_sample_row['center_y']],int)
                max_position_rfp.append(max_position)
                data = readers[-1].compute()
                data_cut = data[int(np.max([0,max_position[0]-pixel_num_to_cut])):max_position[0]+pixel_num_to_cut,int(np.max([0,max_position[1]-pixel_num_to_cut])):max_position[1]+pixel_num_to_cut]
                data_filt=ndimage.filters.gaussian_filter(data_cut,filter_sigma)
                max_value = np.max(data_filt)
                max_value_rfp.append(np.mean(max_value))
                rotation_rfp.append(metadata_sample_row['rotation'])
# =============================================================================
#                 print('done')
#                 time.sleep(1)
# =============================================================================
        metadata_all[sample_name][channel]['z_positions']=z_positions
        metadata_all[sample_name][channel]['fnames']=fnames
        metadata_all[sample_name][channel]['readers']=readers
        metadata_all[sample_name][channel]['readers']=readers
        if channel in channels_to_calculate_max:
            metadata_all[sample_name][channel]['max_pos']=max_position_rfp
            metadata_all[sample_name][channel]['max_val']=max_value_rfp
            metadata_all[sample_name][channel]['rotation']=rotation_rfp
            
#%%
fig = plt.figure()
ax_rfp = fig.add_subplot(2,1,1)
ax_fitc = fig.add_subplot(2,1,2)
channel='RFP'
idxs = list()
samples_used = list()
for sample_name in sample_names:
    if 'xcamp' in sample_name.lower():
        color = 'orange'
    elif '7f' in sample_name.lower():
        color = 'green'
    elif '456' in sample_name.lower():
        color = 'blue'
    elif '686' in sample_name.lower():
        color = 'red'
    elif '688' in sample_name.lower():
        color = 'black'
    color
    if '20x' in sample_name:
        continue
    print(sample_name)
    #%
    z_positions = metadata_all[sample_name][channel]['z_positions']
    max_value_rfp = metadata_all[sample_name][channel]['max_val']
    #%
    max_idx = np.argmax(max_value_rfp)
    samples_used.append(sample_name)
    idxs.append(max_idx)
    ax_rfp.plot(np.asarray(z_positions) - z_positions[max_idx],max_value_rfp,label = sample_name,color = color)
ax_rfp.set_xlabel('distance along rostrocaudal axis (microns)')
ax_rfp.set_ylabel('intensity on anti-GFP channel (AU)')
ax_rfp.legend()               
#%
channel='FITC'
for sample_name,max_idx in zip(samples_used,idxs):
    if 'xcamp' in sample_name.lower():
        color = 'orange'
    elif '7f' in sample_name.lower():
        color = 'green'
    elif '456' in sample_name.lower():
        color = 'blue'
    color
    if '20x' in sample_name:
        continue
    print(sample_name)
    #%
    z_positions = metadata_all[sample_name][channel]['z_positions']
    max_value_rfp = metadata_all[sample_name][channel]['max_val']
    #%
    ax_fitc.plot(np.asarray(z_positions) - z_positions[max_idx],max_value_rfp,label = sample_name,color = color)
ax_fitc.set_xlabel('distance along rostrocaudal axis (microns)')
ax_fitc.set_ylabel('intensity on native GFP channel (AU)')
ax_fitc.legend()   
#%%
import math
def rotate_rectangle(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    #angle = angle*-1 - 90
    ox, oy = origin
    px = points[:,0]
    py = points[:,1]
    
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.asarray([qx,qy]).T
metadata_now = metadata_all[sample_name][channel]
@magicgui(sample={'choices':sample_names},zpos={'choices': z_positions},channel={'choices': channels})
def label_selection(sample,channel,zpos):
    return sample,channel, zpos
imageselector = label_selection.Gui()
#%

def show_image(force_default_rectangle = False):
    rectangle_halfwidth = 2000
    rectangle_halfheight = 1000
    if not type(force_default_rectangle)== bool:
        force_default_rectangle = False
    print('+++++++++++++++++++++++++++++show image starts')
    z_positions=metadata_all[imageselector.sample][imageselector.channel]['z_positions']
    idx = np.argmax(np.asarray(z_positions)==float(imageselector.zpos))
    if 'selected_rectangle' not in metadata_all[imageselector.sample].keys():
        metadata_all[imageselector.sample]['selected_rectangle'] = np.empty([len(z_positions),4,2])
    #%
    
    print(['idx',idx,'force',force_default_rectangle,len(metadata_all[imageselector.sample]['selected_rectangle'][idx]),sum(np.abs(metadata_all[imageselector.sample]['selected_rectangle'][idx].flatten())),not force_default_rectangle and len(metadata_all[imageselector.sample]['selected_rectangle'][idx])>1 and sum(np.abs(metadata_all[imageselector.sample]['selected_rectangle'][idx].flatten()))>100])
    rectangle_center_position = metadata_all[imageselector.sample]['RFP']['max_pos'][idx]
    try:
        if not np.isnan(metadata_all[imageselector.sample]['RFP']['rotation'][idx]):
            print('calculating rotation')
            1/0
        if not force_default_rectangle and len(metadata_all[imageselector.sample]['selected_rectangle'][idx])>1 and sum(np.abs(metadata_all[imageselector.sample]['selected_rectangle'][idx].flatten()))>100:
            image_to_cut = [metadata_all[imageselector.sample]['selected_rectangle'][idx]]
            print(['loaded rectange',image_to_cut])
        else:
            print('loading default rectangle')
            1/0
            
        
    except: #default rectangle
    #%
        
        image_to_cut =[np.array([[rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight],
                       [rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight]])]
        if not np.isnan(metadata_all[imageselector.sample]['RFP']['rotation'][idx]):
            angle =  (metadata_all[imageselector.sample]['RFP']['rotation'][idx])*-1 #+np.pi/2
            image_to_cut=rotate_rectangle(rectangle_center_position, image_to_cut[0],angle)
            image_to_cut = [image_to_cut]
        #%
    #%
    print(imageselector.zpos)
    print(idx)
    image_layer.data=metadata_all[imageselector.sample][imageselector.channel]['readers'][idx].compute()
    image_layer.contrast_limits_range = [0, 32000]
    camerastate = viewer.window.qt_viewer.view.camera.get_state()
    camerastate_new={'rect':vispy.geometry.rect.Rect(rectangle_center_position[1],
                                                     rectangle_center_position[0]-camerastate['rect'].height/2,
                                                     camerastate['rect'].width,
                                                     camerastate['rect'].height)}
    viewer.window.qt_viewer.view.camera.set_state(camerastate_new)
    try:
        #save previous rectangle
        rectangle_layer=viewer.layers.pop(1)
        if 'selected_rectangle' not in metadata_all[rectangle_layer.metadata['sample']].keys():
            metadata_all[rectangle_layer.metadata['sample']]['selected_rectangle'] = np.empty([rectangle_layer.metadata['total_z_count'],4,2])
        metadata_all[rectangle_layer.metadata['sample']]['selected_rectangle'][rectangle_layer.metadata['image_idx']] = rectangle_layer.data[0]
        print(rectangle_layer.data[0])
        print(['rectangle saved',metadata_all[rectangle_layer.metadata['sample']]['selected_rectangle'][rectangle_layer.metadata['image_idx']]])
        
        #rectangle_layer.data.clear()
        #viewer.layers['image_to_cut'].add(image_to_cut, shape_type='rectangle', edge_width=5, edge_color=[1,1,1,1],face_color=[0,0,0,0])
    except:
        print('could not save data - rectangle layer not present?')
        pass
    print(image_to_cut)
    try:
        rectangle_layer = viewer.add_shapes(image_to_cut, shape_type='polygon', edge_width=50, edge_color='white',face_color='transparent',name = 'image_to_cut')
    except:
        pass
        try:
            viewer.layers.pop(1)
        except:
            pass
        print('error displaying rectangle, falling back to default')
        image_to_cut =[np.array([[rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight],
                       [rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight]])]
        if not np.isnan(metadata_all[imageselector.sample]['RFP']['rotation'][idx]):
            angle =  (metadata_all[imageselector.sample]['RFP']['rotation'][idx])*-1 #+np.pi/2
            image_to_cut=rotate_rectangle(rectangle_center_position, image_to_cut[0],angle)
            image_to_cut = [image_to_cut]
        rectangle_layer = viewer.add_shapes(image_to_cut, shape_type='polygon', edge_width=50, edge_color='white',face_color='transparent',name = 'image_to_cut')
    rectangle_layer.metadata = {'image_idx':idx,
                                'sample':imageselector.sample,
                                'channel':imageselector.channel,
                                'z_position':imageselector.zpos,
                                'total_z_count':len(z_positions)}
    print('--------------------------------------show image ends')
 #%
imageselector.zpos_changed.connect(show_image)

#%
def changechannel(channel):
    if channel =='DAPI':
        image_layer.colormap ='blue'
    elif channel =='FITC':
        image_layer.colormap ='green'
    elif channel =='RFP':
        image_layer.colormap ='red'
    show_image()
   
imageselector.channel_changed.connect(changechannel)

def changesample(sample):
    z_positions=metadata_all[imageselector.sample][imageselector.channel]['z_positions']
    imageselector.zpos_widget.clear()
    imageselector.zpos_widget.addItems(np.asarray(z_positions,str))
    for i,z in enumerate(z_positions):
        imageselector.zpos_widget.setItemData(i,z)
    imageselector.zpos_changed.disconnect()
    imageselector.zpos_changed.connect(show_image)
    changechannel(imageselector.channel)
imageselector.sample_changed.connect(changesample)   
    
    

    
#%
file_now = os.path.join(path,filename)
#with napari.gui_qt():
viewer = napari.view_image(readers[0].compute() )
#%
@viewer.bind_key('a',overwrite=True)
def auto_LUT(viewer):
    if imageselector.channel =='RFP':
        z_positions=metadata_all[imageselector.sample][imageselector.channel]['z_positions']
        idx = np.argmax(np.asarray(z_positions)==float(imageselector.zpos))
        image_layer.contrast_limits=[np.min(image_layer.data),metadata_all[imageselector.sample][imageselector.channel]['max_val'][idx]]
    else:
        image_layer.contrast_limits=[np.min(image_layer.data),np.percentile(image_layer.data,99.9)]
#%
@viewer.bind_key('r',overwrite=True)
def reinitialize_rectangle(viewer):
    show_image(force_default_rectangle = True)
@viewer.bind_key('t',overwrite = True)
def calculate_rectangle_rotation(viewer):    
    rectangle =viewer.layers['image_to_cut'].data[0]
    #%
    a = rectangle[1,0]-rectangle[0,0]
    b = rectangle[1,1]-rectangle[0,1]
    rotation = np.arctan(a/b) * 180 / np.pi; 
    if b <0:
        rotation += 180
    print([a,b,rotation])
#%
image_layer = viewer.layers[0]

#Z_menu = create_image_selector_menu(image_layer, z_positions)
viewer.window.add_dock_widget(imageselector)
changesample(imageselector.sample)
# =============================================================================
#     viewer = napari.Viewer()
#     viewer.add_image(file_now)
# =============================================================================
    

#%% calculate center point and rotation
for subject in metadata_all.keys():
    
# =============================================================================
#     #%%
#     subject = 'anm478342-UF456'
# =============================================================================
    rectangles = metadata_all[subject]['selected_rectangle']
    z_positions = metadata_all[subject]['DAPI']['z_positions']
#%
    angles =list()
    center_coords = list()
    for idx in range(rectangles.shape[0]):
        z = z_positions[idx]
        rectangle =rectangles[idx]
        a = rectangle[1,0]-rectangle[0,0]
        b = rectangle[1,1]-rectangle[0,1]
        rotation = np.arctan(a/b) * 180 / np.pi; 
        if b <0:
            rotation += 180
        center_coord = np.mean(rectangle,0)
        angles.append(np.deg2rad(rotation))
        center_coords.append(center_coord)
        #%
    metadata_all[subject]['rectangle_rotation'] = np.asarray(angles)
    metadata_all[subject]['rectangle_center_x'] = np.asarray(center_coords)[:,0]
    metadata_all[subject]['rectangle_center_y'] = np.asarray(center_coords)[:,1]
        
#%% rotate and cut images
Xsize_half_original = 1500
Ysize_half_original = 2000
save_dir = '/home/rozmar/Data/Anatomy/cropped_rotated'
overwrite = False
from scipy import ndimage
from pathlib import Path
from libtiff import TIFF
for channel in ['RFP','DAPI','FITC']:
    for subject in metadata_all.keys():
        if '20x' in subject:
            Xsize_half = Xsize_half_original
            Ysize_half = Ysize_half_original
        else:
            Xsize_half = Xsize_half_original/2
            Ysize_half = Ysize_half_original/2
            
# =============================================================================
#         #%%
#         subject = 'anm478342-UF456'
# =============================================================================
        readers = metadata_all[subject][channel]['readers']
        z_positions = metadata_all[subject][channel]['z_positions']
        save_dir_now = os.path.join(save_dir,subject,channel)
        Path(save_dir_now).mkdir(parents=True, exist_ok=True)
        for idx,reader in enumerate(readers):
            z_position = z_positions[idx]
            filename = '{}_{}_Z_{:06d}.tiff'.format(subject,channel,int(z_position))
            if overwrite or not os.path.exists(os.path.join(save_dir_now,filename)):
                try:
                    img = reader.compute()
# =============================================================================
#                     pivot = np.asarray([metadata_all[subject]['rectangle_center_x'][idx], metadata_all[subject]['rectangle_center_y'][idx]],int)
#                     padX = [img.shape[1] - pivot[0], pivot[0]]
#                     padY = [img.shape[0] - pivot[1], pivot[1]]
#                     imgP = np.pad(img, [padY, padX], 'constant')
#                     imgR = ndimage.rotate(imgP, np.rad2deg(metadata_all[subject]['rectangle_rotation'][idx])*-1, reshape=False)
#                     imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
#                     imgF = imgC[pivot[0]-Xsize_half:pivot[0]+Xsize_half,pivot[1]-Ysize_half:pivot[1]+Ysize_half]
# =============================================================================
                    #%
                    pivot = np.asarray([metadata_all[subject]['rectangle_center_y'][idx], metadata_all[subject]['rectangle_center_x'][idx]],int)
                    padX = [img.shape[1] - pivot[0], pivot[0]]
                    padY = [img.shape[0] - pivot[1], pivot[1]]
                    imgP = np.pad(img, [padY, padX], 'constant')
                    imgR = ndimage.rotate(imgP, np.rad2deg(metadata_all[subject]['rectangle_rotation'][idx])*+1+90, reshape=False)
                    imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
                    #imgF = imgC[pivot[1]-Xsize_half:pivot[1]+Xsize_half,pivot[0]-Ysize_half:pivot[0]+Ysize_half]
                    imgF = imgC[int(np.max([0,pivot[1]-Xsize_half])):int(np.min([pivot[1]+Xsize_half,imgC.shape[0]])),int(np.max([pivot[0]-Ysize_half,0])):int(np.min([pivot[0]+Ysize_half,imgC.shape[1]]))]
                    #%
# =============================================================================
#                     print('waiting')
#                     time.sleep(1000)
# =============================================================================
                    tiff = TIFF.open(os.path.join(save_dir_now,filename), mode='w')
                    tiff.write_image(imgF)
                    tiff.close()
                    print('{} saved'.format(filename))
                except:
                    print('{} could not be saved'.format(filename))
                    time.sleep(1000)
                    #%%
 #%% correlation between green and red channel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib.gridspec as gridspec
min_green_intensity = 500
fname_green = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479116-UF688-20x/FITC/anm479116-UF688-20x_FITC_Z_000850.tiff'
fname_red = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479116-UF688-20x/RFP/anm479116-UF688-20x_RFP_Z_000850.tiff'
fname_green = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479116-UF688-20x/FITC/anm479116-UF688-20x_FITC_Z_000700.tiff'
fname_red = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479116-UF688-20x/RFP/anm479116-UF688-20x_RFP_Z_000700.tiff'
# =============================================================================
# fname_green = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479410-UF456-20x/FITC/anm479410-UF456-20x_FITC_Z_001650.tiff'
# fname_red = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479410-UF456-20x/RFP/anm479410-UF456-20x_RFP_Z_001650.tiff'
# 
# fname_green = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479569-UF686-20x/FITC/anm479569-UF686-20x_FITC_Z_001200.tiff'
# fname_red = '/home/rozmar/Data/Anatomy/cropped_rotated/anm479569-UF686-20x/RFP/anm479569-UF686-20x_RFP_Z_001200.tiff'
# =============================================================================

im_green = Image.open(fname_green)
imarray_green = np.array(im_green).flatten()
im_red= Image.open(fname_red)
imarray_red = np.array(im_red).flatten()
needed = (imarray_red>000)&(imarray_green>min_green_intensity)
#plt.plot(imarray_green,imarray_red,'ko',markersize = 1)
fig = plt.figure()
gs = gridspec.GridSpec(6, 10, figure=fig)
ax_hist = fig.add_subplot(gs[:-1,1:-1])
plt.axis('off')
ax_hist.set_title('only pixels with green intensity > {}: {} % of the image'.format(min_green_intensity,round(len(imarray_green[needed])/len(needed)*10000)/100))
ax_red = fig.add_subplot(gs[:-1,0],sharey = ax_hist)
ax_red.set_ylabel('anti-GFP intensity')
ax_green = fig.add_subplot(gs[-1,1:-1],sharex = ax_hist)
ax_green_cumsum = ax_green.twinx()
ax_green.set_xlabel('native GCaMP intensity')
ax_cbar = fig.add_subplot(gs[:,-1])
plt.axis('off')

im = ax_hist.hist2d(imarray_green[needed],imarray_red[needed],500,norm=colors.LogNorm(), cmap='magma')
cbar = fig.colorbar(im[3],ax=ax_cbar,fraction = 1)
#ax_green.hist(imarray_green[needed],500)
green_hist_values = np.histogram(imarray_green[needed],500)
ax_green.bar(green_hist_values[1][:-1],green_hist_values[0],width = np.diff(green_hist_values[1])[0],color = 'green')
ax_green.set_yscale('log')
ax_green_cumsum.plot(green_hist_values[1][:-1],np.cumsum(green_hist_values[0])/np.sum(green_hist_values[0])*100,'k-',linewidth = 4)
#ax_green_cumsum.set_yscale('log')
red_hist_values = np.histogram(imarray_red[needed],500)
ax_red.barh(red_hist_values[1][:-1],red_hist_values[0],np.diff(red_hist_values[1])[0],color = 'red')#,width = np.diff(red_hist_values[1])[0],left=
ax_red.set_xscale('log')
ax_red.set_xlim(ax_red.get_xlim()[::-1]) 
#%
ratio_im = np.array(im_red)/np.array(im_green)

fig_image = plt.figure()
ax_imgreen = fig_image.add_subplot(221)
ax_imgreen.set_title('native GCaMP')
ax_imred = fig_image.add_subplot(222,sharex = ax_imgreen,sharey = ax_imgreen)
ax_imred.set_title('anti-GFP')
ax_ratio = fig_image.add_subplot(223,sharex = ax_imgreen,sharey = ax_imgreen)
ax_ratio.set_title('gain: anti-GFP/GCaMP')
ax_gain = fig_image.add_subplot(224)
ax_gain.set_xlabel('native GCaMP intensity')
ax_gain.set_ylabel('gain')
im_g = ax_imgreen.imshow(im_green,norm=colors.LogNorm(), cmap='magma')
cbar = fig_image.colorbar(im_g,ax=ax_imgreen)

im_r = ax_imred.imshow(im_red,norm=colors.LogNorm(), cmap='magma')
cbar = fig_image.colorbar(im_r,ax=ax_imred)

im_r = ax_ratio.imshow(ratio_im, norm=colors.LogNorm(),cmap='magma')#,
cbar = fig_image.colorbar(im_r,ax=ax_ratio)
needed = imarray_green>10
ax_gain.hist2d(imarray_green[needed],ratio_im.flatten()[needed],500,norm=colors.LogNorm(), cmap='magma')