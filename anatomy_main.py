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
            if 'center_x' not in metadata_sample.keys() and channel=='RFP':
                data = readers[-1].compute()
                data_filt=ndimage.filters.gaussian_filter(data,filter_sigma)
                max_value = np.max(data_filt)
                max_position =np.unravel_index(np.argmax(data_filt), data_filt.shape)
                max_position_rfp.append(max_position)
                max_value_rfp.append(max_value)
                rotation_rfp.append(np.nan)
            elif channel=='RFP':
                pixel_num_to_cut = 1000
                max_position =np.asarray([metadata_sample_row['center_x'],metadata_sample_row['center_y']],int)
                max_position_rfp.append(max_position)
                data = readers[-1].compute()
                data_cut = data[max_position[0]-pixel_num_to_cut:max_position[0]+pixel_num_to_cut,max_position[1]-pixel_num_to_cut:max_position[1]+pixel_num_to_cut]
                data_filt=ndimage.filters.gaussian_filter(data_cut,filter_sigma)
                max_value = np.max(data_filt)
                max_value_rfp.append(np.mean(max_value))
                rotation_rfp.append(metadata_sample_row['rotation'])
# =============================================================================
#                 print('done')
#                 time.sleep(1000)
# =============================================================================
        metadata_all[sample_name][channel]['z_positions']=z_positions
        metadata_all[sample_name][channel]['fnames']=fnames
        metadata_all[sample_name][channel]['readers']=readers
        metadata_all[sample_name][channel]['readers']=readers
        if channel=='RFP':
            metadata_all[sample_name][channel]['max_pos']=max_position_rfp
            metadata_all[sample_name][channel]['max_val']=max_value_rfp
            metadata_all[sample_name][channel]['rotation']=rotation_rfp
            
#%%
channel='RFP'


for sample_name in sample_names:
    #%
    z_positions = metadata_all[sample_name][channel]['z_positions']
    max_value_rfp = metadata_all[sample_name][channel]['max_val']
    #%
    plt.plot(z_positions,max_value_rfp,label = sample_name)
plt.xlabel('distance along rostrocaudal axis (microns)')
plt.ylabel('intensity on anti-GFP channel (AU)')
plt.legend()               

#%%
import math
def rotate_rectangle(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
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
        if not force_default_rectangle and len(metadata_all[imageselector.sample]['selected_rectangle'][idx])>1 and sum(np.abs(metadata_all[imageselector.sample]['selected_rectangle'][idx].flatten()))>100:
            image_to_cut = [metadata_all[imageselector.sample]['selected_rectangle'][idx]]
            print(['loaded rectange',image_to_cut])
        else:
            print('loading default rectangle')
            1/0
            
        
    except: #default rectangle
    #%
        rectangle_halfwidth = 2000
        rectangle_halfheight = 1000
        image_to_cut =[np.array([[rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight],
                       [rectangle_center_position[0]-rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]+rectangle_halfheight],
                       [rectangle_center_position[0]+rectangle_halfwidth,rectangle_center_position[1]-rectangle_halfheight]])]
        if not np.isnan(metadata_all[imageselector.sample]['RFP']['rotation'][idx]):
            angle =  (metadata_all[imageselector.sample]['RFP']['rotation'][idx] +np.pi/2)
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
    
    

    
#%%
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
#%%
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
        metadata_all[subject]['rectangle_rotation'] = np.asarray(angles)
        metadata_all[subject]['rectangle_center_x'] = np.asarray(center_coords)[:,0]
        metadata_all[subject]['rectangle_center_y'] = np.asarray(center_coords)[:,1]
        
#%% rotate and cut images
Xsize_half = 1500
Ysize_half = 2000
save_dir = '/home/rozmar/Data/Anatomy/cropped_rotated'
overwrite = True
from scipy import ndimage
from pathlib import Path
from libtiff import TIFF
for channel in ['RFP','DAPI','FITC']:
    for subject in metadata_all.keys():
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
                    imgF = imgC[pivot[1]-Xsize_half:pivot[1]+Xsize_half,pivot[0]-Ysize_half:pivot[0]+Ysize_half]
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
                    
