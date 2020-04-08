import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from apsegment import APsegment 

# plot F traces
bucket = np.load(r'F:\ufGCaMP2pData\Analysis\bucket_3AP.npy', allow_pickle=True)

allF = np.asarray([f.Fsegment for f in bucket])

allF = allF.T - np.mean(allF, axis=1)
meanF = np.mean(allF, axis=1)

plt.figure()
plt.plot(allF, color='0.8')
plt.plot(meanF, 'k-')
'''
from polygondrawer import PolygonDrawer

maxProj = tiff.imread(r'F:/ufGCaMP2pData/martonData/20200322-anm472004/cell5/suite2p/plane0/maxproj_chan0_cell5_stim03_.tif')

scaledImg = (maxProj - np.min(maxProj)) / (np.max(maxProj) - np.min(maxProj))

pd = PolygonDrawer("Polygon", scaledImg)
mask = pd.run()

mask = mask.astype(bool) # convert to boolean mask array

'''
