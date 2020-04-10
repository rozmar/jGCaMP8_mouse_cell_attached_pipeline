import numpy as np
import matplotlib.pyplot as plt
from apsegment import APsegment 

# plot F traces
bucket = np.load(r'F:\ufGCaMP2pData\Analysis\bucket_2AP.npy', allow_pickle=True)

allF = np.asarray([f.dFF for f in bucket])
allF = allF.T

# allF = allF.T - np.mean(allF, axis=1)
meanF = np.mean(allF, axis=1)

plt.figure()
plt.plot(allF, color='0.8')
plt.plot(meanF, 'k-')

