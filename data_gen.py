import h5py
import pandas as pd
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from numpy import r_

# Get training data
store = pd.HDFStore('./data/April_20th_wfs.h5','r')
raw = store['raw'] 
xtrain = raw.ix[:60000,4:]/7631.1981111086952
xtest = raw.ix[60000:70000,4:]/7631.1981111086952
xtest = xtest.as_matrix()

# Temp, clean training
xtrain = xtrain.as_matrix()[list(set(r_[0:60000:1]) - 
                                 set((653,1144,2493,6277,
                                      8973,9469,14524,21953,22281,22395,
                                      22836,29706,33711,34579,35848,35915,
                                      45733,46918,48443,49895,56860,58161))),:]

# Get empirical noise
orgin = h5py.File('./data/GLAH01_033_1102_028_0090_1_02_0001.H5')
dataset40 = orgin['Data_40HZ']
valid_bool = dataset40['Waveform']['Characteristics']['i_waveformType'][:]==1
signals = np.max(dataset40['Waveform']['RecWaveform']['r_rng_wf'][:][valid_bool],axis=1) > 0.1
noise_sub = dataset40['Waveform']['RecWaveform']['r_rng_wf'][:][valid_bool][signals,200:].flatten()
noise_pool=noise_sub[(noise_sub <.06)*(noise_sub > -0.06)].ravel()

# Get empirical scaling factors
my_sigs = dataset40['Waveform']['RecWaveform']['r_rng_wf'][:][valid_bool][signals,:]
widthDist = np.load('./data/orbit_width_counts-idx.npy')

# widths 5 to 99
maxes = []
for i in range(5,100):
    mynum = []
    for j in range(-5,5):
        mynum.append(np.max(my_sigs[widthDist[i+j,:],:],axis=1).tolist())
    maxes.append([item for sublist in mynum for item in sublist])
    
# widths > 100 ; treated as a single population
cache = []
for i in range(100,540):
    cache.append(np.max(my_sigs[widthDist[i,:],:],axis=1).tolist())
    maxes_over_100ns = [item for sublist in cache for item in sublist]
maxes_over_100ns = np.array(maxes_over_100ns)

# Generator for random noise
# Returns 1 of 1.7 million valid noise vectors
def return_noise():
    while True:
        y_start = random.choice(r_[0:2290156:544][:-1]) + np.randint(0,460)
        y_noise = noise_pool[y_start:y_start + 544]
        if max(np.abs(y_noise[:-1] - y_noise[1:])) < .045:
            yield y_noise
        else:
            pass #yield return_noise(y)

noise = return_noise()

def scale_and_noise(shifted_wf, width):
    if width < 100:
        peak = np.random.choice(np.array(maxes[width-5]))
    else:
        peak =np.random.choice(maxes_over_100ns)
    shifted_wf = (shifted_wf / max(shifted_wf)) * peak
    return shifted_wf[:] + noise.__next__()[4:]
    

# Function for offsets
offsets = np.argmax(dataset40['Waveform']['RecWaveform']['r_rng_wf'][:][valid_bool][signals,:] > 0.1,axis=1)
def return_shifted(conv_training_wf):
    bounds_arr = np.nonzero(conv_training_wf)[0]
    #min_max = bounds_arr[::max(1, len(testing)-1)]
    candidate = random.choice(offsets)
    while candidate >= (542 - bounds_arr[-1]):
        candidate = random.choice(offsets)
    shifted = np.zeros_like(conv_training_wf)
    length = bounds_arr[-1] - bounds_arr[0]
    #print(length,candidate,candidate+length)
    shifted[candidate:candidate+length] = conv_training_wf[bounds_arr[0]:bounds_arr[-1]]
    scaled = scale_and_noise(shifted, length)
    return scaled
    #if candidate < (540 - bounds_arr[-1]):

# Function to get data

def get_data(number_of_wfs_to_get):
    n = number_of_wfs_to_get
    idx = np.random.choice(r_[0:len(xtrain):1],n,replace=False)
    #y = gaussian_filter1d(xtrain.as_matrix()[idx,:],.8,axis=1)
    y = gaussian_filter1d(xtrain[idx,:],.8,axis=1)
    offset_with_noise = np.zeros_like(y)
    for i, wf in enumerate(y):
        offset_with_noise[i] = return_shifted(wf)
    return offset_with_noise, y, idx
