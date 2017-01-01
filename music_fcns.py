import os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
import numpy as np
import h5py

def getData(read_speech):
    if(read_speech):
        data_path = os.getcwd()+'/Data/speech_wav/'
    else:
        data_path=os.getcwd()+'/Data/music_wav/'
    file_list = os.listdir(data_path)
    tmp_data =[]
    for f in file_list:
        tmp_data.append(wav.read(data_path+f)[1])
    music_data = np.array(tmp_data)
    return music_data
        
# read each file and store it in some hdf5 file
def write_init_h5(fileName):
    speech_flag = 1;    
    music_flag = 0
    speech_data = getData(speech_flag)   
    music_data  = getData(music_flag)
    with h5py.File(fileName, 'w') as hf:
            hf.create_dataset('music', data=music_data)
            hf.create_dataset('speech', data=speech_data)
    return music_data,speech_data

def plotData(data,fig_ID):
    plt.figure(fig_ID)
    plt.plot(data)
    plt.show()

def low_pass_filter(in_data,THRESH):
    if(~np.any(np.iscomplex(in_data)) ):
        low_pass_filter(fftshift(fft(in_data)))
    in_data[in_data>THRESH] =0
    return ifftshift(ifft(in_data))

def sample_data(data_length,num_samples):
    # data is sampled w/o replacement
    sample_idx = []
    while(len(sample_idx)!=num_samples):
        sample_idx = np.random.choice(data_length,num_samples,replace=False)
    return sample_idx

def extract_phase_I_features(in_data):
    # If we are dealing with freq. domain look at the magnitude
    if(np.any(np.iscomplex(in_data)) ):
        extract_phase_I_features(np.abs(in_data) )
    [M,N]       = np.shape(in_data)
    data_mean   = np.zeros(M)
    data_var    = np.zeros(M)
    data_median = np.zeros(M)
    for k in range(len(in_data)):
        data_mean[k] = np.mean(in_data[k])
        data_var[k] = np.var(in_data[k])
        data_median[k] = np.median(in_data[k])
    return data_mean, data_var, data_median

def plotHist(x,fig_ID):
	plt.figure(fig_ID)
    plt.plot(data)
    plt.hist(x)
    plt.show()

def extract_phase_III_features(in_data):
    if(~np.any(np.iscomplex(in_data))):
        extract_phase_III_features(in_data)
        