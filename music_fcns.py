import os
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from numpy.fft import fft
from numpy.fft import fftshift
import numpy as np
import h5py

def getData(read_speech):
    if(read_speech):
        data_path = '/Users/acamacho/Desktop/music_classification/Data/speech_wav/'
    else:
        data_path='/Users/acamacho/Desktop/music_classification/Data/music_wav/'
    file_list = os.listdir(data_path)
    tmp_data =[]
    for f in file_list:
        tmp_data.append(wav.read(data_path+f)[1])
    music_data = np.array(tmp_data)
    return music_data
        
# read each file and store it in some hdf5 file
def write_init_h5():
    speech_flag = 1;    
    music_flag = 0
    speech_data = getData(speech_flag)   
    music_data  = getData(music_flag)
    with h5py.File('/Users/acamacho/Desktop/music_classification/data.h5', 'w') as hf:
            hf.create_dataset('music', data=music_data)
            hf.create_dataset('speech', data=speech_data)
    return music_data,speech_data
