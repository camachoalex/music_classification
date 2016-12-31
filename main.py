#from scipy.io import wavfile as wav 
import h5py
import numpy as np
from sklearn import linear_model
from sklearn import svm
from music_fcns import plotData,extract_phase_I_features,sample_data
"""
The purpose of this program is to classify the the CA500 speech and music dataset.
The goal is to separate comedian speech audio from the audio scripts. 

Data is stored in a HDF5 file where each row pertains to data file:
dimension:  64 x 661500

Features to extract (Move this to a class):
    Phase I features (time):
        mean, median, var, sd
    Phase II features (dim. reduction features):
        principle components, singular values
            - this will be done by reshaping the data into a matrix
    Phase III:
        - bandwidth, center frequency, pulsewidth (related to bw), spectral features such as power and other stuff. (need to do more research)

In summary, the feature vector will be comprised of:

mean_t,median_t, var_t, sd_t, top j singular values, bandwidth, center_freq

"""



def main():
    with h5py.File('/data.h5','r') as hf:
        data1 = hf.get('music')
        data2 = hf.get('speech')
        music_data  = np.array(data1)
        speech_data = np.array(data2)   
        
        #sample the audio signal and create a response/label vector.
        [m,n] = np.shape(music_data)
        num_samples = n*0.5  # << sample half the signal
        idx = sample_data(n, num_samples)
        music_data = music_data[:,idx]
        speech_data = speech_data[:,idx]
        y = np.zeros(2*m)
        y[0:m] = 1
        
        # create training and test set
        X = np.vstack((speech_data,music_data))
        idx = sample_data(X.shape[0], int(np.floor((X.shape[0])*0.75)) ) 
        X_train = X[idx,:]        
        y_train = y[idx]
        X_test = X[~idx,:]
        y_test = y[~idx]
        
        # get phaseI features from time domain
        x_mean,x_var, x_median   = extract_phase_I_features(X_train)
        return X_test,y_test,X_train,y_train#,logreg
        
xt,yt,x_train,y_train = main()
        
        
# note this is done without a feature vector!!!!
# logistic regression 
'''
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x_train,y_train)
print('Logistic regression results: ')
print(sum(logreg.predict(xt)==yt))

#svm stuff
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
print('svm results: ')
print(sum(clf.predict(xt)==yt))
'''