#from scipy.io import wavfile as wav 
import os
import h5py
import numpy as np
from librosa import feature as lb
from sklearn import svm,linear_model, cluster
from music_fcns import plotData,plotHist,extract_phase_I_features,sample_data
"""
The purpose of this program is to classify the the CA500 speech and music 
dataset. The goal is to separate comedian speech audio from the audio scripts. 

Data is stored in a HDF5 file where each row pertains to a data file:
dimension:  64 x 661500

Features to extract (Move this to a class):
    Phase I features (time):
        mean, median, var, sd
    Phase II features: (still need to do more research!!!!!)
        MFCC features. These are calculated by librosa
"""

def doMFCC(xData):
    dOut =lb.mfcc(xData[:,0])
    for i in range(1,xData.shape[0]):
        dOut = np.hstack( (dOut,lb.mfcc(xData[:,i])) )
    return dOut

def getTrainingAndTestingData(fileName):
    with h5py.File(os.getcwd()+fileName,'r') as hf:
        data1 = hf.get('music')
        data2 = hf.get('speech')
        music_data  = np.array(data1)
        speech_data = np.array(data2)  
    #sample the audio signal and create a response/label vector.
    [m,n] = np.shape(music_data)
    num_samples = int(n*0.25) # << sample half the signal
    idx = sample_data(n, num_samples)
    music_data = music_data[:,idx]
    speech_data = speech_data[:,idx]
    Y = np.zeros(2*m)
    Y[0:m] = 1
    
    # create training and test set
    X = np.vstack((speech_data,music_data) )
    num_sample = int(np.floor(3*(X.shape[0])/4))
    idx = sample_data(X.shape[0], num_sample ) 
    X_train = X[idx,:]        
    Y_train = Y[idx]
    X_test = np.delete(X,idx,axis=0) 
    Y_test = np.delete(Y,idx)
    
    return X_train,Y_train,X_test,Y_test
    
def getFeatureVector(fileName):
    X_train,Y_train,X_test,Y_test = getTrainingAndTestingData(fileName)
    # get phase I
    x_mean,x_var, x_median = extract_phase_I_features(X_train)
    feat_mat = np.vstack([x_mean,x_var,x_median]).transpose()
    # get MFCC 
    mfcc_train = doMFCC(X_train).T
    return feat_mat,mfcc_train,X_train,Y_train,X_test,Y_test
###############################################################################
    
fName ='/music_speech_data.h5'
feat_mat,mfcc_train,X_train,Y_train,X_test,Y_test = getFeatureVector(fName)   

# MFCC test components
#mfcc_test = doMFCC(X_test).T
#kmeans = cluster.KMeans(n_clusters=5).fit(mfcc_train)
#print(kmeans.predict(mfcc_test) )
'''
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(mfcc_train,Y_train)
print('Logistic regression results: ')
print(sum(logreg.predict(pow(test_mfcc,10))==Y_test)/100.0)
###svm stuff
clf = svm.SVC(kernel='linear')
clf.fit(mfcc_train,Y_train)
print('svm results: ')
print(sum(clf.predict(pow(test_mfcc,10))==Y_test)/100.0)
'''