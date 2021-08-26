# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:36:50 2021

@author: MSBak
"""

import pandas as pd # !pip install pandas
import numpy as np
import matplotlib.pyplot as plt # !pip install matplotlib

def mslinear_regression(x,y):
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m
    return m, b # bx+a

def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout

def smoothListGaussian(array1,window):  
     window = round(window)
     degree = (window+1)/2
     weight=np.array([1.0]*window)  
     weightGauss=[]  

     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  

     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(array1.shape[0]-window)
     
     weight = weight / np.sum(weight) # nml

     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(array1[i:i+window])*weight)/sum(weight)  

     return smoothed

path = 'A:\\itch_event_detection\\s1214-1\\'
fname = 'Results_5HT_before_1229.csv'

mouselist = []
mouselist.append('Results_5HT_before_1229.csv')
mouselist.append('Results_cap_1-3_1229.csv')
mouselist.append('Results_5HT_5-7_1229.csv')


se = 0
for se in range(len(mouselist)):
    loadpath = path + mouselist[se]
    df = pd.read_csv(loadpath)
    
    matrix = np.array(df)[:,1:]
    roinum = matrix.shape[1]
    allo = [] #np.zeros(matrix.shape) * np.nan
    # allo2 = np.zeros(matrix.shape) * np.nan
    
    cut = 0.3
    
    ws = 1
    for ROI in range(roinum):
        baseratio = 0.3
        base = np.sort(matrix[:,ROI])[0:int(round(matrix.shape[0]*baseratio))]
        mean = np.mean(base)
        deltaF = (matrix[:,ROI] - mean) / mean
        allo.append(smoothListGaussian(deltaF,10))
    
    allo = np.array(allo)
    # allo.shape
    
    savename = mouselist[se] + '.npy'
    filepath = 'A:\\Cascade-master\\raw\\' + savename
    np.save(filepath, allo, allow_pickle=True, fix_imports=True)
        # allo2[:,ROI] = ms_smooth(mssignal=deltaF, ws=2)
    
#%% load, save in vars, vis

path = 'A:\\Cascade-master\\estimated\\'
path_raw = 'A:\\Cascade-master\\raw\\'
for se in range(len(mouselist)):
    traces = np.load(path + 'predictions_' + mouselist[se] +'.npy', allow_pickle=True)
    traces[np.isnan(traces)] = 0
    print(traces.shape)
    
    signals = np.load(path_raw + mouselist[se] +'.npy', allow_pickle=True)
    ROI = 0
    roinum = signals.shape[0]
    for ROI in range(roinum):
        plt.figure()
        figname = mouselist[se] + '_ROI_' + str(ROI) + '.png'
        plt.title(figname)
        plt.plot(traces[ROI,:])
        plt.plot(signals[ROI,:] - np.min(signals[ROI,:]))
        plt.savefig('A:\\Cascade-master\\vis\\' + figname, dpi=200)
        plt.close()

            
            
            
            
            
            
            
            
            










