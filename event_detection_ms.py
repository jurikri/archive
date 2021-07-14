# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:36:50 2021

@author: MSBak
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\mass_save\\itch\\s1214-1\\'
fname = 'Results_5HT_before_1229.csv'

loadpath = path + fname
df = pd.read_csv(loadpath)


matrix = np.array(df)[:,1:]
roinum = matrix.shape[1]
allo = np.zeros(matrix.shape) * np.nan

for ROI in range(roinum):
    baseratio = 0.1
    base = np.sort(matrix[:,ROI])[0:int(round(matrix.shape[0]*baseratio))]
    mean = np.mean(base)
    deltaF = (matrix[:,ROI] - mean) / mean
    allo[:,ROI] = deltaF

# allo는 따로 하드에 저장

def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout

for ROI in range(roinum):
    baseratio = 0.05
    msout = ms_smooth(mssignal=allo[:,ROI], ws=3)
    base = np.sort(msout)[0:int(round(msout.shape[0]*baseratio))]
    mean = np.median(base)
    std = np.std(base)
    cut = mean + (std*3)
    
    if False:
        plt.figure()
        plt.plot(msout)
        plt.plot(np.ones(msout.shape[0]) * mean)
        plt.plot(np.ones(msout.shape[0]) * cut)
        
    # 분리된 언덕 시작과 끝 정하기
    segments = []
    e = -np.inf
    for f in range(msout.shape[0]):
        if f <= e: continue
    
        if msout[f] > cut: 
            s = int(f)
            for f2 in range(s+1, msout.shape[0]):
                if msout[f2] <= cut or f2 == msout.shape[0]-1: 
                    e = int(f2)
                    print(s, e)
                    segments.append([s,e])
                    break
            
    if False:
        plt.figure()
        plt.plot(msout)
        plt.plot(np.ones(msout.shape[0]) * mean)
        plt.plot(np.ones(msout.shape[0]) * cut)
        for j1, j2 in segments:
            plt.scatter(j1, msout[j1], c='g')
            plt.scatter(j2, msout[j2], c='r')
            print(j1, j2)
            
    for j in range(len(segments)):
        s = segments[j][0]
        e = segments[j][1]
        
        seg = msout[s:e]
        diff = seg[1:] - seg[:-1]
        # plt.plot(seg); plt.plot(diff)
        
        k2 = -np.inf
        for k in range(len(diff)):
            if k <= k2: continue
        
            if diff[k] < 0:
                peak = int(k)
                for k2 in range(k+1, len(diff)):
                    if diff[k2] > 0:
                        rebound = int(k)
                        break
    
    
    
    
















