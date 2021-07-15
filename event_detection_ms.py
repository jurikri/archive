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

ROI = 0
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
        
    # peak check

    diff = msout[1:] - msout[:-1]
    # plt.plot(msout); plt.plot(diff)
    
    peaks_ix = []; peak_sw = True
    for k in range(len(diff)):
        if diff[k] < 0 and msout[k] > cut and peak_sw:
            peaks_ix.append(int(k))
            peak_sw = False
        if diff[k] > 0 and not(peak_sw):
            peak_sw = True
    peaks_ix = np.array(peaks_ix)

    if False:
        plt.plot(msout); plt.scatter(peaks_ix, msout[peaks_ix], c='r'); plt.plot(np.ones(len(msout)) * cut) 
        plt.savefig('C:\\mass_save\\itch\\figure1.png', dpi=1000)
    peaks_ix2 = np.array(peaks_ix)
    passw = False
    while True:
        passw = True
        for w in range(len(peaks_ix2)-1):
            first = msout[peaks_ix2[w]]
            second = msout[peaks_ix2[w+1]]
            minimum = np.min(msout[peaks_ix2[w]:peaks_ix2[w+1]])
            
            # print(w, first, second, minimum)
            
            if np.min([first, second]) < minimum*2:
                minix = np.argmin([first, second])
                peaks_ix2 = np.delete(peaks_ix2, w + minix)
                passw = False
                break
        print(peaks_ix2)
        if passw: break
    if False:
        plt.plot(msout); plt.scatter(peaks_ix2, msout[peaks_ix2], c='r'); plt.plot(np.ones(len(msout)) * cut)
        plt.savefig('C:\\mass_save\\itch\\figure2.png', dpi=1000)











