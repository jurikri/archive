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
    allo = np.zeros(matrix.shape) * np.nan
    allo2 = np.zeros(matrix.shape) * np.nan
    
    cut = 0.3
    
    ws = 1
    for ROI in range(roinum):
        baseratio = 0.3
        base = np.sort(matrix[:,ROI])[0:int(round(matrix.shape[0]*baseratio))]
        mean = np.mean(base)
        deltaF = (matrix[:,ROI] - mean) / mean

        allo[:,ROI] = ms_smooth(mssignal=deltaF, ws=ws)
        allo2[:,ROI] = ms_smooth(mssignal=deltaF, ws=2)
    
    # allo는 따로 하드에 저장
    
    # diff가 0이하이면 peak로 만들어 놓고
    # 제외 1 - df 0.3 이하면 제외
    # 제외 2 - peak 시점으로 부터 상승 - 하강이 아니면 제외. regression slope 으로 판정
    # 제외 3 - 앞 peak 들의 max 로부터 35% 이하의 감소가 없고, 10 frame 이내면 제외
    
    
    ROI = 9
    for ROI in np.array(np.arange(0, roinum, 1), dtype=int):
        # diff가 0이하이면 peak로 만들어 놓고

        mssignal = allo[:,ROI]
        diff = mssignal[1:]- mssignal[:-1]
        # plt.plot(msout); plt.plot(diff)
        
        peaks_ix = []; peak_sw = True
        # slope = np.mean(diff) - np.std(diff)
        slope = 0
        for k in range(len(diff)):
            if diff[k] < slope and peak_sw:
                peaks_ix.append(int(k))
                peak_sw = False
            if diff[k] > slope and not(peak_sw):
                peak_sw = True
        peaks_ix = np.array(peaks_ix)
    
        
        if False:
            plt.figure(); plt.title('all_peak_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
            plt.plot(mssignal); plt.scatter(peaks_ix, mssignal[peaks_ix], c='r'); plt.plot(np.ones(len(mssignal)) * cut) 
            # plt.savefig(figsavename + '_all_peak.png', dpi=100); plt.close()
        
        # 제외 1 - df 0.3 이하면 제외
        
        vix = np.where(mssignal[peaks_ix] > 0.3)[0]
        peaks_ix2 = peaks_ix[vix]
        if False: 
            plt.figure(); plt.plot(mssignal); plt.scatter(peaks_ix2, mssignal[peaks_ix2], c='r'); plt.plot(np.ones(len(mssignal)) * cut) 
            
            
        # 제외 2 - peak 시점으로 부터 상승 - 하강이 아니면 제외. regression slope 으로 판정

        l1, l2 = 4, 0
        
        # p = peaks_ix2[0]
        peaks_ix3 = []
        for p in peaks_ix2:
            cluster_before = mssignal[np.max([p-l1, 0]):np.min([p+l2, len(mssignal)])]
            if len(cluster_before) > 1:
                x = range(len(cluster_before)) ; y = cluster_before
                m_before, b = mslinear_regression(x,y)
            else: m_before = 1
            
            cluster_after = mssignal[np.max([p-l2, 0]):np.min([p+l1, len(mssignal)])]
            if len(cluster_after) > 1:
                x = range(len(cluster_after)) ; y = cluster_after
                m_after, b = mslinear_regression(x,y)
            else: m_after = -1
            
            if m_before > 0 and m_after < 0: peaks_ix3.append(p)
        if False: 
            plt.figure(); plt.plot(mssignal); plt.scatter(peaks_ix3, mssignal[peaks_ix3], c='r'); plt.plot(np.ones(len(mssignal)) * cut) 

        
        # 제외 3 - 앞 peak 들의 max 로부터 35% 이하의 감소가 없고, 10 frame 이내면 제외
        # peak 제거시 update 되도록 수정해야함
        p = 1
        peaks_ix4 = list(peaks_ix3)
        for epoch in range(int(1e6)):
            passsw = True
            if len(peaks_ix4) > 0:
                for p in range(1, len(peaks_ix4)):
                    c1 = mssignal[peaks_ix4[p-1]] * (1-0.35) > np.min(mssignal[peaks_ix4[p-1]:peaks_ix4[p]])
                    c2 = peaks_ix4[p] - peaks_ix4[p-1] > 10
                    if not(c1 or c2): peaks_ix4.pop(p); passsw = False; break
            if passsw: break
                
        if False: 
            plt.figure(); plt.plot(mssignal); plt.scatter(peaks_ix4, mssignal[peaks_ix4], c='r'); plt.plot(np.ones(len(mssignal)) * cut) 
            
        if True:
            figsavename = 'A:\\itch_event_detection\\figsave\\0811\\' + 'se_'+str(se) + '_ws_'+str(ws) + '_ROI_'+str(ROI)
            plt.figure(); plt.title('method4_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
            plt.plot(mssignal, lw=0.5); plt.scatter(peaks_ix4, mssignal[peaks_ix4], c='r', s=10); plt.plot(np.ones(len(mssignal)) * cut, lw=0.5) 
            plt.savefig(figsavename + '_method4.png', dpi=300); plt.close()
      
#%%


# save as npy
sample1 = np.transpose(np.array(signalss[273][2]))
sample2 = np.transpose(np.array(signalss[273][3]))

npy_save = []
npy_save.append(sample1)
npy_save.append(sample2)

print(tmp.shape)
plt.plot(np.mean(tmp, axis=0))

filepath = 'A:\\Cascade-master\\raw\\test.npy'
np.save(filepath, npy_save, allow_pickle=True, fix_imports=True)


# load, save in vars, vis


file_path = 'A:\\Cascade-master\\raw\\predictions_test.npy'
traces = np.load(file_path, allow_pickle=True)
traces[np.isnan(traces)] = 0
print(traces.shape)
ROI = 0
roinum = sample1.shape[0]
for ROI in range(roinum):
    plt.figure()
    figname = str(ROI) + 'png'
    plt.title(figname)
    plt.plot(traces[ROI,:])
    plt.plot(sample1[ROI,:] - np.min(sample1[ROI,:]))
    plt.savefig('A:\\Cascade-master\\vis\\' + figname, dpi=200)
    plt.close()

            
            
            
            
            
            
            
            
            










