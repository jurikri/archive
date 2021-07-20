# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:36:50 2021

@author: MSBak
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mslinear_regression(x,y):
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m

    return m, b # bx+a

path = 'C:\\mass_save\\itch\\s1214-1\\'
fname = 'Results_5HT_before_1229.csv'

mouselist = []
mouselist.append('Results_5HT_before_1229.csv')
mouselist.append('Results_cap_1-3_1229.csv')
mouselist.append('Results_5HT_5-7_1229.csv')

for se in range(len(mouselist)):
    loadpath = path + mouselist[se]
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
    for ws in [1, 3, 5]:
        for ROI in np.array(np.arange(0, 10, 5), dtype=int):
            baseratio = 0.05
            msout = ms_smooth(mssignal=allo[:,ROI], ws=ws)
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
            # slope = np.mean(diff) - np.std(diff)
            slope = 0
            for k in range(len(diff)):
                if diff[k] < slope  and msout[k] > cut and peak_sw:
                    peaks_ix.append(int(k))
                    peak_sw = False
                if diff[k] > slope  and not(peak_sw):
                    peak_sw = True
            peaks_ix = np.array(peaks_ix)
        
            figsavename = 'C:\\mass_save\\itch\\figsave\\method2\\' + 'se_'+str(se) + '_ws_'+str(ws) + '_ROI_'+str(ROI)
            if False:
                plt.figure(); plt.title('all_peak_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
                plt.plot(msout); plt.scatter(peaks_ix, msout[peaks_ix], c='r'); plt.plot(np.ones(len(msout)) * cut) 
                plt.savefig(figsavename + '_all_peak.png', dpi=100); plt.close()
            
            # method1, 두 paek 사이 minimum에 2배를 넘는지
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
                # print(peaks_ix2)
                if passw: break
            
            if False:
                plt.figure(); plt.title('method1_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
                plt.plot(msout); plt.scatter(peaks_ix2, msout[peaks_ix2], c='r'); plt.plot(np.ones(len(msout)) * cut)
                plt.savefig(figsavename + '_method1.png', dpi=100); plt.close()
                
            # method2, 첫 paek이후 reset 조건
            peaks_ix3 = np.array(peaks_ix)
            onsw = True
            reset_thr = 0.5
            saved = []; t = 0

            for t2 in range(len(msout)):
                if t2 < t: continue
                if onsw and t2 in peaks_ix3:
                    onsw = False
                    saved.append(t2)
                    
                    highpeaks = [t2]; onsw2 = False; accu = 0
                    for t in range(t2+1, len(msout)):
                        accu += 1
                        for t3 in [highpeaks[0]]:
                            if msout[t3]*reset_thr > msout[t]: onsw2 = True
                                
                        if onsw2: onsw = True; break
                        if t in peaks_ix3: highpeaks.append(t) # peak 갱신
                
            if True:
                plt.figure(); plt.title('method2_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
                plt.plot(msout); plt.scatter(saved, msout[saved], c='r'); plt.plot(np.ones(len(msout)) * cut)
                plt.savefig(figsavename + '_method2.png', dpi=500); plt.close()
        
                
            # method3, 2에서 개량, 모든 peak 대상으로 점검, 시간이후 자동 reset
            peaks_ix3 = np.array(peaks_ix)
            onsw = True
            reset_thr = 0.5
            saved = []; t = 0

            for t2 in range(len(msout)):
                if t2 < t: continue
                if onsw and t2 in peaks_ix3:
                    onsw = False
                    saved.append(t2)
                    
                    highpeaks = [t2]; onsw2 = False; accu = 0
                    for t in range(t2+1, len(msout)):
                        accu += 1
                        for t3 in highpeaks:
                            if msout[t3]*reset_thr > msout[t]: onsw2 = True
                                
                        if onsw2 or accu > 10: onsw = True; break
                        if t in peaks_ix3: highpeaks.append(t) # peak 갱신
                
            if True:
                plt.figure(); plt.title('method3_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
                plt.plot(msout); plt.scatter(saved, msout[saved], c='r'); plt.plot(np.ones(len(msout)) * cut)
                plt.savefig(figsavename + '_method3.png', dpi=500); plt.close()
            
            # method4, 3에서 개량, 하락중이면 peak 무효, 
            peaks_ix3 = np.array(peaks_ix)
            onsw = True
            reset_thr = 0.35
            saved = []; t = 0

            for t2 in range(len(msout)):
                if t2 < t: continue
                if onsw and t2 in peaks_ix3:
                    onsw = False
                    saved.append(t2)
                    
                    highpeaks = [t2]; onsw2 = False; accu = 0
                    for t in range(t2+1, len(msout)):
                        accu += 1
                        for t3 in highpeaks:
                            if msout[t3]*reset_thr > msout[t]: onsw2 = True
                                
                        if onsw2 or accu > 10: onsw = True; break
                        if t in peaks_ix3: highpeaks.append(t) # peak 갱신
            
            saved2 = []
            for peak in saved:
                cluster = msout[np.max([peak-7, 0]):np.min([peak+2, len(msout)])] 
                x = range(len(cluster))
                y = cluster
                m, b = mslinear_regression(x,y)
                if m > 0: saved2.append(peak)
 
            if True:
                plt.figure(); plt.title('method4_' + mouselist[se] + '_ws_'+str(ws) + '_ROI_'+str(ROI))
                plt.plot(msout); plt.scatter(saved2, msout[saved2], c='r'); plt.plot(np.ones(len(msout)) * cut)
                plt.savefig(figsavename + '_method4.png', dpi=500); plt.close()

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            











