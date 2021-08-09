# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:46:35 2021

@author: MSBak
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# import random


import sys; 
msdir = 'C:\\Users\\skklab\\Documents\\mscode'; sys.path.append(msdir)
sys.path.append('D:\\mscore\\code_lab\\')
import msFunction
import os  
try: import pickle5 as pickle
except: import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import random
import time
from tqdm import tqdm
from scipy import stats

MAXSE = 20
#% mFunction

def msROC(class0, class1):
    import numpy as np
    from sklearn import metrics
    
    pos_label = 1; roc_auc = -np.inf; fig = None

    class0 = np.array(class0); class1 = np.array(class1)
    class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
    
    anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
    predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)       
    fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
    maxix = np.argmax((1-fpr) * tpr)
    specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
    accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
    roc_auc = metrics.auc(fpr,tpr)
    
    return accuracy, roc_auc


def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

def ms_syn(target_signal=None, target_size=None):
    downratio = target_signal.shape[0] / target_size
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
    return allo

def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout

#% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['behavss2']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열
signalss_raw = msdata_load['signalss_raw']

highGroup = msGroup['highGroup']    # 5% formalin
midleGroup = msGroup['midleGroup']  # 1% formalin
lowGroup = msGroup['lowGroup']      # 0.25% formalin
salineGroup = msGroup['salineGroup']    # saline control
restrictionGroup = msGroup['restrictionGroup']  # 5% formalin + restriciton
ketoGroup = msGroup['ketoGroup'] # 5% formalin + keto 100
lidocaineGroup = msGroup['lidocaineGroup'] # 5% formalin + lidocaine
capsaicinGroup = msGroup['capsaicinGroup'] # capsaicin
yohimbineGroup = msGroup['yohimbineGroup'] # 5% formalin + yohimbine
pslGroup = msGroup['pslGroup'] # partial sciatic nerve injury model
shamGroup = msGroup['shamGroup']
adenosineGroup = msGroup['adenosineGroup']
highGroup2 = msGroup['highGroup2']
CFAgroup = msGroup['CFAgroup']
chloroquineGroup = msGroup['chloroquineGroup']
itSalineGroup = msGroup['itSalineGroup']
itClonidineGroup = msGroup['itClonidineGroup']
ipsaline_pslGroup = msGroup['ipsaline_pslGroup']
ipclonidineGroup = msGroup['ipclonidineGroup']
gabapentinGroup = msGroup['gabapentinGroup']
beevenomGroup =  msGroup['beevenomGroup']
oxaliGroup =  msGroup['oxaliGroup']
glucoseGroup =  msGroup['glucoseGroup']
PSLscsaline =  msGroup['PSLscsaline']
highGroup3 =  msGroup['highGroup3']
PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']
KHUsham = msGroup['KHUsham']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = bahavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = downsampling(behav_tmp, signalss[SE][se].shape[0])

#%% grouping
group_pain_training = []
group_nonpain_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    if not SE in [179, 181]: # ROI 매칭안되므로 임시 제거
        for se in range(MAXSE):
            painc, nonpainc, test_only = [], [], []
            # snu
            if True:
                nonpainc.append(SE in salineGroup and se in [0,1,2,3,4])
                nonpainc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [0])
                painc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [1])
                
                painc.append(SE in CFAgroup and se in [1,2])
                painc.append(SE in capsaicinGroup and se in [1])
                
                # snu psl pain
                painc.append(SE in pslGroup and se in [1,2])
                nonpainc.append(SE in pslGroup and se in [0])
                nonpainc.append(SE in shamGroup and se in [0,1,2])
                
                # snu psl+
                # painc.append(SE in ipsaline_pslGroup and se in [1,2])
                painc.append(SE in ipsaline_pslGroup and se in [1,3])
                nonpainc.append(SE in ipsaline_pslGroup and se in [0])
                painc.append(SE in ipclonidineGroup and se in [1,3])
                nonpainc.append(SE in ipclonidineGroup and se in [0])
            
                # GBVX 30 mins
                if True:
                    GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
                    nonpainc.append(SE in GBVX and se in [0,1])
                    # nonpainc.append(SE in [164, 166] and se in [2,3,4,5])
                    # nonpainc.append(SE in [167] and se in [4,5,6,7])
                    # nonpainc.append(SE in [172] and se in [4,5,7,8])
                    # nonpainc.append(SE in [174] and se in [4,5])
                    # nonpainc.append(SE in [177,179,181] and se in [2,3,6,7,10,11])
                    # painc.append(SE in [179] and se in [8,9])
                    # painc.append(SE in [181] and se in [4,5])
            
                # snu oxali
                if True:
                    painc.append(SE in oxaliGroup and se in [1])
                    painc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [2])
                    nonpainc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [3])
                    nonpainc.append(SE in [188, 189, 200, 201] and se in [2])
                    nonpainc.append(SE in glucoseGroup and se in [0,1,2,3,4])
            
            
            # khu formalin
            painc.append(SE in list(range(230, 239)) and se in [1])
            painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
            painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])
            
            nonpainc.append(SE in list(range(230, 239)) and se in [0])
            nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
            nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
            nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
            
            # khu psl
            nonpainc.append(SE in PSLgroup_khu and se in [0])
            painc.append(SE in PSLgroup_khu and se in [1,2])
            
            nonpainc.append(SE in morphineGroup and se in [0,1])
            nonpainc.append(SE in morphineGroup and se in [10,11,12])
            painc.append(SE in morphineGroup and se in [2,3,4,5,6,7,8,9])
            
            nonpainc.append(SE in KHUsham and se in range(0,10))
            nonpainc.append(SE in KHUsham and se in range(10,13))
             
            # PD
            nonpainc.append(SE in PDnonpain and se in list(range(2,10)))
            nonpainc.append(SE in PDnonpain and se in list(range(0,2)))
            painc.append(SE in PDpain and se in list(range(2,6)))
            nonpainc.append(SE in PDpain and se in list(range(0,2)))
            
            if [SE, se] in [[285, 4],[290, 5]]: continue # 시간짧음, movement 불일치
            
            if np.sum(np.array(painc)) > 0: group_pain_training.append([SE, se])   
            if np.sum(np.array(nonpainc)) > 0: group_nonpain_training.append([SE, se])

#%% 데이터 만들기

X0 = np.zeros((N, MAXSE)) * np.nan
X1 = np.zeros((N, MAXSE)) * np.nan
for SE in range(N):
    for se in range(len(signalss[SE])):
        label = None
        if [SE, se] in group_nonpain_training: label = [1, 0]
        if [SE, se] in group_pain_training: label = [0, 1]
        
        if label == [1,0]: X0[SE,se] = np.mean(signalss[SE][se])
        if label == [0,1]: X1[SE,se] = np.mean(signalss[SE][se])
        
X0 = msFunction.nanex(X0)
X1 = msFunction.nanex(X1)

#%%

def ms_EMalgorithm(X, vissw=True, epochs=10):
    # init
    n_data = len(X)
    rn = np.max([int(n_data/10), 5])
    sample0 = random.sample(list(X), rn)
    sample1 = random.sample(list(X), rn)
    mu = np.mean(sample0)
    sigma = np.std(sample0)
    pi = 0.5
    mu2 = np.mean(sample1)
    sigma2 = np.std(sample1)
    pi2 = 0.5
   
    
    if vissw:
        plt.figure()
        plt.title('init')
        plt.hist(X, bins=100, density=True)
        mm = np.mean(X)
        ss = np.std(X)
        x = np.linspace(mm-5*ss, mm+5*ss, 1000)
        p = stats.norm.pdf(x, mu, sigma); plt.plot(x, p)
        p = stats.norm.pdf(x, mu2, sigma2); plt.plot(x, p)
    
    
    # estimation
    for epoch in range(epochs):
        l0 = stats.norm.pdf(X, mu, sigma)
        l1 = stats.norm.pdf(X, mu2, sigma2)
        
        pi = np.sum(l0)/n_data
        l0 = l0 * pi / (l0 * pi + l1 * pi2)
        l1 = l1 * pi2 / (l0 * pi + l1 * pi2)
        mu0 = np.sum((X * l0)) / np.sum(l0)
        std0 = np.sqrt(np.sum((X - mu0)**2 * l0)  / np.sum(l0))
        
        pi2 = np.sum(l1)/n_data
        mu1 = np.sum((X * l1)) / np.sum(l1)
        std1 = np.sqrt(np.sum((X - mu1)**2 * l1)  / np.sum(l1))
        
        if vissw:
            plt.figure()
            plt.title(epoch)
            plt.hist(X, bins=100, density=True)
            p = stats.norm.pdf(x, mu0, std0); plt.plot(x, p)
            p = stats.norm.pdf(x, mu1, std1); plt.plot(x, p)
        
        # update
        mu = mu0; sigma = std0
        mu2 = mu1; sigma2 = std1
        
    return mu, sigma, mu2, sigma2
    


mu, sigma, mu2, sigma2 = ms_EMalgorithm(X0, vissw=False, epochs=10)

mm = np.mean(X0)
ss = np.std(X0)
x = np.linspace(mm-5*ss, mm+5*ss, 1000)
p = stats.norm.pdf(x, mu, sigma); plt.plot(x, p, c='b')
p = stats.norm.pdf(x, mu2, sigma2); plt.plot(x, p, c='b')


mu, sigma, mu2, sigma2 = ms_EMalgorithm(X1, vissw=False, epochs=10)

mm = np.mean(X1)
ss = np.std(X1)
x = np.linspace(mm-5*ss, mm+5*ss, 1000)
p = stats.norm.pdf(x, mu, sigma); plt.plot(x, p, c='r')
p = stats.norm.pdf(x, mu2, sigma2); plt.plot(x, p, c='r')

    
    
label = np.zeros(n_data) * np.nan
for i in range(n_data):
    p0 = stats.norm.pdf(X[i], mu, sigma)
    p1 = stats.norm.pdf(X[i], mu2, sigma2)
    
    if p0 > p1: label[i] = 0
    else: label[i] = 1

p0_ix = np.where(label==0)[0]
p1_ix = np.where(label==1)[0]

plt.figure()
plt.hist(X, bins=100, density=True)
p = stats.norm.pdf(x, np.mean(X[p0_ix]), np.std(X[p0_ix])); plt.plot(x, p)
p = stats.norm.pdf(x, np.mean(X[p1_ix]), np.std(X[p1_ix])); plt.plot(x, p)
















