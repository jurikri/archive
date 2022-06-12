# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:19:13 2020

@author: MSBak
"""

#%%

import numpy as np

def nanex(array1):
    array1 = np.array(array1)
    array1 = array1[np.isnan(array1)==0]
    return array1

def msarray(listlike, nonesw=False):
    out = []
    [out.append([]) for u in range(listlike[0])]

    if nonesw: dlevel = len(listlike)
    
    if nonesw and dlevel==1:
        for i in range(listlike[0]):
            out[i] = None
    
    if len(listlike) > 1:
        for i in range(listlike[0]):
            [out[i].append([]) for u in range(listlike[1])]
            
            if nonesw and dlevel==2:
                for j in range(listlike[1]):
                    out[i][j] = None
            
            if len(listlike) > 2:
                for j in range(listlike[1]):
                    [out[i][j].append([]) for u in range(listlike[2])]
                    
                    if nonesw and dlevel==3:
                        for k in range(listlike[2]):
                            out[i][j][k] = None
                    
                    if len(listlike) > 3:
                        for k in range(listlike[2]):
                            [out[i][j][k].append([]) for u in range(listlike[3])]
                        
                            if nonesw and dlevel==4:
                                for l in range(listlike[3]):
                                    out[i][j][k][l] = None
                            
    return out

def msROC(class0, class1, repeat=1, pcolor=None, mslabel='', figsw=False):
    import numpy as np
    from sklearn import metrics
    import matplotlib.pyplot as plt
    
    
    pos_label = 1; roc_auc = -np.inf; fig = None
    class0 = np.array(class0); class1 = np.array(class1)
    class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
    anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
    predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)  
    for ii in range(repeat):   
        fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
        maxix = np.argmax((1-fpr) * tpr)
        specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
        accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
        roc_auc = metrics.auc(fpr,tpr)
        if roc_auc >= 0.5: break
        pos_label = 0
        
    msdict = {}
    if figsw:
        sz = 1
        fig = plt.figure(1, figsize=(7*sz, 5*sz))
        lw = 2
        if pcolor is None:
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        else:
            plt.plot(fpr, tpr, color=pcolor,
                     lw=lw, label=mslabel + ' ' + str(round(roc_auc, 2)))
            
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    #        plt.title('ROC')
        plt.legend(loc="lower right")
        msdict = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        # plt.show()

    return accuracy, roc_auc, fig, msdict

def ms_smooth(mssignal=None, ws=None):
    import numpy as np
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout


def downsampling(mssignal, wanted_size): # scipy.signal.resample 로 대체
    import numpy as np
    
    if len(mssignal.shape)==1:
        mssignal = np.reshape(mssignal, (1,mssignal.shape[0]))
    
    downratio = mssignal.shape[1]/wanted_size
    downsignal = np.zeros((mssignal.shape[0], int(wanted_size))) * np.nan

    for frame in range(int(wanted_size)):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[:, frame] = np.mean(mssignal[:,s:e], axis=1)
        
    return np.array(downsignal)

def mslinear_regression(x,y):
    import numpy as np
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m
    return m, b # bx+a

def ms_wavelet(xdata=None, SR=None): # 확인 후 속도 개선
    import numpy as np
    # input shape -> trial x xlen x channel
    # SR = 1000

    tn = xdata.shape[0]
    xlen = xdata.shape[1]
    cn = xdata.shape[2]
    
    msout = []
    min_freq =  1;
    max_freq = 40;
    num_frex = 40;
    frex = np.linspace(min_freq,max_freq,num_frex);
    
    range_cycles = [4, 10];
    beta0 = np.log10(range_cycles[0])
    beta1 = np.log10(range_cycles[1])

    s = np.logspace(beta0, beta1, num_frex) / (2*np.pi*frex)
    wavtime = np.arange(-2,2,1/SR)
    half_wave = int((len(wavtime))/2)
    
    nWave = len(wavtime)
    nData = xlen
    nConv = nWave + nData - 1
        
    for channel2use in range(cn):
        tf = np.zeros((len(frex), xlen, tn)) * np.nan;
        
        for fi in range(len(frex)):
            # create wavelet and get its FFT
            # the wavelet doesn't change on each trial...
            wavelet  = np.exp(2*1j*np.pi*frex[fi] * wavtime) * np.exp(-wavtime**2 / (2*s[fi]**2));

            waveletX = np.fft.fft(wavelet,n=nConv);
            waveletX = waveletX / max(waveletX);

            for trial_i in range(tn):
                dataX = np.fft.fft(xdata[trial_i, :, channel2use], n=nConv)
                ms_as = np.fft.ifft(waveletX * dataX);
                ms_as = ms_as[half_wave+1:-half_wave+2];
                tf[fi,:,trial_i] = np.square(np.abs(ms_as));
        msout.append(tf)
    msout = np.array(msout)
    return msout

def ms_wavelet2(xdata=None, SR=None): # 확인 후 속도 개선
    import numpy as np
    # input shape -> trial x xlen x channel
    # SR = 1000

    tn = xdata.shape[0]
    xlen = xdata.shape[1]
    cn = xdata.shape[2]
    
    msout = []
    min_freq =  1;
    max_freq = 100;
    num_frex = 80                                ;
    frex = np.linspace(min_freq,max_freq,num_frex);
    
    range_cycles = [4, 10];
    beta0 = np.log10(range_cycles[0])
    beta1 = np.log10(range_cycles[1])

    s = np.logspace(beta0, beta1, num_frex) / (2*np.pi*frex)
    wavtime = np.arange(-2,2,1/SR)
    half_wave = int((len(wavtime))/2)
    
    nWave = len(wavtime)
    nData = xlen
    nConv = nWave + nData - 1
        
    for channel2use in range(cn):
        tf = np.zeros((len(frex), xlen, tn)) * np.nan;
        
        for fi in range(len(frex)):
            # create wavelet and get its FFT
            # the wavelet doesn't change on each trial...
            wavelet  = np.exp(2*1j*np.pi*frex[fi] * wavtime) * np.exp(-wavtime**2 / (2*s[fi]**2));

            waveletX = np.fft.fft(wavelet,n=nConv);
            waveletX = waveletX / max(waveletX);

            for trial_i in range(tn):
                dataX = np.fft.fft(xdata[trial_i, :, channel2use], n=nConv)
                ms_as = np.fft.ifft(waveletX * dataX);
                ms_as = ms_as[half_wave+1:-half_wave+2];
                tf[fi,:,trial_i] = np.square(np.abs(ms_as));
        msout.append(tf)
    msout = np.array(msout)
    return msout

def ms_minmax(X): # [0,1]
    X = np.array(X)
    msmin = np.min(X)
    msmax = np.max(X)
    
    X_std = (X - msmin) / (msmax - msmin)
    X_scaled = X_std * (1 - 0) + 0
    return X_scaled





















