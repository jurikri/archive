# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 09:49:01 2021

@author: MSBak
"""

    #%%

# raw_data

# filtered_data


if False:
    filepath = 'A:\\tmp1.npy'
    np.save(filepath, b, allow_pickle=True, fix_imports=True)
    filepath = 'A:\\tmp2.npy'
    np.save(filepath, xin, allow_pickle=True, fix_imports=True)
    filepath = 'A:\\tmp3.npy'
    np.save(filepath, raw_data, allow_pickle=True, fix_imports=True)
    filepath = 'A:\\tmp4.npy'
    np.save(filepath, zi, allow_pickle=True, fix_imports=True)

# tmp = raw_data[0,:]-np.mean(raw_data, axis=0)
    



# y = butter_bandpass_filter(tmp, 0.5, 50, 1000, order=1)

# print(np.sum(np.isnan(y)), np.mean(y))

# f = scipy.signal.firwin(6600+1, [0.5, 50], width=None, window='hamming', pass_zero=True, scale=True, fs=1000)
# plt.plot(f)

# scipy.signal.lfilter(f, [1.0], tmp)

#%%
def matlab_FIRfilter(msdata = None, SR=1000):
    import numpy as np
    import scipy.signal as scipy_signal
    
    def matlab_windows(t='hamming', filtorder=6600):
        if t == 'hamming':
            m = filtorder+1
            
            isOddLength = np.mod(m, 2)
            if isOddLength: x = np.array(range(0,int((m-1)/2+1))) / (m-1)
            else: x = np.array(range(0,int((m/2)-1+1))) / (m-1)
                
            a = 0.54 # hamming 에서 fix
            w = a - (1 - a) * np.cos(2 * np.pi * x)
            
            if isOddLength:
                ix = list(range(len(w)))
                rix = np.sort(ix)[::-1][1:]
                w = np.concatenate((w, w[rix]), axis=0)    
            else:
                ix = list(range(len(w)))
                rix = np.sort(ix)[::-1]
                w = np.concatenate((w, w[rix]), axis=0)
        return w

    gfiltorder = int(6600 * (SR/1000))
    winArray = matlab_windows(filtorder = gfiltorder)
    # SR = 1000
    cutoffArray = np.array([0.2500, 50.2500]) # 0.5 / 50 기준
    fNyquist = SR/2
    

# b = firws(g.filtorder, cutoffArray / fNyquist, winArray);


    m = gfiltorder
    f = cutoffArray / fNyquist
    t = winArray
    
    f = f / 2
    w = t

    def matlab_fkernel(m,f1,w):
        m1 = np.array(range(int(-m/2),int(m/2)+1))
        b = np.zeros(len(m1)) * np.nan
        
        b[m1==0] = 2*np.pi*f1
        b[m1!=0] = np.sin(2 * np.pi * f1 * m1[m1!=0]) / m1[m1!=0]
        b = b * w
        b = b/np.sum(b)
        return b

    def matlab_fspecinv(b):
        b = -b
        b[int((len(b)-1-1)/2+1)] = b[int((len(b)-1-1)/2+1)]+1
        return b

    b = matlab_fkernel(m, f[0], w);

    if len(f) == 2:
        b = b + matlab_fspecinv(matlab_fkernel(m, f[1], w));
        # plt.plot(b)
        if True:
            b = matlab_fspecinv(b); # 특정조건문임 일단 그냥 적용
            # plt.figure()
            # plt.plot(b)
    
    
# firfilt


# np.mean(raw_data2)


    groupDelay = int((len(b) - 1) / 2)
    dcArray = [1, msdata.shape[1]+1]

    iDc = 0 # 유효하지 않은 loop문 var
    
    ziDataDur = np.min([groupDelay, dcArray[iDc + 1] - dcArray[iDc]]);

    chaninds = list(range(len(msdata)))

    ms1 = msdata[chaninds][:, np.array(np.ones(groupDelay) * dcArray[iDc], dtype=int)-1]
    ms2 = msdata[chaninds, dcArray[iDc]-1:(dcArray[iDc] + ziDataDur - 1)]
    xin =  np.concatenate((ms1,ms2), axis=1)

    # print(np.mean(ms1), np.mean(ms2))
    
    # zi = signal.lfilter_zi(b, a)
    y, zi = scipy_signal.lfilter(b, 1, xin, axis=1, zi=np.zeros((len(chaninds),len(b)-1)))
    
    # print(np.mean(b), np.mean(xin))
    # print(np.mean(y), np.mean(zi))
    
    # blockArray = [(dcArray[iDc] + groupDelay)-1:nFrames:(dcArray(iDc + 1) - 1) dcArray(iDc + 1)];
    # for iBlock = 1:(length(blockArray) - 1)
    
    #     % Filter the data
    #     [EEG.data(chaninds, (blockArray(iBlock) - groupDelay):(blockArray(iBlock + 1) - groupDelay - 1)), zi] = ...
    #         filter(b, 1, double(EEG.data(chaninds, blockArray(iBlock):(blockArray(iBlock + 1) - 1))), zi, 2);
    
    #     % Update progress indicator
    #     [step, strLength] = mywaitbar((blockArray(iBlock + 1) - groupDelay - 1), size(EEG.data, 2), step, nSteps, strLength);
    # end
    
    
    nFrames = 1000 # fix
    # [temp, zi] = filter(b, 1, , [], 2);
    
    
    ms3 = np.array((range((dcArray[iDc] + groupDelay)-1, (dcArray[iDc + 1] - 1), nFrames)))
    ms4 = np.array([dcArray[iDc + 1]])
    blockArray = np.concatenate((ms3, ms4), axis=0);
    
    # ms3.shape
    
    for iBlock in range(len(blockArray)-1):
        # Filter the data
        
        xin = msdata[chaninds, blockArray[iBlock]-1:(blockArray[iBlock + 1] - 1)]
        y, zi = scipy_signal.lfilter(b, 1, xin, axis=1, zi=zi)
        msdata[chaninds, (blockArray[iBlock] - groupDelay) : (blockArray[iBlock + 1] - groupDelay - 1)+1] = y
    
    
    # temp = filter(b, 1, double(EEG.data(chaninds, ones(1, groupDelay) * (dcArray(iDc + 1) - 1))), zi, 2);
    xin = msdata[chaninds][:,np.array(np.ones(groupDelay) * (dcArray[iDc + 1] - 1)-1, dtype=int)]
    # print(np.mean(xin))
    temp, _ = scipy_signal.lfilter(b, 1, xin, axis=1, zi=zi)
    # print(np.mean(b))
    # print(np.mean(temp))
    
    
    xin = temp[:, (-ziDataDur+1-1):];
    # print(xin.shape, np.mean(xin))
    msdata[chaninds, (dcArray[iDc + 1] - ziDataDur)-1:(dcArray[iDc + 1] - 1)] = xin
    
    return msdata

#%%
if False:
    raw_data2 = np.array(raw_data)
    
    # average referencing
    raw_data2 = raw_data2 - np.mean(raw_data2, axis=0) 
    raw_data2_tmp = np.array(raw_data2)
    
    if False:
        filepath = 'A:\\tmp3.npy'
        np.save(filepath, raw_data2, allow_pickle=True, fix_imports=True)
    
    
    raw_data3 = matlab_FIRfilter(msdata = raw_data2)
    
    s = 1000
    e = 4000
    
    plt.plot(raw_data2_tmp[0,s:e])
    plt.plot(raw_data3[0,s:e])
    plt.plot(filtered_data[0,s:e])
    
    print(np.mean(filtered_data[:,0]), np.mean(raw_data[:,0]), np.mean(raw_data2_tmp[:,0]), np.mean(raw_data3[:,0]))

































