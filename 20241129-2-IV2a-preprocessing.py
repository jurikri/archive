# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:14:51 2024

@author: PC

 {'1023': 1,
  '1072': 2,
  '276': 3,
  '277': 4,
  '32766': 5,
  '768': 6,
  '769': 7,
  '770': 8,
  '771': 9,
  '772': 10})


"""


import os
import glob
import numpy as np
import msanalysis
import matplotlib.pyplot as plt

# 1. E:\BCI competition\IV2\BCICIV_2b_gdf 경로에 모든 *.gdf 파일 리스트업
gdf_path = r'E:\BCI competition\IV2\BCICIV_2a_gdf'  # 경로 설정
gdf_files = glob.glob(os.path.join(gdf_path, '*.gdf'))  # .gdf 파일 리스트업

# 선별된 파일들을 저장할 리스트
selected_files = []

# 2. 파일명에서 subject ID와 session number 추출
for file in gdf_files:
    file_name = os.path.basename(file)  # 파일명만 추출
    subject_id = file_name[1:3]  # subject ID 추출
    # session_number = file_name[3:5]  # session number 추출

    # 3. session number가 01 또는 02인 경우 선별하여 저장
    # if session_number in ['01', '02', '03', '04', '05']:
    selected_files.append([file, subject_id])
selected_files = np.array(selected_files)

SR = 250
# 선별된 파일 출력
import pickle
import mne

# if not os.path.isfile(rpath + 'XYtmpsave.pkl'):
print("선별된 파일 목록:")
from tqdm import tqdm 

rpath = r'E:\BCI competition\IV2\BCICIV_2a_gdf' + '\\'

X_power, X_phase, Z = [], [], []
for i in tqdm(range(len(selected_files[:,0]))):
    filename_with_ext = os.path.basename(selected_files[:,0][i])
    filename, ext = os.path.splitext(filename_with_ext)
    
    morlet_savepath = rpath + 'morlet_' +  filename + '.pkl'
    raw_data = mne.io.read_raw_gdf(selected_files[:,0][i], preload=True)
    
    # channel_names = raw_data.ch_names
    # np.array(channel_names)[[7,9,11]]

    if not os.path.isfile(morlet_savepath):
        eeg_data = raw_data.get_data()
        eeg_data_onlyeeg = eeg_data[[7,9,11], :]
        
        power, phase = msanalysis.msmain(EEGdata_ch_x_time = eeg_data_onlyeeg, SR=SR)
        
        # pickle 파일로 저장
        msdict = {'power':power, 'phase':phase}
        with open(morlet_savepath, 'wb') as file:
            pickle.dump(msdict, file)
          
    with open(morlet_savepath, 'rb') as file:
        msdict = pickle.load(file)
    power = msdict['power']; phase = msdict['phase']
    
    # channel_names = raw_data.ch_names
    events = mne.events_from_annotations(raw_data)
    events2 = np.array(events[0])
    ix6 = np.where(events2[:,2]==6)[0]
    
    # t = ix9[0]
    pre_baseline = None
    for t in range(0, len(ix6)-1):
        ytmp = None
        log = events2[:,2][ix6[t]:ix6[t+1]]
        if not 1 in log:
            c0 = 7 in log
            c1 = 8 in log
            if c0 and c1: ytmp = 'both'
            if c0 and not(c1): ytmp = 'left'
            elif c1 and not(c0): ytmp = 'right'
            
            tstamp = events2[ix6[t-1]][0]  
            one_trial_power = power[:, :, int(tstamp) : int(tstamp + SR*8.5)] # 0~8.5 one trial
            one_trial_phase = phase[:, :, int(tstamp) : int(tstamp + SR*8.5)] # 0~8.5 one trial
            pre_baseline = [one_trial_power[:, :, :int(SR*3)], one_trial_phase[:, :, :int(SR*3)]]
            
            if not ytmp is None:
                tstamp = events2[ix6[t]][0]  
                one_trial_power = power[:, :, int(tstamp) : int(tstamp + SR*8.5)] # 0~8.5 one trial
                one_trial_phase = phase[:, :, int(tstamp) : int(tstamp + SR*8.5)] # 0~8.5 one trial
        
                # baseline = [one_trial_power[:, :, :int(SR*3)], one_trial_phase[:, :, :int(SR*3)]]
                # MI = [one_trial_power[:, :, int(SR*3):int(SR*6)], one_trial_phase[:, :, int(SR*3):int(SR*6)]]
                
                def downsample(baseline_power):  
                    time_tlen = baseline_power.shape[2]
                    baseline_power_down = []
                    dt = 5
                    for t3 in range(0, time_tlen, dt):
                        baseline_power_down.append(np.median(baseline_power[:,:,t3:t3+dt], axis=2))
                    baseline_power_down = np.array(baseline_power_down)
                    return baseline_power_down

                baseline_power = np.array(one_trial_power[:, :, :int(SR*3)])
                baseline_power_down = downsample(baseline_power)
                
                MI_power = np.array(one_trial_power[:, :, int(SR*3):int(SR*6)])
                MI_power_down = downsample(MI_power)
                
                def phase_calc(baseline_phase):
                    baseline_phase_calc = []
                    for ch in [0,2]:
                        baseline_phase_calc.append(np.median((baseline_phase[ch] - baseline_phase[1]), axis=1))
                    return np.array(baseline_phase_calc)
                
                baseline_phase = np.array(one_trial_phase[:, :, :int(SR*3)])
                baseline_phase_calc = phase_calc(baseline_phase)
                
                MI_phase = np.array(one_trial_phase[:, :, int(SR*3):int(SR*6)])
                MI_phase_calc = phase_calc(MI_phase)
 

                X_power.append([baseline_power_down])
                X_phase.append([baseline_phase_calc])
                Z.append(['baseline', t, selected_files[i,1]])
                
                X_power.append([MI_power_down])
                X_phase.append([MI_phase_calc])
                Z.append([ytmp, t, selected_files[i,1]])
                
psavepath = r'F:\BCIIV_2_data' + '\\IV2a_XZ_v4.pkl'
msdict = {'X_power':X_power, 'X_phase':X_phase, 'Z':Z}
with open(psavepath, 'wb') as file:
    pickle.dump(msdict, file)
   

#%% vis

Y = []
for i in tqdm(range(len(Z))):
    ytmp = [0,0]
    if Z[i][0] == 'baseline': ytmp = [1,0]
    if Z[i][0] in ['left', 'right', 'both']: ytmp = [0,1]
    Y.append(ytmp)
Y = np.array(Y)

class0 = np.where(Y[:,0]==1)[0]
class1 = np.where(Y[:,1]==1)[0]

X_power = np.array(X_power)
X_power_nmr = []
for i in tqdm(range(len(X_power))):
    pre_allo = np.zeros(X_power[i].shape) * np.nan
    for ch in range(3):
        for fi in range(50):
            pre_allo[0,:,ch,fi] = X_power[i][0,:,ch,fi] / np.mean(X_power[i][0,:,ch,fi])
    X_power_nmr.append(pre_allo)
X_power_nmr = np.array(X_power_nmr)

ch, fi = 1, 10
# plt.plot(np.median(X_power[class0][:,0,:,ch,fi], axis=0))
# plt.plot(np.median(X_power[class1][:,0,:,ch,fi], axis=0))

plt.plot(np.median(X_power_nmr[class0][:,0,:,ch,fi], axis=0))
plt.plot(np.median(X_power_nmr[class1][:,0,:,ch,fi], axis=0))











