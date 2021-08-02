# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:05:21 2021

ref@author: MSBak
"""


#%% oir -> tif
import os
path = 'C:\\mass_save\\auto_turboreg_run\\work3'

# must follow these rules!
# 1. C: <- should be typed mannualy. Do not copy and paste!
# 2. \ -> \\
# 3. / -> \\

file_list = os.listdir(path)
path_fix = str(path)
path_fix = path_fix.replace('\\', '/')

macro_list = []
for i in range(len(file_list)):
    if file_list[i].endswith('.oir'):
        filepath = path + '\\' + file_list[i]
        macro = 'run("Bio-Formats Importer", "open=' + path_fix + '/' + file_list[i] + \
            ' " + "autoscale color_mode=Default"); saveAs("Tiff", "' + path_fix + '/' + file_list[i] + '.tif ");' + '\n'
        macro_list.append(macro)
        
with open(path + '\\hello.txt', 'w') as file:    # hello.txt 파일을 쓰기 모드(w)로 열기
    file.writelines(macro_list)

import sys; sys.exit()


#%% turboreg and save
from pystackreg import StackReg # pip install pystackreg
from skimage import io # pip install scikit-image (체크필요)
import imageio # pip install imageio
import numpy as np
import matplotlib.pyplot as plt
import time

file_list = os.listdir(path)
listsave = []
for i in range(len(file_list)):
    if file_list[i].endswith('.tif'):
        filepath = path + '\\' + file_list[i]
        listsave.append(filepath) 
print('total sample #', len(listsave))

i  = 0
for i in range(len(listsave)):       
    filepath = listsave[i]
    print(i, '/', len(listsave), '>>>', filepath)
    img0 = io.imread(filepath) # 3 dimensions : frames x width x height
    # print(np.mean(np.std(img0, axis=0)))
    sr = StackReg(StackReg.RIGID_BODY)
    # register to mean image
    start = time.time()  
    out_mean = sr.register_transform_stack(img0, reference='mean')
    print("time :", time.time() - start) 

    out_mean2 = np.array(out_mean, dtype='uint32')
    savename = filepath + '_python_reg32.tif'
    imageio.mimwrite(savename, out_mean2)
    print('save as', savename)
    
    # out_mean2 = np.array(out_mean, dtype='uint16')
    # savename = filepath + '_python_reg16.tif'
    # imageio.mimwrite(savename, out_mean2)
    # print('save as', savename)
    

#%% validation
if False:
    filepath = 'C:\\mass_save\\auto_turboreg_run\\work2\\' + 'D12.oir_ij_reg.tif'
    img1 = io.imread(filepath) # 3 dimensions : frames x width x height

    # filepath = 'C:\\mass_save\\auto_turboreg_run\\work2\\' + 'D12.oir.tif_python_reg.tif'
    # out_mean_valid = io.imread(filepath) # 3 dimensions : frames x width x height
    
    for f in range(len(out_mean2.shape)):
        np.sum(np.abs(out_mean2[f,:,:] - img1[f,:,:]) > 1)
    
    
#%% validation2
if False:
    import pandas as pd
    
    loadpath = 'C:\\mass_save\\auto_turboreg_run\\work3\\' + 'ij.csv'
    ij = np.array(pd.read_csv(loadpath))[:,1:]
    print(ij.shape)
    
    loadpath = 'C:\\mass_save\\auto_turboreg_run\\work3\\' + 'python16.csv'
    python16 = np.array(pd.read_csv(loadpath))[:,1:]
    print(python16.shape)
    
    loadpath = 'C:\\mass_save\\auto_turboreg_run\\work3\\' + 'python32.csv'
    python32 = np.array(pd.read_csv(loadpath))[:,1:]
    print(python32.shape)
    
    
    for ROI in range(0, 10, 1):
        plt.figure()
        plt.plot(ij[:,ROI], label='ij')
        plt.plot(python16[:,ROI], label='ij')
        plt.plot(python32[:,ROI], label='ij')
        plt.legend()
    










































