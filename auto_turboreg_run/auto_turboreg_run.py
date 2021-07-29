# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:05:21 2021

@author: MSBak
"""


#%% oir -> tif
import os
path = 'C:\\mass_save\\auto_turboreg_run\\work'

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




#%% turboreg and save
from pystackreg import StackReg # pip install pystackreg
from skimage import io # pip install scikit-image (체크필요)
import imageio # pip install imageio
import numpy as np
import matplotlib.pyplot as plt
import time

# import javabridge
# import bioformats # pip install python-bioformats
# javabridge.start_vm(class_path=bioformats.JARS)

file_list = os.listdir(path)
listsave = []
for i in range(len(file_list)):
    if file_list[i].endswith('.tif'):
        filepath = path + '\\' + file_list[i]
        listsave.append(filepath) 
print('total sample #', len(listsave))
    
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
    # print(np.mean(np.std(out_mean, axis=0)))
    out_mean2 = np.array(out_mean, dtype='uint16')
    imageio.mimwrite(filepath + '_reg.tif', out_mean2)
    print('save as', filepath + '_reg.tif')

#%% validation
if False:
    filepath = path + '\\' + 'Image5_Sham_D10_M10_reg.tif'
    img1 = io.imread(filepath) # 3 dimensions : frames x width x height
    print(np.mean(np.std(img1, axis=0)))
    
    plt.figure()
    plt.plot(np.std(img0, axis=0).flatten(),alpha=0.7, label='original')
    plt.plot(np.std(img1, axis=0).flatten(),alpha=0.5, label='image J')
    plt.legend()
    
    plt.figure()
    plt.plot(np.std(out_mean, axis=0).flatten(),alpha=0.7, label='python')
    plt.plot(np.std(img1, axis=0).flatten(),alpha=0.5, label='image J')
    plt.legend()






