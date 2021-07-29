# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:05:21 2021

@author: MSBak
"""


from pystackreg import StackReg # pip install pystackreg
from skimage import io # pip install scikit-image (체크필요)

import numpy as np
import matplotlib.pyplot as plt
import os

import javabridge
# https://www.youtube.com/watch?v=IJ-PJbvJBGs
# conda install -c conda-forge javabridge
import bioformats # pip install python-bioformats
javabridge.start_vm(class_path=bioformats.JARS)


path = 'C:\\mass_save\\auto_turboreg_run\\'
# C: <- should be typed mannualy. Do not copy and paste!
file_list = os.listdir(path)

i = 3
for i in range(len(file_list)):
    if file_list[i].endswith('.oir'):
        print(path+file_list[i])
        
        imgs = []
        t = 0
        while True:
            try:
                print(t)
                img = bioformats.load_image(path + file_list[i], t=t)  # Image data
                imgs.append(img[:,:,1])
                t += 1
            except: break
        
        imgs = np.array(imgs)
        print('imgs.shape', imgs.shape)

    
path = 'C:\\mass_save\\auto_turboreg_run\\'
filename = 'Image6_D11_M_10mig.oir'
img0 = io.imread() # 3 dimensions : frames x width x height

img0 = imgs
print(np.mean(np.std(img0, axis=0)))

sr = StackReg(StackReg.RIGID_BODY)

# register to mean image
out_mean = sr.register_transform_stack(img0, reference='mean')
print(np.mean(np.std(out_mean, axis=0)))


# register to mean of first 10 images

path = 'C:\\mass_save\\auto_turboreg_run\\'
filename = 'Registered2.tif'
aligned = io.imread(path + filename) # 3 dimensions : frames x width x height
print(aligned.shape)
print(np.mean(np.std(aligned, axis=0)))

row = 150 
col = 150

plt.figure()
plt.plot(np.mean(np.mean(img0[:,row:row+20,col:col+20], axis=1), axis=0))
plt.plot(np.mean(np.mean(out_mean[:,row:row+20,col:col+20], axis=1), axis=0))
plt.plot(np.mean(np.mean(aligned[:,row:row+20,col:col+20], axis=1), axis=0))


print('original', np.mean(np.std(img0[:,row:row+20,col:col+20], axis=0)))
print('python', np.mean(np.std(out_mean[:,row:row+20,col:col+20], axis=0)))
print('image J', np.mean(np.std(aligned[:,row:row+20,col:col+20], axis=0)))








