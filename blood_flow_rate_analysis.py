# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:17:24 2022

@author: PC
"""


from skimage import io # pip install scikit-image
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from PIL import Image 
from tqdm import tqdm

# filepath = 'C:\\Temp\\Image5_ time series_ spon_line scan_Galvano.tif'
filepath = 'C:\\Temp\\Image6_ time series_ spon_line scan_Galvano.tif'

img0 = io.imread(filepath) # 3 dimensions : frames x width x height
img0 = np.array(img0)

width = img0.shape[1]
bins = int(width/2)

msbins = np.arange(0, img0.shape[0]-bins+1, bins, dtype=int)
peak_save = []
for row in tqdm(msbins):
    # row = 2300
    # plt.imshow(img0[row:row+width, :])
    # background = np.mean(img0[], axis=0)
    
    stdsave = []
    resolution = 0.1
    angle_list = np.arange(0, 180.0001, resolution)
    figsw = False
    for ro in angle_list:
        crop = img0[row:row+width, :]
        crop_img = Image.fromarray(crop - np.mean(crop, axis=0))
        rotate_img = crop_img.rotate(ro)
        rotate_img_array = np.array(rotate_img, dtype=float)
        rotate_img_array[rotate_img_array==0] = np.nan
        
        vix = np.where(np.isnan(np.mean(rotate_img_array, axis=0))==0)[0]
        
        if figsw:
            plt.figure()
            plt.imshow(np.array(rotate_img))
            plt.title(str(ro))
            
        mean_trace = np.nanmean(rotate_img_array[:,vix], axis=0)
        # plt.plot(np.array(mean_trace))
        # plt.title(str(ro))
        
        # tmp = np.abs(mean_trace[1:] - mean_trace[:-1])
        # mix = np.argsort(tmp)[:5]
        # formula1 = np.mean(tmp[mix])
        formula2 = np.std(mean_trace)
    
        stdsave.append(formula2)
        
    stdsave = np.array(stdsave)
    ws = int(round(31 * (0.1 / resolution)))
    smooth = np.convolve(stdsave, np.ones((ws,))/ws, mode='valid')
    stdsave_smooth = np.zeros(stdsave.shape)
    
    stdsave_smooth[int(ws/2):int(ws/2)+len(smooth)] = smooth
    
    # print(smooth.shape, np.array(stdsave).shape)
    
    if figsw:
        plt.figure()
        plt.plot(angle_list, stdsave)
        plt.plot(angle_list, stdsave_smooth)

    peak_save.append([np.max(stdsave_smooth), np.argmax(stdsave_smooth)])
    
peak_save = np.array(peak_save)
plt.figure()
plt.plot(msbins, peak_save[:,0])

mix = np.argsort(peak_save[:,0])[::-1][1:int(len(peak_save)*0.18)]
ix = np.array(peak_save[mix,1], dtype=int)
median_angle = np.median(angle_list[ix])

print(median_angle)

#%% vis

for i in range(len(mix)):
    row = msbins[mix[i]]
    otimized_angle = np.round(angle_list[int(peak_save[mix[i],1])], 2)
    
    crop = img0[row:row+width, :]
    
    line_matrix = np.zeros(crop.shape)
    lix = np.arange(10, crop.shape[1], 10, dtype=int)
    line_matrix[:,lix] = np.max(crop) * 2
    line_matrix = Image.fromarray(line_matrix)
    
    rotate_img = line_matrix.rotate(-otimized_angle)
    rotate_rgb = rotate_img.convert("RGB")
    
    crop_img = Image.fromarray(crop * 4)
    crop_rgb = crop_img.convert("RGB")
    crop_array = np.array(crop_rgb)
    
    crop_array[:,:,0] = np.array(rotate_rgb)[:,:,0]
    crop_array_img =  Image.fromarray(crop_array)
    
    fig = plt.figure()
    rows = 1
    cols = 2
    
    ax1 = fig.add_subplot(rows, cols, 2)
    ax1.imshow(crop_array_img)
    ax1.set_title('Merge ' + str(otimized_angle) + ' degree')
    ax1.axis("off")
    
    ax2 = fig.add_subplot(rows, cols, 1)
    ax2.imshow(Image.fromarray(crop))
    ax2.set_title('Original_# ' + str(i))
    ax2.axis("off")
    
















































