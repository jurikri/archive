# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:17:24 2022

@author: PC
"""

#%%

from skimage import io # pip install scikit-image (체크필요)
import matplotlib.pyplot as plt
import numpy as np
# import cv2
from PIL import Image 
from tqdm import tqdm

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale

# filepath = 'C:\\Temp\\Image5_ time series_ spon_line scan_Galvano.tif'

filepath = 'D:\\HR_bloodflow\\STZ\\3C1\\' + '3C1_Image2_line scan.bmp'


filepath = 'C:\\SynologyDrive\\worik in progress\\20220517 - flow calc\\' + '2022-03-24_TR_VNS_5m-line.tif'

img0 = io.imread(filepath) # 3 dimensions : frames x width x height
img0 = np.array(img0)[:, 110:370, 0]

plt.imshow(img0[:int(img0.shape[0]/20), :])

full_width = img0.shape[1]
width = int(full_width / np.sqrt(2))
hw = int(width/2)
bins = int(width/2)

msbins = np.arange(0, img0.shape[0]-bins+1, bins, dtype=int)
peak_save = []
for row in tqdm(msbins):
    # row = 2300
    # plt.imshow(img0[row:row+full_width, :])

    
    stdsave, stdsave2 = [], []
    resolution = 1
    angle_list = np.arange(0, 180.0001, resolution)
    figsw = False
    for ro in angle_list:
        crop = img0[row:row+full_width, :]
        crop_img = Image.fromarray(crop / np.mean(crop, axis=0) - 1)
        rotate_img = crop_img.rotate(ro)
        rotate_img_array = np.array(rotate_img, dtype=float)
        
        c = int(full_width/2)
        
        crop2 = rotate_img_array[c-hw : c+hw+1, c-hw: c+hw+1]
        if np.isnan(np.mean(crop2)): print('nan')
        
        theta = np.linspace(0., 180., max(crop2.shape), endpoint=False)
        sinogram = radon(crop2, theta=theta)

        score = np.std(sinogram) / np.std(crop2)
        stdsave.append(score)
        
        # method2
        
        mean_trace = np.nanmean(crop2, axis=0)
        formula2 = np.std(mean_trace)
        stdsave2.append(formula2)
        
        
        
    stdsave = np.array(stdsave)
    ws = int(round(31 * (0.1 / resolution)))
    smooth = np.convolve(stdsave, np.ones((ws,))/ws, mode='valid')
    stdsave_smooth = np.zeros(stdsave.shape)
    stdsave_smooth[int(ws/2):int(ws/2)+len(smooth)] = smooth

    plt.figure()
    plt.plot(stdsave[1:-1])
    plt.plot(stdsave_smooth[1:-1])
    
    
    stdsave2 = np.array(stdsave2)
    ws = int(round(31 * (0.1 / resolution)))
    smooth = np.convolve(stdsave2, np.ones((ws,))/ws, mode='valid')
    stdsave_smooth = np.zeros(stdsave2.shape)
    stdsave_smooth[int(ws/2):int(ws/2)+len(smooth)] = smooth
    
    plt.figure()
    plt.plot(stdsave2[1:-1])
    plt.plot(stdsave_smooth[1:-1])
    
    
    angle_list[np.argmax(stdsave_smooth)]
    
    
        
        rotate_img_array[rotate_img_array==0] = np.nan
        

    
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

# for i in range(len(mix)):
    
for i in range(200, 250):
    row = msbins[mix[i]]
    otimized_angle = np.round(angle_list[int(peak_save[mix[i],1])], 2)
    
    crop = img0[row:row+width, :]
    
    # crop = img0[1230:1500, :]
    # plt.imshow(crop)
    
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
    


#%% radon transform


import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale


theta = np.linspace(0., 180., max(crop.shape), endpoint=False)
sinogram = radon(crop, theta=theta)

plt.figure()
plt.imshow(crop)

plt.figure()
plt.imshow(sinogram)


resolution = 1
angle_list = np.arange(0, 180.0001, resolution)
figsw = False

stdsave, stdsave2 = [], []
for ro in angle_list:
    crop_img = Image.fromarray(crop - np.mean(crop, axis=0))
    rotate_img = crop_img.rotate(ro)
    rotate_img_array = np.array(rotate_img, dtype=float)
    rotate_img_array[rotate_img_array==0] = np.nan

    if figsw:
        plt.figure()
        plt.imshow(np.array(rotate_img))
        plt.title(str(ro))
        
    vix = np.where(np.isnan(np.mean(rotate_img_array, axis=0))==0)[0]
    theta = np.linspace(0., 180., max(rotate_img_array.shape), endpoint=False)
    sinogram = radon(rotate_img_array, theta=theta)

    score = np.nanstd(rotate_img_array) / np.nanstd(sinogram)
    stdsave.append(score)
    
    mean_trace = np.nanmean(rotate_img_array[:,vix], axis=0)
    formula2 = np.std(mean_trace)
    stdsave2.append(formula2)
    
    #%%
    
otimized_angle = 159

otimized_angle = 169

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


#%%
stdsave = []
resolution = 10
angle_list = np.arange(90, 200.0001, resolution)
for ro in angle_list:

    line_matrix = np.zeros((width,width))
    lix = np.arange(0, line_matrix.shape[1], 5, dtype=int)
    line_matrix[:,lix] = 10
    line_matrix = Image.fromarray(line_matrix)
    
    rotate_img = line_matrix.rotate(-ro)
    rotate_rgb = rotate_img.convert("RGB")
    
    # plt.imshow(rotate_rgb)
    
    crop = np.array(rotate_img)
    crop2 = crop / np.mean(crop) - 1

    # plt.figure()
    # plt.imshow(crop2)

    theta = np.linspace(0., 180., max(crop2.shape), endpoint=False)
    sinogram = radon(crop2, theta=theta)

    score = np.nanstd(sinogram) / np.nanstd(crop2)
    stdsave.append(score)

plt.figure()
plt.plot(angle_list, stdsave)





















