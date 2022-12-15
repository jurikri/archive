# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 14:39:51 2022

@author: msbak
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('C:\\mscode\\files_tmp\\sunflow in focus.png')
img2 = np.array(img)
img3 = np.array(img2[:,:,[2,1,0]], dtype=np.uint8)

plt.imshow(img3)


n1 = 25
kernel = np.ones((n1,n1), np.float32) / n1**2
dst = cv2.filter2D(img3, -1, kernel)



plt.subplot(121),plt.imshow(img3),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])

savepath = 'C:\\mscode\\files_tmp\\sunflow out focus.png'
plt.savefig(savepath, dpi=200)
