# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:59:25 2022

@author: PC
"""

import hdf5storage
import matplotlib.pyplot as plt
import numpy as np

loadpath = 'C:\\Users\\PC\\Downloads\\s0518-2_5HT_1-3.avi.mat'
df = hdf5storage.loadmat(loadpath)
diffplot = df['msdiff_gauss']


filepath = ''
filename = ''


savename = filepath + filename + '.pickle'
mssave = diffplot[0,:]
with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)
    print(savename, '저장되었습니다.') 