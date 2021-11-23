# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:11:44 2021

@author: MSBak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

path = 'A:\\data_tmp\\s21002121.xlsx'
dfsave = []; MAXSE = 20 # session이 몇갠지 모르는데 대충 20개 이하라는 뜻

for i in range(MAXSE):
    try:
        df = pd.read_excel(path, sheet_name=i, header=0)
        dfsave.append(np.array(df)[:,1:]) 
    except:
        break

for se in range(len(dfsave)):
    matrix = np.array(dfsave[se])
    
    mssave = []
    for ROI in range(matrix.shape[1]):
        
        mssignal = np.array(matrix[:,ROI])
        
        f0 = np.mean(np.sort(mssignal)[0:int(round(mssignal.shape[0]*0.3))])
        
        if True:
            plt.plot(mssignal)
            
        df = (mssignal - f0) / f0
        mssave.append(df)
    mssave = np.array(mssave)
    
    savepath = 'A:\\data_tmp\\' + 'df_' + str(se) + '.pickle'
    msdict = {'명성': mssave, '동철': '123'}
    with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(savepath, '저장되었습니다.')    

se = 1
msload = 'A:\\data_tmp\\' + 'df_' + str(se) + '.pickle'
with open(msload, 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
mssave = msdata_load['명성']
mssave2 = msdata_load['동철']




















