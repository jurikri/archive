
import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

path1 = 'C:\\Users\\MSBak\\Downloads\\behavior-20220502T064636Z-001\\behavior\\'

file_list = os.listdir(path1)

for fname in file_list:
    path2 = path1 + fname + '\\'
    file_list2 = os.listdir(path2)
    
    for i in range(len(file_list2)):
        if file_list2[i][-4:] == '.mat':
            df = hdf5storage.loadmat(path2 + file_list2[i])
            diffplot = df['msdiff_gauss']
            mssave = diffplot[0,:]

            savename = path2 + file_list2[i][:-7] + 'pickle'
            with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)
                print(savename, '저장되었습니다.') 
                
#%%
path1 = 'C:\\Users\\MSBak\\Downloads\\behavior-20220502T064636Z-001\\behavior\\s0518-2\\'
file_list = os.listdir(path1)
for i in range(len(file_list)):
    if file_list[i][-7:] == '.pickle':
        print(file_list[i])

    with open(path1 + file_list[i], 'rb') as f:  # Python 3: open(..., 'rb')
        repeat_save = pickle.load(f)
        
    plt.figure()
    plt.plot(repeat_save)
    
    import sys; sys.exit()
        
