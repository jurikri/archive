# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:31:04 2021

@author: MSBak
"""

import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
fname = 's210629_1_Sham_M10'
file_path = 'G:\\paindecoder_khu_behavior\\PSL\\new\\' + fname + '\\'
file_names = os.listdir(file_path)

i = 0; 
name = file_names[0]
print(file_names)
numlist = [0,1,6,7,8,9,10,11,2,3,4,5]
numlist = [0,1,8,9,10,11,2,3,4,5]

for i, name in enumerate(file_names):
    src = os.path.join(file_path, name)
    dst = fname + '_' + str(numlist[i]) + '.avi'
    print(name, '->', dst)
    
    #%%
    
for i, name in enumerate(file_names):
    src = os.path.join(file_path, name)
    dst = fname + '_' + str(numlist[i]) + '.avi'
    print(name, '->', dst)
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)



