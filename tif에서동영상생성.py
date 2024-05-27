# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:02:31 2024

@author: PC
"""

import imageio
import cv2
import numpy as np
import imageio_ffmpeg as ffmpeg

    
def normalize_to_uint8(data):
    # Normalize the data to the range [0, 255]
    data_min = data.min()
    data_max = data.max()
    data_normalized = 255 * (data - data_min) / (data_max - data_min)
    return data_normalized.astype(np.uint8)
    
def create_sample_video(tif_file_path, video_file_path, fps=15):
    tif = imageio.volread(tif_file_path)
    depth, height, width = tif.shape
    
    # Create a sample 3D numpy array with random values
    # sample_array = np.random.randint(0, 256, tif.shape, dtype=np.uint8)
    sample_array = normalize_to_uint8(tif)
    # sample_array = np.array(tif, dtype=np.uint8)
    
    # Define the codec and create VideoWriter object
    writer = imageio.get_writer(video_file_path, fps=fps, codec='libx264', quality=8)    
    # Write frames to the video file
    for i in range(depth):
        frame = sample_array[i]
        writer.append_data(frame)
    
    # Close the writer
    writer.close()
    
    return video_file_path

# Example usage
tif_file_path = r'C:\mscode\private\개인서류\지원서류\20180518_1372_1reg.tif'
video_file_path = r'C:\\mscode\\private\개인서류\\지원서류\\' + 'output_video.mp4'

convert_tif_to_video(tif_file_path, video_file_path)
