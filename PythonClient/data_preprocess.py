#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import os
import shutil
from pathlib import Path
import math

from queue import Queue

df=pd.read_csv("D:\carla\CARLA_0.8.4\Data\dataCopy.csv")

steer = df.iloc[:,8]
throttle = df.iloc[:,9]
steer_frames = df.iloc[:,11]
num_bins = 25
hist, bins = np.histogram(steer,num_bins)
center = (bins[:-1]+bins[1:])*0.5
plt.bar(center, hist, width = 0.05)

samples_per_bins = 1000

## Processing the images with steering angle equal to zero

df_zero_steering = df.loc[df["steer"]==0]
df_zero_processed_data = pd.DataFrame(columns=['pos_x','pos_y','lane_x','lane_y','lane_z','pitch','yaw','roll','steer','throttle','frame_count','episode'])

    
main_dir = "D:\carla\CARLA_0.8.4\PythonClient\_out0\episode_0001"

image_folder = os.path.join(main_dir,r"CameraRGBCopy")

subfolder1 = os.path.join(main_dir, "tempFolder")

###################################################################################################

## Data Augmentation for zero steering angles

# check if they already exits to prevent error
if not os.path.exists(subfolder1):
    os.makedirs(subfolder1)

# Initializing a queue
q = Queue(maxsize = df_zero_steering.shape[0])
    
for i in range(df_zero_steering.shape[0]):
    a = df_zero_steering.iloc[i,:]
    q.put(a)

# move images with zero steering angle to another temporary folder
i=0 
a=q.get()
size_queue = q.qsize()
frame = str(int(a["frame_count"]))
for file in os.listdir(image_folder):
    print(frame)
    print(file)
    print(frame in file)
    if ((frame in file) and i < size_queue):
        print(frame)
        print("file to be copied : ",file)
        fileCopy_path = os.path.join(image_folder,file)
        file_to_copy = Path(fileCopy_path)
#        if file_to_copy.exists():
#        print(file)
#        print("exists")
        shutil.copy(fileCopy_path,subfolder1) 
        print("copied file")
        i = i + 1
        a=q.get()
        frame = str(int(a["frame_count"]))


def augment_brightness_camera_images(image):
     image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
     image1 = np.array(image1, dtype = np.float64)
     random_bright = .5+np.random.uniform()
     image1[:,:,2] = image1[:,:,2]*random_bright
     image1[:,:,2][image1[:,:,2]>255]  = 255
     image1 = np.array(image1, dtype = np.uint8)
     image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
     return image1

def trans_image(image,steer,trans_range):
     # Translation
     tr_x = trans_range*np.random.uniform()-trans_range/2
     steer_ang = steer + tr_x/trans_range*2*.2
     tr_y = 40*np.random.uniform()-40/2
     #tr_y = 0
     Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
     if steer_ang > 1:
         steer_ang = 1.0
     if steer_ang < -1:
         steer_ang = -1
     cols, rows = image.shape[0], image.shape[1]
     image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
#     plt.subplot(131)
#     print("image in trans image")
#     plt.imshow(image)
     return image_tr,steer_ang
 
def preprocessImage(image):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    new_size_col = 128
    new_size_row = 128
    image = cv2.resize(image,(new_size_col,new_size_row),interpolation=cv2.INTER_AREA)    
    norm_img = np.zeros((image.shape[0],image.shape[1]))
    final_img = cv2.normalize(image,  norm_img, 0, 255, cv2.NORM_MINMAX)
#    image = image/255.#-.5
    return final_img


def preprocess_image_file_train(image,y_steer):
#    i_lrc = np.random.randint(3)
#    if (i_lrc == 0):
#        path_file = line_data['left'][0].strip()
#        shift_ang = .25
#    if (i_lrc == 1):
#        path_file = line_data['center'][0].strip()
#        shift_ang = 0.
#    if (i_lrc == 2):
#        path_file = line_data['right'][0].strip()
#        shift_ang = -.25
#    y_steer = line_data['steer_sm'][0] + shift_ang
#    image = cv2.imread(subfolder1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_tr,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image_tr)
#    plt.subplot(132)
#    print("image in preprocess")
#    plt.imshow(image)
    image = preprocessImage(image)
#    image = np.array(image)
#    ind_flip = np.random.randint(2)
#    if ind_flip==0:
#        image = cv2.flip(image,1)
#        y_steer = -y_steer
    
    return image,y_steer

subfolder2 = os.path.join(main_dir, "tempFolder_zero_processeddata")

# check if they already exits to prevent error
if not os.path.exists(subfolder2):
    os.makedirs(subfolder2)
    
count=0
i = 0
for file in os.listdir(subfolder1):
    path = os.path.join(subfolder1,file)
    image = cv2.imread(path)
#    img = mpimg.imread(image)
    plt.imshow(image)
#    plt.show()
    a = df_zero_steering.iloc[i,:]
    y_steer = a["steer"]
    image_processed,y_steer_processed = preprocess_image_file_train(image,y_steer)
    plt.subplot(133)
    print("image in for loop")
    plt.imshow(image_processed)
    filename, ext = file.split(".")
    filename = str(count + 9535)
    a["steer"] = y_steer_processed
    a["frame_count"] = count + 9535
    df_zero_processed_data = df_zero_processed_data.append(a)
    print(i)
    count = count + 1
    i = i+1
    path1 = os.path.join(subfolder2,filename+"_processed."+ext)
    cv2.imwrite(path1, image_processed)
    

    
###########################################################################################################\

## Processing the images with steering angle greater than zero
    
def flip_images(image,y_steer):
    ind_flip = np.random.randint(1)
#    if ind_flip==0:
    image = cv2.flip(image,1)
    y_steer_process = -y_steer
    return image,y_steer_process
    
df_nonzero_steering = df[(df['steer'] == 1) | (df['steer'] == -1) ]
df_nonzero_flip_data = pd.DataFrame(columns=['pos_x','pos_y','lane_x','lane_y','lane_z','pitch','yaw','roll','steer','throttle','frame_count','episode'])
df_nonzero_processed_data = pd.DataFrame(columns=['pos_x','pos_y','lane_x','lane_y','lane_z','pitch','yaw','roll','steer','throttle','frame_count','episode'])


copy_to = os.path.join(main_dir,"tempFolder_images_nonzero")

subfolder3 = os.path.join(main_dir, "tempFolder_processed_nonzero_flip")
    
subfolder4 = os.path.join(main_dir, "tempFolder_processed_nonzero")

# check if they already exits to prevent error
if not os.path.exists(copy_to):
    os.makedirs(copy_to)
    
if not os.path.exists(subfolder3):
    os.makedirs(subfolder3)

if not os.path.exists(subfolder4):
    os.makedirs(subfolder4)

# move images with non zero steering angle to another temporary folder
# Initializing a queue
q = Queue(maxsize = df_nonzero_steering.shape[0])
    
for i in range(df_nonzero_steering.shape[0]):
    a = df_nonzero_steering.iloc[i,:]
    q.put(a)

    
i=0 
a=q.get()
size_queue = q.qsize()
frame = str(int(a["frame_count"]))

for file in os.listdir(image_folder):
    print(frame)
    print(file)
    print(frame in file)
    if file == "217":
        break
    if ((frame in file) and i < size_queue):
        print(frame)
        print("file to be copied : ",file)
        fileCopy_path = os.path.join(image_folder,file)
        file_to_copy = Path(fileCopy_path)
#        if file_to_copy.exists():
#        print(file)
#        print("exists")
        shutil.copy(fileCopy_path,copy_to) 
        print("copied file")
        i = i + 1
        a=q.get()
        frame = str(int(a["frame_count"]))

i = 0        
for file in os.listdir(copy_to):
    path = os.path.join(copy_to,file)
    image = cv2.imread(path)
    a = df_nonzero_steering.iloc[i,:]
    y_steer = a["steer"]
    image_processed,y_steer_processed = flip_images(image,y_steer)
    filename, ext = file.split(".")
#    filename = str(count + int(a["frame_count"]))
#    filename = str(int(a["frame_count"]))
    filename = str(count + 9535)
    print("filename : ",filename)
    path1 = os.path.join(subfolder3,filename+"_processed."+ext)
    a["steer"] = y_steer_processed
    a["frame_count"] = count + 9535
    df_nonzero_flip_data = df_nonzero_flip_data.append(a)
    count = count + 1
    i = i + 1
    print(count)
    cv2.imwrite(path1, image_processed)
    print("image copied")
    
   
i = 0
for file in os.listdir(copy_to):
    path = os.path.join(copy_to,file)
    image = cv2.imread(path)
#    img = mpimg.imread(image)
    plt.imshow(image)
#    plt.show()
    a = df_nonzero_steering.iloc[i,:]
    y_steer = a["steer"]
    image_processed,y_steer_processed = preprocess_image_file_train(image,y_steer)
    plt.subplot(133)
    print("image in for loop")
    plt.imshow(image_processed)
    filename, ext = file.split(".")
    filename = str(count + 9535)
#    filename = str(int(a["frame_count"]))
    path1 = os.path.join(subfolder4,filename+"_processed."+ext)
    a["steer"] = y_steer_processed
    a["frame_count"] = count + 9535
    df_nonzero_processed_data = df_nonzero_processed_data.append(a)
    print(i)
    count = count + 1
    print(count)
    i = i + 1
    cv2.imwrite(path1, image_processed)
    
    
    
##################################################################################################
## Adding the images of processed zero and non-zero data

df=pd.read_csv("D:\carla\CARLA_0.8.4\Data\dataCopy.csv")

#df_zero_steering = df.loc[df["steer"]==0]

df = df.append(df_zero_processed_data)
for file in os.listdir(subfolder2):
    path = os.path.join(subfolder2,file)
    shutil.copy(path,image_folder)
    print(file)
#    
#
df = df.append(df_nonzero_flip_data)
for file in os.listdir(subfolder3):
    path = os.path.join(subfolder3,file)
    shutil.copy(path,image_folder)
    print(file)
    
df = df.append(df_nonzero_processed_data)
for file in os.listdir(subfolder4):
    path = os.path.join(subfolder4,file)
    shutil.copy(path,image_folder)
    print(file)

df.to_csv(r'D:\carla\CARLA_0.8.4\Data\train.csv',index=False)
##################################################################################################
## Remove some zero steering data

## remove leading zeros from filename of images
for file in os.listdir(image_folder):
    print(file)
    new = file.lstrip("0")
    print("new : ",new)
    old_name = os.path.join(image_folder,file)
    new_name = os.path.join(image_folder,new)
    os.rename(old_name,new_name)

df_train_zero=pd.read_csv(r'D:\carla\CARLA_0.8.4\Data\train.csv')
df_train_zero_steering = df_train_zero.loc[df_train_zero["steer"]==0]

df_remove = pd.DataFrame(columns=['pos_x','pos_y','lane_x','lane_y','lane_z','pitch','yaw','roll','steer','throttle','frame_count','episode'])   
##df_remove = df_remove.append(df_zero_steering.iloc[:51,:])
df_remove = df_remove.append(df_train_zero_steering)
### Shuffle the data
#
df_remove = df_remove.sample(frac = 1) 
samples_to_be_removed = 7000
#
df_remove_list = pd.DataFrame(columns=['pos_x','pos_y','lane_x','lane_y','lane_z','pitch','yaw','roll','steer','throttle','frame_count','episode'])   
df_remove_list = df_remove_list.append(df_remove.iloc[:samples_to_be_removed,:])
#remove_folder = os.path.join(main_dir,"remove")
remove_folder = os.path.join(image_folder)
#
#
i = 0
#for file in os.listdir(remove_folder):
#    index = -4
#    image = os.path.join(remove_folder,file)
#    print(image)
#    filename,ext = file.split(".")
##    if ("processed" in filename):
##        break
#    frame = int(filename)
#    index = (df_remove_list.index[df_remove_list["frame_count"] == frame])
#    print(index)
#    if index.size > 0:
#        df_train_zero = df_train_zero.drop(index[0])
#        os.remove(image)
#    i = i + 1

for i in range(df_remove_list.shape[0]):
    a = df_remove_list.iloc[i,:]
    frame = (int(a["frame_count"]))
    file_remove = os.path.join(remove_folder,str(frame)+".png")
    print(file_remove)
    df_train_zero = df_train_zero.drop(frame)
    os.remove(file_remove)
    

df_train_zero.to_csv('data_train.csv', index=False)
########################################################################################
### see the final data distribution
##
data_train=pd.read_csv('data_train.csv')

steer = data_train.iloc[:,8]
throttle = data_train.iloc[:,9]
steer_frames = data_train.iloc[:,11]
num_bins = 25
hist, bins = np.histogram(steer,num_bins)
center = (bins[:-1]+bins[1:])*0.5
plt.bar(center, hist, width = 0.05)