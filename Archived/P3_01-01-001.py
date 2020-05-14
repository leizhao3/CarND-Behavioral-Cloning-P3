#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Behavioral Cloning
# 
# 
# In this project, neural network will be trained to drive the vehicle in a simulator using human steering angle as input. 

# ---
# ## Step 0: Load The Data
# 
# Keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
# 
# ### Load The Data

# In[4]:


import os
import csv
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the headers
    for line in reader:
        samples.append(line)
samples = samples[:32] # only use first 32 sample for trail

samples_dataframe = pd.DataFrame(samples)
samples_dataframe.columns = ['center','left','right','steering','throttle','brake','speed']

samples_dataframe[:3]


# ---
# ## Step 1: Data Preprocessing & Augmentation
# **ATTENTION**: input image is in RGB???? , need to do ??? 
# 

# ### Cropping Image

# In[45]:


# Display Picture Processing
import random

def image_crop(imgs, y=50, x=0, h=90, w=320):
    imgs = np.array(imgs)
    
    if len(imgs.shape) == 4:
        imgs_out = imgs[:,y:y+h,x:x+w,:]
    else: imgs_out = imgs[y:y+h,x:x+w,:]
    
    return imgs_out
            

# Display 1 set of images (center, left, right)
f, a = plt.subplots(2,3,figsize=(12,4))
index = random.randint(0,len(samples))
title_array = ['center','left','right']
images = []

for i in range(3):
    path = 'data/IMG/'+samples[index][i].split('/')[-1]
    image = plt.imread(path)
    images.append(image)
    
    a[0][i].imshow(image)
    a[0][i].set_title('%s image before cropping'%title_array[i])
    
    image = image_crop(image)
    a[1][i].imshow(image)
    a[1][i].set_title('%s image after cropping'%title_array[i])

images = np.array(images)

f.savefig('test_images_output/Before_After_ImgCropping.jpg')
            


# ### Data Preprocessing 

# In[47]:


'Normalized the data'
def norm_center(imgs):
    imgs = normalize(imgs, axis=3)
    imgs -= 0.5
    return imgs

means = images.mean(axis=(1,2), dtype='float64')
stds = images.std(axis=(1,2), dtype='float64')
print('Data Before Normalizing & Center')
print('Means: %s' % means[:3])
print('stds: %s' % stds[:3])
print('Mins: %s \nMaxs: %s' % (images.min(axis=(1,2))[:3], images.max(axis=(1,2))[:3]))

images = norm_center(images)
means = images.mean(axis=(1,2), dtype='float64')
print('\n\nData After Normalizing & Center')
print('Means: %s' % means[:3])
print('Mins: %s \nMaxs: %s' % (images.min(axis=(1,2))[:3], images.max(axis=(1,2))[:3]))


# ### Data Augmentation
# ```python
# import numpy as np
# image_flipped = np.fliplr(image)
# measurement_flipped = -measurement
# ```

# In[ ]:





# ## NN Architecture
# Following Nvidia paper

# In[1]:


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('len of train_samples is', len(train_samples))
print('len of validation_samples is', len(validation_samples))

import sklearn
from keras.utils import normalize

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images_center = []
            images_left = []
            images_right = []
            angles_center = []
            angles_left = []
            angles_right = []
            X_train = []
            y_train = []
            correction = 0.2 # this is a parameter to tune
            
            for batch_sample in batch_samples:
                image_center = plt.imread('data/IMG/'+batch_sample[0].split('/')[-1])
                image_left = plt.imread('data/IMG/'+batch_sample[1].split('/')[-1])
                image_right = plt.imread('data/IMG/'+batch_sample[2].split('/')[-1])
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                images_center.append(image_center)
                images_left.append(image_left)
                images_right.append(image_right)
                angles_center.append(angle_center)
                angles_left.append(angle_left)
                angles_right.append(angle_right)
            
            # Cropping the Image
            images_center = image_crop(images_center)
            images_left = image_crop(images_left)
            images_right = image_crop(images_right)
            
            X_train = np.vstack((images_center, images_left, images_right))
            
            # Normalized the Data
            X_train = norm_center(X_train)
            
            angles_center = np.array(angles_center)
            angles_left = np.array(angles_left)
            angles_right = np.array(angles_right)
            
            y_train = np.hstack((angles_center,angles_left,angles_right))
            
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=2

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# In[2]:


import tensorflow as tf
import math
from keras.layers import Input, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from sklearn.utils import shuffle
from keras.layers.normalization import BatchNormalization



col, row, ch = 90, 320, 3 # Trimmed image format
input_shape=(col, row, ch)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

'''
# Nvidia model
model.add(Convolution2D(input_shape=input_shape,
                        nb_filter=24, nb_row=5, nb_col=5, 
                        activation='relu', 
                        border_mode='valid'))

model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, 
                        activation='relu', 
                        border_mode='valid'))

model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, 
                        activation='relu', 
                        border_mode='valid'))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, 
                        activation='relu', 
                        border_mode='valid'))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, 
                        activation='relu', 
                        border_mode='valid'))

model.add(Flatten())

model.add(Dense(output_dim=1164,
                activation='relu'))

model.add(Dense(output_dim=100,
                activation='relu'))

model.add(Dense(output_dim=50,
                activation='relu'))

model.add(Dense(output_dim=1))
'''



# Tutorial Model
model.add(Convolution2D(input_shape=input_shape, 
                        nb_filter=6, nb_row=5, nb_col=55, 
                        activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(nb_filter=6, nb_row=5, nb_col=55, 
                        activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator=train_generator,
                    samples_per_epoch=batch_size, 
                    nb_epoch=5, 
                    validation_data=validation_generator, 
                    nb_val_samples=batch_size, 
                    verbose=1)

'''
fit_generator(self, 
              generator, 
              samples_per_epoch, 
              nb_epoch, 
              verbose=1, 
              callbacks=None, 
              validation_data=None, 
              nb_val_samples=None, 
              class_weight=None, 
              max_q_size=10, 
              nb_worker=1, 
              pickle_safe=False, 
              initial_epoch=0)
'''


# # Training

# In[ ]:





# 
