#!/usr/bin/env python
# coding: utf-8
'import the libiary and function'
import os
import csv
import re
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import sklearn
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.layers import Input, MaxPooling2D, Conv2D, Cropping2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.utils import shuffle


'Get the name of the samples to run'
paths = []
filenames = os.listdir('SimulationData/')
for filename in filenames:
    paths.append('SimulationData/'+filename+'/')
paths.pop(paths.index('SimulationData/Track_2/')) # Not include in training
paths.pop(paths.index('SimulationData/Smooth_Corner/')) # Not include in training
    
samples = []
for path in paths:
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the headers
        for line in reader:
            img_path = path+'IMG/'+line[0].split('/')[-1]
            if os.path.exists(img_path):
                line[0] = img_path
            else: line[0] = path+'IMG/'+line[0].split('\\')[-1]
                    
            img_path = path+'IMG/'+line[1].split('/')[-1]
            if os.path.exists(img_path):
                line[1] = img_path
            else: line[1] = path+'IMG/'+line[1].split('\\')[-1]
            
            img_path = path+'IMG/'+line[2].split('/')[-1]
            if os.path.exists(img_path):
                line[2] = img_path
            else: line[2] = path+'IMG/'+line[2].split('\\')[-1]
            
            samples.append(line)

'''
samples_dataframe = pd.DataFrame(samples)
samples_dataframe.columns = ['center','left','right','steering','throttle','brake','speed']
samples_dataframe[:3]
'''



'Normalized the data'
def norm_center(imgs):
    imgs = normalize(imgs, axis=3)
    imgs -= 0.5
    return imgs

'''
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
'''


'Image Augmentation'
def image_aug(imgs, angles):
    image_flipped = np.fliplr(imgs)
    angles_flipped = -angles
    return image_flipped, angles_flipped



'Create Training & Validation set'
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('len of train_samples is', len(train_samples))
print('len of validation_samples is', len(validation_samples))


angles = np.array([]) #For angle distribution study

def generator(samples, batch_size=32):
    global angles
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images_center = []
            images_center_aug = []
            images_left = []
            images_left_aug = []
            images_right = []
            images_right_aug = []
            angles_center = []
            angles_center_aug = []
            angles_left = []
            angles_left_aug = []
            angles_right = []
            angles_right_aug = []
            X_train = []
            y_train = []
            correction = 0.12 # this is a parameter to tune
            
            for batch_sample in batch_samples:
                # Read images in the batch samples
                image_center = plt.imread(batch_sample[0])
                if image_center.shape != (160, 320, 3):
                    image_center = cv2.resize(image_center, (160, 320, 3), interpolation = cv2.INTER_AREA)
                image_center = cv2.cvtColor(image_center, cv2.COLOR_RGB2YUV)
                
                
                image_left = plt.imread(batch_sample[1])
                if image_left.shape != (160, 320, 3):
                    image_left = cv2.resize(image_left, (160, 320, 3), interpolation = cv2.INTER_AREA)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2YUV)
                
                
                image_right = plt.imread(batch_sample[2])
                if image_right.shape != (160, 320, 3):
                    image_right = cv2.resize(image_right, (160, 320, 3), interpolation = cv2.INTER_AREA)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2YUV)
                
                # Read steering angles
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                
                
                images_center.append(image_center)
                images_left.append(image_left)
                images_right.append(image_right)
                angles_center.append(angle_center)
                angles_left.append(angle_left)
                angles_right.append(angle_right)
                
                
                # Augment the images & steering angles
                image_center_aug, angle_center_aug = image_aug(image_center, angle_center)
                images_center_aug.append(image_center_aug)
                angles_center_aug.append(angle_center_aug)
                    
                image_left_aug, angle_left_aug = image_aug(image_left, angle_left)
                images_left_aug.append(image_left_aug)
                angles_left_aug.append(angle_left_aug)
                    
                image_right_aug, angle_right_aug = image_aug(image_right, angle_right)
                images_right_aug.append(image_right_aug)
                angles_right_aug.append(angle_right_aug)
                
            # Normalized the Data
            X_train = np.vstack((images_center, images_left, images_right, images_center_aug, images_left_aug, images_right_aug))
            X_train = norm_center(X_train)
            
            angles_center = np.array(angles_center)
            angles_left = np.array(angles_left)
            angles_right = np.array(angles_right)
            
            angles_center_aug = np.array(angles_center_aug)
            angles_left_aug = np.array(angles_left_aug)
            angles_right_aug = np.array(angles_right_aug)
            
            y_train = np.hstack((angles_center, angles_left, angles_right, angles_center_aug, angles_left_aug, angles_right_aug))

            
            angles = np.hstack((angles, y_train))
            
            yield sklearn.utils.shuffle(X_train, y_train)

# Set the parameter
batch_size=32
dropout_ratio = 0.5


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


col, row, ch = 160, 320, 3 # Trimmed image format
input_shape=(col, row, ch)


'Model Architecture'
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

# Nvidia model
model.add(Cropping2D(input_shape=input_shape, 
                     cropping=((70,25),(0,0))))

model.add(Conv2D(filters=24, 
                 kernel_size=5, 
                 strides=(2, 2),
                 activation='relu', 
                 border_mode='valid'))
model.add(Dropout(p=dropout_ratio))

model.add(Conv2D(filters=36, 
                 kernel_size=5, 
                 strides=(2, 2),
                 activation='relu', 
                 border_mode='valid'))
model.add(Dropout(p=dropout_ratio))

model.add(Conv2D(filters=48, 
                 kernel_size=5, 
                 strides=(2, 2),
                 activation='relu', 
                 border_mode='valid'))
model.add(Dropout(p=dropout_ratio))

model.add(Conv2D(filters=64, 
                 kernel_size=3, 
                 strides=(1, 1),
                 activation='relu', 
                 border_mode='valid'))
model.add(Dropout(p=dropout_ratio))

model.add(Conv2D(filters=64, 
                 kernel_size=3, 
                 strides=(1, 1),
                 activation='relu', 
                 border_mode='valid'))
model.add(Dropout(p=dropout_ratio))

model.add(Flatten())

model.add(Dense(output_dim=1164,
                activation='relu'))

model.add(Dense(output_dim=100,
                activation='relu'))

model.add(Dense(output_dim=50,
                activation='relu'))

model.add(Dense(output_dim=1))

model.summary()

# Create Models output folder
if os.path.exists("Models_Log/"):
    print('Model will be saved to /Models_Log')
else: 
    print('Models_Log will be created.')
    os.makedirs("Models_Log/")
my_callbacks = [
    EarlyStopping(min_delta=0.0001, patience=3),
    ModelCheckpoint(filepath='Models_Log/model.{epoch:02d}-{val_loss:.4f}.h5'),
]

model.compile(loss='mse', optimizer='adam')


model.fit_generator(generator=train_generator,
                    samples_per_epoch=batch_size, 
                    nb_epoch=5, 
                    validation_data=validation_generator, 
                    nb_val_samples=batch_size, 
                    callbacks=my_callbacks,
                    verbose=1)

model.save('model.h5')

print('Data feed to training is')
print(pd.DataFrame(paths))
print('Model will be saved as model.h5')

'Angles Distribution'
num_bins = 20
bins = (np.arange(num_bins+2)-(num_bins+1)/2)/10
x_label = (np.arange(num_bins+1)-num_bins/2)/10
num_samples_bin, _, _ = plt.hist(angles, bins=bins , rwidth=0.5)

print(pd.DataFrame((x_label,num_samples_bin)))




