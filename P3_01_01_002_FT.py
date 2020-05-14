# load and evaluate a saved model
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
from keras.models import load_model, Model
from keras.utils import normalize
from sklearn.model_selection import train_test_split
from keras.layers import Input, MaxPooling2D, Conv2D, Cropping2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.utils import shuffle



#from P3_01_01_002 import norm_center, image_aug, generator

'Normalized the data'
def norm_center(imgs):
    imgs = normalize(imgs, axis=3)
    imgs -= 0.5
    return imgs

'Image Augmentation'
def image_aug(imgs, angles):
    image_flipped = np.fliplr(imgs)
    angles_flipped = -angles
    return image_flipped, angles_flipped

'Image Generator'
def generator(samples, batch_size=32):   
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
            correction = 0.1 # this is a parameter to tune
            
            for batch_sample in batch_samples:
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
                
                angle_center = float(batch_sample[3])
                angle_left = angle_center + correction
                angle_right = angle_center - correction
                
                image_center_aug, angle_center_aug = image_aug(image_center, angle_center)
                image_left_aug, angle_left_aug = image_aug(image_left, angle_left)
                image_right_aug, angle_right_aug = image_aug(image_right, angle_right)
                
                images_center.append(image_center)
                images_left.append(image_left)
                images_right.append(image_right)
                angles_center.append(angle_center)
                angles_left.append(angle_left)
                angles_right.append(angle_right)
                
                images_center_aug.append(image_center_aug)
                images_left_aug.append(image_left_aug)
                images_right_aug.append(image_right_aug)
                angles_center_aug.append(angle_center_aug)
                angles_left_aug.append(angle_left_aug)
                angles_right_aug.append(angle_right_aug)
            
            X_train = np.vstack((images_center, images_left, images_right, images_center_aug, images_left_aug, images_right_aug))
            
            # Normalized the Data
            X_train = norm_center(X_train)
            
            angles_center = np.array(angles_center)
            angles_left = np.array(angles_left)
            angles_right = np.array(angles_right)
            
            angles_center_aug = np.array(angles_center)
            angles_left_aug = np.array(angles_left)
            angles_right_aug = np.array(angles_right)
            
            y_train = np.hstack((angles_center, angles_left, angles_right, angles_center_aug, angles_left_aug, angles_right_aug))
            
            yield sklearn.utils.shuffle(X_train, y_train)

'load data'
paths = ['SimulationData/Smooth_Corner/']
    
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

print('\n\n\nsamples',samples)
            
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('len of train_samples is', len(train_samples))
print('len of validation_samples is', len(validation_samples))
   


'Set the parameter'
batch_size=32
dropout_ratio = 0.5
col, row, ch = 160, 320, 3 # Trimmed image format
input_shape = (col, row, ch)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


  
'load model'
model_0 = load_model('model_01_01_002.h5')
# summarize model.
print('Pre-train model is model_01_01_002.h5')
model_0.summary()
'modify model'
model_0.pop()
model_0.pop()
model_0.pop()
model_0.pop()
# freeze layer
for layer in model_0.layers:
        layer.trainable = False
print('Pre-train model after pop')
model_0.summary()
        
'NN architecture' 
inputs = Input(shape=input_shape)
y = model_0(inputs)
y = Dense(output_dim=1164,activation='relu')(y)
y = Dense(output_dim=100,activation='relu')(y)
y = Dense(output_dim=50,activation='relu')(y)
outputs = Dense(output_dim=1)(y)

model = Model(inputs, outputs, name="Modified Model")
model.summary()

# Create Models output folder
if os.path.exists("Models_Log/"):
    print('Model will be saved to /Models_Log')
else: 
    print('Models_Log will be created.')
    os.makedirs("Models_Log/")
my_callbacks = [
    EarlyStopping(min_delta=0.0001, patience=5),
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
